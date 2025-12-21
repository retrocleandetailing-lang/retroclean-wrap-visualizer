import os
import io
import base64
import time
import urllib.request
from typing import Literal, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageChops, ImageFilter

import replicate
from replicate.exceptions import ReplicateError

# One click = 2 predictions (segmentation + inpaint). Keep cooldown conservative.
COOLDOWN_SECONDS = 45

Angle = Literal["front_3q", "side", "rear_3q"]
Finish = Literal["gloss", "satin", "matte"]

COLOR_MAP = {
    "nardo_grey": "Nardo Grey",
    "satin_black": "Satin Black",
    "gloss_black": "Gloss Black",
    "miami_blue": "Miami Blue",
    "british_racing_green": "British Racing Green",
    "ruby_red": "Deep Ruby Red",
    "pearl_white": "Pearl White",
}

# Stable Diffusion inpaint likes fixed widths/heights (multiples supported by the model).
# We'll hard-resize to one of these, preserving aspect ratio via letterbox.
# Choose 768 as a good quality/cost tradeoff.
INPAINT_SIZE = 768  # must be in {64,128,...,1024} per your 422 error
REPLICATE_BURST_SLEEP = 12  # helps avoid "burst of 1" throttling between 2 predictions


app = FastAPI(title="RetroClean Wrap Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://retrocleandetailing.com",
        "https://www.retrocleandetailing.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def img_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def download_image(url: str, timeout: int = 30) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data))


def normalize_replicate_output(out) -> Optional[str]:
    """
    Replicate can return:
      - "https://..." string
      - [ "https://..." ]
      - dict with common keys
      - file-like objects (has .url)
    This returns a single URL/data string or None.
    """
    if out is None:
        return None

    if isinstance(out, str):
        return out

    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "url"):
            try:
                return str(first.url)
            except Exception:
                return str(first)
        return str(first)

    if isinstance(out, dict):
        for k in ("output", "image", "images", "result", "url"):
            v = out.get(k)
            if not v:
                continue
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v:
                return str(v[0])
            if hasattr(v, "url"):
                try:
                    return str(v.url)
                except Exception:
                    return str(v)
            return str(v)

    try:
        s = str(out)
        return s if s else None
    except Exception:
        return None


def build_prompt(color_name: str, finish: Finish, angle: Angle) -> str:
    angle_hint = {
        "front_3q": "front three-quarter angle",
        "side": "side profile angle",
        "rear_3q": "rear three-quarter angle",
    }[angle]

    finish_text = {
        "gloss": "high-gloss vinyl wrap with clearcoat-like highlights",
        "satin": "satin vinyl wrap with soft reflections",
        "matte": "matte vinyl wrap with diffused reflections",
    }[finish]

    # Metallic flake only for gloss (subtle!)
    flake = ""
    if finish == "gloss":
        flake = (
            "Add subtle fine metallic flake micro-sparkle ONLY in bright highlights "
            "(not glittery; premium OEM-like metallic). "
        )

    return (
        f"Photorealistic wrap preview of the SAME car photo. "
        f"Keep the same exact camera perspective ({angle_hint}), same lighting, same background. "
        f"ONLY change the painted body panels (doors, fenders, hood, quarter panels, bumpers) "
        f"to {color_name} {finish_text}. "
        f"{flake}"
        f"Preserve panel gaps and seam lines (door lines, hood lines, trunk lines) clearly. "
        f"Do NOT change wheels, tires, brakes, headlights, taillights, windows, windshield, trim, badges, grille, or interior. "
        f"Do NOT change the environment. "
        f"Maintain original reflections shape and realism; only recolor the painted panels."
    )


def negative_prompt() -> str:
    return (
        "different car, different angle, changed body kit, changed wheels, changed background, "
        "cartoon, CGI, illustration, anime, warped panels, melted shapes, extra parts, "
        "text, watermark, logo, blurry, low quality"
    )


def letterbox_to_square(img: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    """
    Resize while preserving aspect ratio, then pad to size x size.
    This ensures width/height are valid for models that require fixed dims.
    """
    img = img.copy()
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), fill)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(img, (x, y))
    return canvas


def letterbox_mask_to_square(mask_l: Image.Image, size: int) -> Image.Image:
    """
    Same as letterbox_to_square, but for L masks.
    White = inpaint region, black = preserve.
    """
    mask_l = mask_l.copy().convert("L")
    w, h = mask_l.size
    if w == 0 or h == 0:
        raise ValueError("Invalid mask size")

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    mask_l = mask_l.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("L", (size, size), 0)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(mask_l, (x, y))
    return canvas


def refine_mask_reduce_windows_wheels(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Heuristic cleanup: remove very dark + low-saturation pixels from the mask area
    to reduce windows/wheels getting recolored.
    """
    mask = mask_l.resize(original_rgb.size).convert("L")
    mask = ImageOps.autocontrast(mask)

    hsv = original_rgb.convert("HSV")
    _, s, v = hsv.split()

    v_ok = v.point(lambda px: 255 if px > 55 else 0)
    s_ok = s.point(lambda px: 255 if px > 35 else 0)

    keep_if = ImageChops.lighter(v_ok, s_ok)
    cleaned = ImageChops.multiply(mask, keep_if)

    cleaned = cleaned.filter(ImageFilter.GaussianBlur(radius=1.2))
    cleaned = cleaned.point(lambda px: 255 if px > 110 else 0)

    return cleaned


@app.get("/")
def root():
    return {"message": "RetroClean Wrap Visualizer API. Use /health or POST /render."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.55),
):
    # Server-side cooldown
    now = time.time()
    last = getattr(app.state, "last_call_ts", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait ~{COOLDOWN_SECONDS}s and try again.",
        )
    app.state.last_call_ts = now

    if color not in COLOR_MAP:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    # Load image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = pil.size
        if w < 900 or h < 900:
            raise HTTPException(status_code=400, detail="Photo too small. Use at least ~900px on each side.")
        # Keep uploads reasonable
        if w > 3000 or h > 3000:
            pil.thumbnail((3000, 3000))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Env vars
    token = os.getenv("REPLICATE_API_TOKEN")
    seg_version = os.getenv("REPLICATE_SEG_VERSION")
    img_version = os.getenv("REPLICATE_IMG_VERSION")

    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    if not seg_version or not img_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION or REPLICATE_IMG_VERSION on server.")

    os.environ["REPLICATE_API_TOKEN"] = token

    # 1) Segmentation (run on the original aspect image)
    original_data_url = img_to_data_url(pil, fmt="PNG")

    try:
        seg_out = replicate.run(
            seg_version,
            input={
                "image": original_data_url,
                "text_prompt": "car body panels",
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (segmentation): {str(e)}")

    mask_url = normalize_replicate_output(seg_out)
    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")

    # Download mask and clean it
    try:
        mask_img = download_image(mask_url).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    mask_img = mask_img.resize(pil.size).convert("L")
    mask_img = ImageOps.autocontrast(mask_img)

    try:
        mask_img = refine_mask_reduce_windows_wheels(pil, mask_img)
    except Exception:
        pass

    # IMPORTANT: Stable Diffusion inpainting uses WHITE = inpaint, BLACK = keep.
    # Ensure mask is binary-ish
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    mask_img = mask_img.point(lambda px: 255 if px > 120 else 0)

    # 2) Prepare fixed-size image + mask to satisfy SD inpaint 422 width/height constraints
    pil_sq = letterbox_to_square(pil, INPAINT_SIZE, fill=(0, 0, 0))
    mask_sq = letterbox_mask_to_square(mask_img, INPAINT_SIZE)

    inpaint_image_data_url = img_to_data_url(pil_sq, fmt="PNG")
    inpaint_mask_data_url = img_to_data_url(mask_sq, fmt="PNG")

    # Wait a bit to avoid Replicate burst throttling (segmentation -> inpaint)
    time.sleep(REPLICATE_BURST_SLEEP)

    prompt = build_prompt(COLOR_MAP[color], finish, angle)

    # Conservative settings to reduce “car changes”
    # (Models differ, but these are usually safer.)
    try:
        out = replicate.run(
            img_version,
            input={
                "image": inpaint_image_data_url,
                "mask": inpaint_mask_data_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt(),
                # Try to keep original structure
                "strength": float(strength),  # you can tune from Webflow
                "guidance_scale": 5.5,
                "num_inference_steps": 28,
                "num_outputs": 1,
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (inpaint): {str(e)}")

    after_url = normalize_replicate_output(out)
    if not after_url:
        raise HTTPException(status_code=500, detail="Render failed (no after image returned).")

    # Accept either http(s) or data URLs
    if not (after_url.startswith("http://") or after_url.startswith("https://") or after_url.startswith("data:image/")):
        raise HTTPException(status_code=500, detail="Render failed (invalid after image).")

    return {
        "before": original_data_url,  # original input (data url)
        "after": after_url,           # output image url/data
        "mask": mask_url,             # segmentation mask url
    }
