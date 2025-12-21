import os
import io
import base64
import time
import urllib.request
from typing import Literal, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter, ImageChops

import replicate
from replicate.exceptions import ReplicateError


# ----------------------------
# CONFIG
# ----------------------------
COOLDOWN_SECONDS = 45  # one click = 2 predictions (seg + inpaint)

SEG_MODEL = "tmappdev/lang-segment-anything:891411c38a6ed2d44c004b7b9e44217df7a5b07848f29ddefd2e28bc7cbf93bc"
INPAINT_MODEL = "stability-ai/stable-diffusion-inpainting:c2172c447eb69551b59f62fd2d61dd84054e9fb7bc8a42fbe398c2a7a072ed68"

# SD inpainting commonly enforces these widths/heights (you saw this exact list in the 422)
ALLOWED_DIMS = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]

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


# ----------------------------
# APP
# ----------------------------
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


# ----------------------------
# HELPERS
# ----------------------------
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


def nearest_allowed(n: int) -> int:
    # choose the closest allowed dimension, but clamp to [64..1024]
    n = max(64, min(1024, n))
    return min(ALLOWED_DIMS, key=lambda x: abs(x - n))


def fit_to_allowed_dims(img: Image.Image, target_max: int = 1024) -> Tuple[Image.Image, int, int]:
    """
    Resize (preserve aspect) then pad to the nearest allowed dims (fixes 422 width/height validation).
    Returns (new_img, width, height).
    """
    w, h = img.size

    # scale down if needed so the larger side <= target_max
    scale = min(1.0, float(target_max) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    w, h = img.size
    new_w = nearest_allowed(w)
    new_h = nearest_allowed(h)

    # pad to exact allowed dims (do NOT stretch the image)
    pad_w = max(0, new_w - w)
    pad_h = max(0, new_h - h)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(0, 0, 0))
    return img, new_w, new_h


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

    metallic = ""
    if finish == "gloss":
        metallic = (
            "Add subtle fine metallic micro-flake ONLY inside specular highlights; "
            "premium OEM-like finish, NOT glittery. "
        )

    return (
        "Photorealistic edit of the SAME EXACT input photo. "
        f"Keep the same car and same exact camera perspective ({angle_hint}), same lens, same framing, "
        "same background, same lighting direction, same reflections SHAPE. "
        "ONLY recolor the painted body panels (doors, fenders, hood, quarter panels, bumpers) "
        f"to {color_name} {finish_text}. "
        f"{metallic}"
        "Preserve crisp panel gaps and seam lines (door lines, hood lines, trunk lines). "
        "Do NOT modify wheels, tires, brakes, windows, windshield, trim, badges, grille, headlights, taillights, "
        "interior, stance, body kit, or environment. "
        "No car swapping. No changing make/model. No changing background."
    )


def negative_prompt() -> str:
    return (
        "different car, different make, different model, car swap, changed background, "
        "changed wheels, changed windows, changed headlights, changed grille, changed body kit, "
        "cartoon, anime, illustration, CGI, warped, melted, extra parts, text, watermark, logo, blurry, low quality"
    )


def clean_mask(mask_l: Image.Image) -> Image.Image:
    """
    Basic cleanup: autocontrast + blur + threshold + slight dilate/erode to stabilize edges.
    White = edit, Black = keep.
    """
    m = mask_l.convert("L")
    m = ImageOps.autocontrast(m)

    # smooth noise
    m = m.filter(ImageFilter.GaussianBlur(radius=1.2))

    # threshold to solid mask
    m = m.point(lambda px: 255 if px > 140 else 0)

    # morphological-ish cleanup using min/max filters
    m = m.filter(ImageFilter.MaxFilter(5))  # dilate
    m = m.filter(ImageFilter.MinFilter(3))  # erode (slightly)
    return m


# ----------------------------
# ROUTES
# ----------------------------
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
    strength: float = Form(0.28),  # default low to prevent re-imagining the entire car
):
    # cooldown (server-side)
    now = time.time()
    last = getattr(app.state, "last_call_ts", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail=f"Too many requests. Please wait ~{COOLDOWN_SECONDS}s and try again.")
    app.state.last_call_ts = now

    if color not in COLOR_MAP:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    # token required
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    os.environ["REPLICATE_API_TOKEN"] = token

    # read & validate image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        if pil.size[0] < 700 or pil.size[1] < 700:
            raise HTTPException(status_code=400, detail="Photo too small. Please use a higher-resolution photo.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # fit to allowed dims for SD-inpaint (fixes 422 width/height)
    pil, out_w, out_h = fit_to_allowed_dims(pil, target_max=1024)

    # build urls
    original_data_url = img_to_data_url(pil, fmt="PNG")

    # 1) segmentation
    try:
        seg_out = replicate.run(
            SEG_MODEL,
            input={
                "image": original_data_url,
                "text_prompt": "car body panels",  # tighter than "car"
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (segmentation): {str(e)}")

    mask_url = None
    if isinstance(seg_out, dict):
        mask_url = seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    elif isinstance(seg_out, list) and seg_out:
        mask_url = seg_out[0]
    elif isinstance(seg_out, str):
        mask_url = seg_out

    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")

    # download + clean mask, then force size match
    try:
        mask_img = download_image(mask_url).convert("L")
        mask_img = mask_img.resize((out_w, out_h), Image.NEAREST)
        mask_img = clean_mask(mask_img)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download/prepare segmentation mask.")

    # IMPORTANT: inpainting expects white=edit, black=keep
    # If your mask comes inverted, toggle this line:
    # mask_img = ImageOps.invert(mask_img)

    mask_data_url = img_to_data_url(mask_img, fmt="PNG")

    # Replicate “burst=1” behavior can still 429 if you chain calls too fast
    time.sleep(10)

    # 2) inpaint (heavy constraints)
    prompt = build_prompt(COLOR_MAP[color], finish, angle)

    # clamp strength to a safe band
    strength = float(strength)
    strength = max(0.18, min(0.40, strength))

    # NOTE: the stability-ai/stable-diffusion-inpainting model supports width/height.
    # If your run still complains, we’ll adjust the parameter names to match the model schema.
    try:
        out = replicate.run(
            INPAINT_MODEL,
            input={
                "image": original_data_url,
                "mask": mask_data_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt(),
                "strength": strength,

                # critical: lock image structure
                "guidance_scale": 5.5,          # higher = more “creative” (bad for car identity)
                "num_inference_steps": 28,

                # fix 422 dimension validation
                "width": out_w,
                "height": out_h,

                "num_outputs": 1,
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=422 if "Input validation failed" in str(e) else 429,
                            detail=f"Replicate error (inpaint): {str(e)}")

    result_url = None
    if isinstance(out, list) and out:
        result_url = out[0]
    elif isinstance(out, str):
        result_url = out

    if not result_url:
        raise HTTPException(status_code=500, detail="Render failed (no image returned).")

    return {"before": original_data_url, "after": result_url, "mask": mask_url}
