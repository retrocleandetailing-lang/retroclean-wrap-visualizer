import os
import io
import base64
import time
import urllib.request
from typing import Literal, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageChops, ImageFilter

import replicate
from replicate.exceptions import ReplicateError


# ----------------------------
# CONFIG
# ----------------------------

# One click = 2 predictions (segment + inpaint). Keep conservative.
COOLDOWN_SECONDS = 45

# Your exact model versions
SEG_VERSION = "tmappdev/lang-segment-anything:891411c38a6ed2d44c004b7b9e44217df7a5b07848f29ddefd2e28bc7cbf93bc"
INPAINT_VERSION = "stability-ai/stable-diffusion-inpainting:c2172c447eb69551b59f62fd2d61dd84054e9fb7bc8a42fbe398c2a7a072ed68"

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


def round_to_64(n: int) -> int:
    return max(64, (n // 64) * 64)


def fit_max(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


def build_prompt(color_name: str, finish: Finish, angle: Angle) -> str:
    angle_hint = {
        "front_3q": "front three-quarter view",
        "side": "side profile view",
        "rear_3q": "rear three-quarter view",
    }[angle]

    finish_text = {
        "gloss": "high-gloss vinyl wrap with clearcoat-like highlights, subtle premium metallic micro-flake in highlights only",
        "satin": "satin vinyl wrap with soft reflections",
        "matte": "matte vinyl wrap with diffused reflections",
    }[finish]

    # Keep this SHORT + STRICT to prevent hallucinating a different car.
    return (
        f"Same exact car photo ({angle_hint}). "
        f"Do NOT change the car make/model/body kit/wheels/windows/lights/grille/background/camera. "
        f"Only recolor the painted body panels to {color_name} {finish_text}. "
        f"Preserve panel gaps and seam lines (doors/hood/trunk). "
        f"Keep original reflections and lighting structure; only change paint color."
    )


def negative_prompt() -> str:
    # Add “old car / classic car” to reduce random vintage swaps.
    return (
        "different car, different make, different model, classic car, vintage car, old car, "
        "changed wheels, changed windows, changed background, changed camera angle, "
        "cartoon, CGI, illustration, anime, warped panels, melted shapes, extra parts, "
        "text, watermark, logo, blurry, low quality"
    )


def refine_mask_reduce_windows_wheels(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Heuristic cleanup:
    Remove very-dark AND low-saturation areas from the mask (often windows/wheels/tires).
    This won't be perfect, but it prevents the worst “blue wheels / tinted glass”.
    """
    mask = mask_l.resize(original_rgb.size).convert("L")
    mask = ImageOps.autocontrast(mask)

    hsv = original_rgb.convert("HSV")
    _, s, v = hsv.split()

    # Thresholds (tunable)
    v_ok = v.point(lambda px: 255 if px > 60 else 0)     # not too dark
    s_ok = s.point(lambda px: 255 if px > 35 else 0)     # has some color

    keep_if = ImageChops.lighter(v_ok, s_ok)             # v_ok OR s_ok
    cleaned = ImageChops.multiply(mask, keep_if)         # apply only inside mask

    cleaned = cleaned.filter(ImageFilter.GaussianBlur(radius=1.2))
    cleaned = cleaned.point(lambda px: 255 if px > 120 else 0)

    return cleaned


def extract_mask_url(seg_out) -> str | None:
    if isinstance(seg_out, dict):
        return seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    if isinstance(seg_out, list) and seg_out:
        return seg_out[0]
    if isinstance(seg_out, str):
        return seg_out
    return None


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
    strength: float = Form(0.45),  # kept for compatibility w/ your Webflow, not used by SD inpaint
):
    # Cooldown (server-side)
    now = time.time()
    last = getattr(app.state, "last_call_ts", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail=f"Too many requests. Please wait ~{COOLDOWN_SECONDS}s and try again.")
    app.state.last_call_ts = now

    if color not in COLOR_MAP:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    # Replicate token (must be set in Render env vars)
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    os.environ["REPLICATE_API_TOKEN"] = token

    # Load image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = pil.size
        if w < 900 or h < 900:
            raise HTTPException(status_code=400, detail="Photo too small. Use at least ~900px on each side.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Resize to SD-inpaint supported sizes:
    # width/height must be one of: 64..1024 in steps of 64
    W, H = pil.size

    max_dim = 1024
    scale = min(max_dim / W, max_dim / H, 1.0)  # never upscale
    new_w = int(W * scale)
    new_h = int(H * scale)

    # round DOWN to nearest multiple of 64 (must be >= 64)
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)

    # IMPORTANT: resize the actual image to match the exact dims we send to Replicate
    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    W64, H64 = new_w, new_h

    original_data_url = img_to_data_url(pil, fmt="PNG")

    # 1) Segmentation
    try:
        seg_out = replicate.run(
            SEG_VERSION,
            input={
                "image": original_data_url,
                "text_prompt": "car body panels",
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (segmentation): {str(e)}")

    mask_url = extract_mask_url(seg_out)
    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")

    # Download mask
    try:
        mask_img = download_image(mask_url).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    # Prepare mask: white=inpaint, black=preserve :contentReference[oaicite:2]{index=2}
    mask_img = mask_img.resize(pil.size)
    mask_img = ImageOps.autocontrast(mask_img)

    # Heuristic cleanup (helps reduce wheels/windows recolor)
    try:
        mask_img = refine_mask_reduce_windows_wheels(pil, mask_img)
    except Exception:
        pass

    # OPTIONAL: tighten edges a bit
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    mask_img = mask_img.point(lambda px: 255 if px > 140 else 0)

    mask_data_url = img_to_data_url(mask_img, fmt="PNG")

    # Replicate “burst of 1” is common; spacing helps if they throttle
    time.sleep(12)

    # 2) Inpainting (Stable Diffusion Inpainting)
    prompt = build_prompt(COLOR_MAP[color], finish, angle)

    # Settings tuned to reduce hallucination:
    # - guidance lower
    # - steps moderate
    guidance = 5.0
    steps = 35
    if finish == "matte":
        guidance = 4.7
        steps = 32

    try:
        out = replicate.run(
            INPAINT_VERSION,
            input={
                "prompt": prompt,
                "negative_prompt": negative_prompt(),
                "image": original_data_url,
                "mask": mask_data_url,
                "width": W64,
                "height": H64,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "num_outputs": 1,
                # leave seed blank/random; you can add a fixed int if you want repeatability
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (inpaint): {str(e)}")

    result_url = None
    if isinstance(out, list) and out:
        result_url = out[0]
    elif isinstance(out, str):
        result_url = out

    if not result_url:
        raise HTTPException(status_code=500, detail="Render failed (no image returned).")

    return {
        "before": original_data_url,  # always the real original
        "after": result_url,          # SD output URL
        "mask": mask_url,             # raw mask URL (debug)
    }
