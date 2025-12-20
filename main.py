
import os
import io
import base64
import time
import urllib.request
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageChops, ImageFilter

import replicate
from replicate.exceptions import ReplicateError

# Replicate rate limits can get tighter on low spend.
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
            "Add subtle fine metallic flake micro-sparkle in highlights only "
            "(do NOT look glittery; keep it premium and realistic). "
        )

    # Keep prompt strict and non-creative
    return (
        f"Repaint ONLY the visible car body panels (doors, fenders, hood, quarter panels, bumpers) "
        f"with a {color_name} {finish_text}. "
        f"{flake}"
        f"Preserve original panel gaps and seam lines (door lines, hood lines, trunk lines) clearly. "
        f"Do not change car make/model/year, body shape, camera perspective ({angle_hint}), background, lighting, or reflections shape. "
        f"Do not change wheels, tires, brakes, headlights, taillights, glass/windows, trim, badges, grille, mirrors, or interior. "
        f"This is a repaint of the same photo, not a new car."
        f"STRICT image edit. Do NOT generate a new car. "
        f"Do NOT change vehicle identity. "
        f"This is a recolor of the exact same photograph. "

    )


def negative_prompt() -> str:
    return (
        "new car, different car, different model, different brand, "
        "redesign, restyle, concept car, vintage car, classic car, "
        "changed grille, changed headlights, changed body shape, "
        "changed badge, changed emblem, changed proportions, "
        "cartoon, CGI, illustration, anime, low quality, blurry"
    )


def refine_mask_reduce_windows_wheels(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Heuristic cleanup:
    - Windows + wheels are often very dark and/or low-saturation.
    - We remove very-dark & low-sat pixels from the inpaint mask using the original image.
    WHITE = inpaint, BLACK = preserve.
    """
    mask = mask_l.resize(original_rgb.size).convert("L")
    mask = ImageOps.autocontrast(mask)

    hsv = original_rgb.convert("HSV")
    _, s, v = hsv.split()

    # Thresholds tuned for typical car photos
    v_ok = v.point(lambda px: 255 if px > 55 else 0)
    s_ok = s.point(lambda px: 255 if px > 35 else 0)

    # keep_if = v_ok OR s_ok
    keep_if = ImageChops.lighter(v_ok, s_ok)

    # Apply only inside current mask: cleaned = mask AND keep_if
    cleaned = ImageChops.multiply(mask, keep_if)

    # Smooth edges + threshold
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
    strength: float = Form(0.25),  # backend default; Webflow can override if needed
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
        # Keep size reasonable (faster + more consistent)
        if w > 2200 or h > 2200:
            pil.thumbnail((2200, 2200))
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
    if not seg_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION on server.")
    if not img_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_IMG_VERSION on server.")

    os.environ["REPLICATE_API_TOKEN"] = token

    # Convert original image to data URL (PNG is safest)
    original_data_url = img_to_data_url(pil, fmt="PNG")

    # 1) Segmentation
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

    # Extract mask URL
    mask_url = None
    if isinstance(seg_out, dict):
        mask_url = seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    elif isinstance(seg_out, list) and seg_out:
        mask_url = seg_out[0]
    elif isinstance(seg_out, str):
        mask_url = seg_out

    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")

    # Download and refine mask
    try:
        mask_img = download_image(mask_url).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    cleaned_mask = mask_img.resize(pil.size).convert("L")
    cleaned_mask = ImageOps.autocontrast(cleaned_mask)

    # Optional: heuristic cleanup to reduce windows/wheels bleed
    try:
        cleaned_mask = refine_mask_reduce_windows_wheels(pil, cleaned_mask)
    except Exception:
        # If heuristic fails, fall back to raw mask
        cleaned_mask = mask_img.resize(pil.size).convert("L")
        cleaned_mask = ImageOps.autocontrast(cleaned_mask)

    # Replicate inpainting: WHITE = inpaint, BLACK = preserve.
    mask_data_url = img_to_data_url(cleaned_mask, fmt="PNG")

    # Small delay to satisfy Replicate burst behavior
    time.sleep(12)

    # 2) Inpainting model
    prompt = build_prompt(COLOR_MAP[color], finish, angle)

    # Clamp strength to safe range (prevents hallucinated “new cars”)
    try:
        strength_val = float(strength)
    except Exception:
        strength_val = 0.12
    strength_val = max(0.15, min(0.35, strength_val))

    # Use conservative settings to keep the SAME car
    guidance_scale = 3.8
    steps = 22

    try:
        out = replicate.run(
        img_version,
        input={
            "image": original_data_url,
            "mask": mask_data_url,

            "prompt": prompt,
            "negative_prompt": negative_prompt(),

            # 🔒 VERY IMPORTANT — RECOLOR ONLY
            "strength": 0.12,
            "guidance_scale": 3.8,
            "num_inference_steps": 22,

            "num_outputs": 1,
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
        "before": original_data_url,
        "after": result_url,
        "mask": mask_url,
    }
