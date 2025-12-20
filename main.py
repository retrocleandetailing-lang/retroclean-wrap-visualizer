import os
import io
import base64
import time
import urllib.request
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

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

    return (
        f"Photorealistic wrap preview of the SAME car photo. "
        f"Keep the same car make/model, same exact camera perspective ({angle_hint}), "
        f"same background and lighting. "
        f"ONLY change the painted body panels (doors, fenders, hood, quarter panels, bumpers) "
        f"to {color_name} {finish_text}. "
        f"{flake}"
        f"Preserve all panel gaps and seam lines (door lines, hood lines, trunk lines) clearly. "
        f"Do NOT change wheels, tires, brakes, headlights, taillights, windows, windshield, trim, badges, grille, or interior. "
        f"Do NOT change the environment. "
        f"Maintain original reflections shape and realism, just recolor the painted panels."
    )


def negative_prompt() -> str:
    return (
        "different car, changed body kit, changed wheels, changed background, "
        "cartoon, CGI, illustration, anime, warped panels, melted shapes, "
        "extra parts, text, watermark, logo, blurry, low quality"
    )


def refine_mask_reduce_windows_wheels(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Heuristic cleanup:
    - Windows + wheels are often very dark and low-saturation.
    - We remove very-dark & low-sat pixels from the mask area using the original image.
    This is not perfect, but usually improves the “blue wheels / tinted glass” problem a lot.
    """
    # Ensure same size
    mask = mask_l.resize(original_rgb.size).convert("L")
    mask = ImageOps.autocontrast(mask)

    # Convert to HSV for saturation/value
    hsv = original_rgb.convert("HSV")
    h, s, v = hsv.split()

    # Create "keep" map: keep pixels that are NOT both (very dark) and (low sat)
    # Tuned for car photos; adjust if needed.
    # v: 0-255 (brightness), s: 0-255 (saturation)
    v_ok = v.point(lambda px: 255 if px > 55 else 0)
    s_ok = s.point(lambda px: 255 if px > 35 else 0)

    # keep_if = v_ok OR s_ok
    keep_if = ImageChops.lighter(v_ok, s_ok)

    # Only apply inside the original mask region:
    # cleaned_mask = mask AND keep_if
    cleaned = ImageChops.multiply(mask, keep_if)

    # Slight blur/threshold to smooth edges (no hard jaggies)
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
        raise HTTPException(status_code=429, detail=f"Too many requests. Please wait ~{COOLDOWN_SECONDS}s and try again.")
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
    if not seg_version or not img_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION or REPLICATE_IMG_VERSION on server.")

    os.environ["REPLICATE_API_TOKEN"] = token

    # Convert original image to data URL (PNG is safest for Replicate image inputs)
    original_data_url = img_to_data_url(pil, fmt="PNG")

    # 1) Segmentation (try to focus on body panels)
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

    # --- Optional: heuristic cleanup to reduce windows/wheels bleed ---
    # These imports are here to avoid unused warnings if you remove the cleanup.
    from PIL import ImageChops, ImageFilter
    cleaned_mask = mask_img.resize(pil.size).convert("L")
    cleaned_mask = ImageOps.autocontrast(cleaned_mask)

    # Try cleanup (safe fallback if anything goes wrong)
    try:
        cleaned_mask = refine_mask_reduce_windows_wheels(pil, cleaned_mask)
    except Exception:
        pass

    # Replicate inpainting: WHITE = inpaint, BLACK = preserve.
    # We want to inpaint ONLY the body panels.
    mask_data_url = img_to_data_url(cleaned_mask, fmt="PNG")

    # 2) Inpainting model (realistic wrap)
    prompt = build_prompt(COLOR_MAP[color], finish, angle)

    # Slightly different defaults by finish
    guidance = 7.5
    steps = 30
    if finish == "matte":
        guidance = 7.0
        steps = 28

    try:
        out = replicate.run(
            img_version,
            input={
                "prompt": prompt,
                "image": original_data_url,
                "mask": mask_data_url,
                "negative_prompt": negative_prompt(),
                "num_outputs": 1,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                # Many SD inpaint models use prompt_strength; your provided strength can map to it.
                # If your chosen version doesn't support it, remove this line.
                "prompt_strength": float(strength),
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
