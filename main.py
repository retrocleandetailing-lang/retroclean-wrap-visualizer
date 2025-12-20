import os
import io
import base64
import time
from typing import Literal

import requests
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import replicate
from replicate.exceptions import ReplicateError


# -----------------------------
# SETTINGS
# -----------------------------
COOLDOWN_SECONDS = 20  # 1 replicate call per request; 20s is safe

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

COLOR_HEX = {
    "nardo_grey": "#9FA4A9",
    "satin_black": "#1A1A1A",
    "gloss_black": "#0B0B0B",
    "miami_blue": "#00B7D6",
    "british_racing_green": "#0B3D2E",
    "ruby_red": "#7A0F1C",
    "pearl_white": "#F2F2F0",
}


# -----------------------------
# HELPERS
# -----------------------------
def img_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def hex_to_rgb(hexstr: str):
    hexstr = hexstr.strip().lstrip("#")
    return tuple(int(hexstr[i:i + 2], 16) for i in (0, 2, 4))


def screen_blend(a: Image.Image, b: Image.Image) -> Image.Image:
    """Screen blend for RGB images (photographic highlight behavior)."""
    inv = ImageChops.invert(ImageChops.multiply(ImageChops.invert(a), ImageChops.invert(b)))
    return inv


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# -----------------------------
# APP
# -----------------------------
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


@app.get("/")
def root():
    return {"message": "RetroClean Wrap Visualizer API. Use /health or POST /render."}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug-token")
def debug_token():
    tok = os.getenv("REPLICATE_API_TOKEN", "")
    if not tok:
        return {"token_present": False}
    return {"token_present": True, "token_prefix": tok[:6], "token_suffix": tok[-4:]}


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),      # kept for UI; not needed for overlay method
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.45),  # used as intensity control
):
    # -----------------------------
    # Cooldown
    # -----------------------------
    now = time.time()
    if not hasattr(app.state, "last_call_ts"):
        app.state.last_call_ts = 0.0

    if now - app.state.last_call_ts < COOLDOWN_SECONDS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait about {COOLDOWN_SECONDS} seconds and try again."
        )
    app.state.last_call_ts = now

    # -----------------------------
    # Validate inputs
    # -----------------------------
    if color not in COLOR_MAP or color not in COLOR_HEX:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    try:
        strength = float(strength)
    except Exception:
        strength = 0.45
    strength = max(0.05, min(0.90, strength))

    # -----------------------------
    # Load image
    # -----------------------------
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = pil.size
        if w < 900 or h < 900:
            raise HTTPException(status_code=400, detail="Photo too small. Use at least ~900px on each side.")
        if w > 5000 or h > 5000:
            pil.thumbnail((5000, 5000))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # -----------------------------
    # Replicate env
    # -----------------------------
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    os.environ["REPLICATE_API_TOKEN"] = token

    seg_version = os.getenv("REPLICATE_SEG_VERSION")
    if not seg_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION on server.")

    original_data_url = img_to_data_url(pil, fmt="JPEG")

    # -----------------------------
    # 1) Segmentation
    # -----------------------------
    try:
        seg_out = replicate.run(
            seg_version,
            input={"image": original_data_url, "text_prompt": "car body"},
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate segmentation error: {e}")

    mask_url = None
    if isinstance(seg_out, dict):
        mask_url = seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    elif isinstance(seg_out, list) and seg_out:
        mask_url = seg_out[0]
    elif isinstance(seg_out, str):
        mask_url = seg_out

    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")

    # -----------------------------
    # 2) Download & refine mask (remove windows/wheels)
    # -----------------------------
    try:
        mask_bytes = requests.get(str(mask_url), timeout=30).content
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    mask_img = mask_img.resize(pil.size)
    mask = ImageOps.autocontrast(mask_img)

    # Heuristic removal: windows/tires are dark → subtract them from mask
    gray_orig = ImageOps.grayscale(pil)
    dark_regions = gray_orig.point(lambda p: 255 if p < 65 else 0)  # threshold tuned for car shots
    dark_regions = dark_regions.filter(ImageFilter.GaussianBlur(radius=3))
    mask = ImageChops.subtract(mask, dark_regions)

    # Clean edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    mask = ImageOps.autocontrast(mask)

    # Optional: keep only strong mask areas (reduces spill)
    mask = mask.point(lambda p: 255 if p > 70 else 0).filter(ImageFilter.GaussianBlur(radius=1.5))

    # -----------------------------
    # 3) Base wrap overlay (preserve lighting)
    # -----------------------------
    r, g, b = hex_to_rgb(COLOR_HEX[color])

    base_gray = ImageOps.grayscale(pil)
    colored = ImageOps.colorize(base_gray, black=(0, 0, 0), white=(r, g, b)).convert("RGB")

    # Intensity mapping
    overlay_amt = 0.55 + (0.30 * strength)  # 0.55 → 0.82
    overlay_amt = clamp01(overlay_amt)

    blended = Image.blend(pil, colored, overlay_amt)
    wrapped = Image.composite(blended, pil, mask)

    # -----------------------------
    # 4) Panel separation (door/hood lines)
    # -----------------------------
    # Edge detect on grayscale, then apply only on wrapped areas
    edges = gray_orig.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    # Keep only stronger edges (panel gaps)
    edges = edges.point(lambda p: 255 if p > 140 else 0)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=1))

    # Convert edges to a dark line layer
    panel_lines = ImageOps.colorize(edges, black=(0, 0, 0), white=(0, 0, 0)).convert("RGB")

    # Apply panel lines only where mask applies
    panel_lines_only = Image.composite(panel_lines, Image.new("RGB", pil.size, (0, 0, 0)), mask)

    # Multiply dark lines into wrapped (subtle)
    wrapped = ImageChops.multiply(wrapped, ImageChops.invert(panel_lines_only).point(lambda p: max(0, p - 35)))

    # -----------------------------
    # 5) Metallic flake (gloss only)
    # -----------------------------
    if finish == "gloss":
        # Noise texture
        noise = Image.effect_noise(pil.size, 22).convert("L")  # grain size
        noise = ImageOps.autocontrast(noise).point(lambda p: 255 if p > 210 else 0)  # keep only bright specks
        noise = noise.filter(ImageFilter.GaussianBlur(radius=0.6))

        # Turn specks into near-white sparkle
        sparkle = ImageOps.colorize(noise, black=(0, 0, 0), white=(240, 240, 240)).convert("RGB")
        sparkle = Image.composite(sparkle, Image.new("RGB", pil.size, (0, 0, 0)), mask)

        # Screen blend sparkle very lightly
        wrapped = Image.blend(wrapped, screen_blend(wrapped, sparkle), 0.12)

    # -----------------------------
    # 6) Clearcoat highlights (gloss + satin)
    # -----------------------------
    if finish in ("gloss", "satin"):
        # Extract highlights from original luminance
        hi = gray_orig.point(lambda p: 255 if p > 200 else 0)
        hi = hi.filter(ImageFilter.GaussianBlur(radius=8))
        hi = ImageOps.autocontrast(hi)

        highlight_layer = ImageOps.colorize(hi, black=(0, 0, 0), white=(255, 255, 255)).convert("RGB")
        highlight_layer = Image.composite(highlight_layer, Image.new("RGB", pil.size, (0, 0, 0)), mask)

        strength_hi = 0.18 if finish == "gloss" else 0.10
        wrapped = Image.blend(wrapped, screen_blend(wrapped, highlight_layer), strength_hi)

    # -----------------------------
    # 7) Finish tuning
    # -----------------------------
    if finish == "gloss":
        wrapped = ImageEnhance.Contrast(wrapped).enhance(1.07)
        wrapped = ImageEnhance.Color(wrapped).enhance(1.12)
    elif finish == "satin":
        wrapped = ImageEnhance.Contrast(wrapped).enhance(1.03)
        wrapped = ImageEnhance.Color(wrapped).enhance(1.06)
    elif finish == "matte":
        wrapped = ImageEnhance.Contrast(wrapped).enhance(0.97)
        wrapped = ImageEnhance.Color(wrapped).enhance(0.92)

    after_data_url = img_to_data_url(wrapped, fmt="JPEG")

    return {"before": original_data_url, "after": after_data_url, "mask": str(mask_url)}
