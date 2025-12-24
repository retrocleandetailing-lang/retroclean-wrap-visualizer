import os
import io
import base64
import time
import urllib.request
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageEnhance

import replicate
from replicate.exceptions import ReplicateError


# ====== CONFIG ======
COOLDOWN_SECONDS = 20  # segmentation is 1 call; keep UI snappy

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

# More accurate base tones (tweak any time)
COLOR_HEX = {
    "nardo_grey": "#9FA4A9",
    "satin_black": "#1A1A1A",
    "gloss_black": "#0B0B0B",
    "miami_blue": "#00B7D6",
    "british_racing_green": "#0B3D2E",
    "ruby_red": "#7A0F1C",
    "pearl_white": "#F2F2F0",
}


# ====== APP ======
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


# ====== HELPERS ======
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


def hex_to_rgb(hexstr: str):
    h = hexstr.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def cleanup_mask(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Improve the segmentation mask:
    - autocontrast + blur + threshold
    - remove very dark areas (wheels/windows tend to be very dark)
    - shrink slightly to avoid bleeding onto trim/glass
    """
    mask = mask_l.resize(original_rgb.size).convert("L")
    mask = ImageOps.autocontrast(mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.3))
    mask = mask.point(lambda p: 255 if p > 120 else 0)

    # Remove very dark pixels (windows/wheels) using original luminance
    gray = ImageOps.grayscale(original_rgb)
    not_dark = gray.point(lambda p: 255 if p > 45 else 0)  # keep brighter zones
    mask = ImageChops.multiply(mask, not_dark)

    # Shrink mask slightly (erode) to reduce edge bleed
    # PIL doesn't have morphological erode built-in; approximate by MinFilter.
    mask = mask.filter(ImageFilter.MinFilter(size=3))

    # Smooth edges again
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.9))
    mask = mask.point(lambda p: 255 if p > 110 else 0)

    return mask


def make_panel_lines(original_rgb: Image.Image) -> Image.Image:
    """
    Extract panel seams/edges as a dark line layer.
    """
    gray = ImageOps.grayscale(original_rgb)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    # Keep only stronger edges
    edges = edges.point(lambda p: 255 if p > 110 else 0)
    # Slight blur to feel natural
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.6))
    return edges.convert("L")


def make_clearcoat_highlights(original_rgb: Image.Image, intensity: float) -> Image.Image:
    """
    Create a soft highlight layer from luminance (specular-ish).
    """
    gray = ImageOps.grayscale(original_rgb)
    # emphasize highlights
    hi = gray.point(lambda p: max(0, p - 160) * 3)  # only bright zones
    hi = hi.filter(ImageFilter.GaussianBlur(radius=6))
    hi = ImageOps.autocontrast(hi)
    # scale intensity
    hi = hi.point(lambda p: int(p * intensity))
    return hi.convert("L")


def make_metallic_flake(original_rgb: Image.Image, amount: float) -> Image.Image:
    """
    Metallic micro-sparkle: subtle noise, only on highlight areas.
    """
    gray = ImageOps.grayscale(original_rgb)
    hi = gray.point(lambda p: 255 if p > 175 else 0).filter(ImageFilter.GaussianBlur(radius=2))

    # pseudo-noise using built-in effect noise
    noise = Image.effect_noise(original_rgb.size, 35).convert("L")
    noise = ImageOps.autocontrast(noise)
    noise = noise.point(lambda p: 255 if p > 210 else 0)  # sparse sparkles

    flake = ImageChops.multiply(hi, noise)
    flake = flake.filter(ImageFilter.GaussianBlur(radius=0.7))
    flake = flake.point(lambda p: int(p * amount))
    return flake.convert("L")


def apply_wrap(
    original_rgb: Image.Image,
    mask: Image.Image,
    color_key: str,
    finish: Finish,
    strength: float,
) -> Image.Image:
    """
    Deterministic wrap render:
    - preserve shading using grayscale
    - recolor using target color
    - add panel lines
    - add clearcoat highlights + metallic flake (gloss)
    """
    strength = clamp01(strength)

    target_hex = COLOR_HEX.get(color_key)
    if not target_hex:
        raise HTTPException(status_code=400, detail="Unsupported color.")
    r, g, b = hex_to_rgb(target_hex)

    # Base shading
    base_gray = ImageOps.grayscale(original_rgb)

    # Colorize shading into target hue (keeps reflections/shadows)
    colored = ImageOps.colorize(base_gray, black=(0, 0, 0), white=(r, g, b)).convert("RGB")

    # Blend original + colored for realism
    # Higher strength = stronger color change
    blended = Image.blend(original_rgb, colored, strength)

    # Finish tuning
    if finish == "gloss":
        blended = ImageEnhance.Contrast(blended).enhance(1.10)
        blended = ImageEnhance.Color(blended).enhance(1.15)
    elif finish == "satin":
        blended = ImageEnhance.Contrast(blended).enhance(1.05)
        blended = ImageEnhance.Color(blended).enhance(1.08)
    else:  # matte
        blended = ImageEnhance.Contrast(blended).enhance(0.95)
        blended = ImageEnhance.Color(blended).enhance(0.92)

    # Composite ONLY on mask area
    wrapped = Image.composite(blended, original_rgb, mask)

    # Panel separation lines (darken seams a touch only on car)
    panel = make_panel_lines(original_rgb)
    panel_on_car = ImageChops.multiply(panel, mask)
    panel_on_car = panel_on_car.point(lambda p: int(p * 0.45))  # subtle
    # Darken wrapped using panel lines
    wrapped = ImageChops.subtract(wrapped, Image.merge("RGB", (panel_on_car, panel_on_car, panel_on_car)))

    # Clearcoat highlights (gloss/satin)
    if finish in ("gloss", "satin"):
        hi_intensity = 0.65 if finish == "gloss" else 0.35
        clear = make_clearcoat_highlights(original_rgb, hi_intensity)
        clear_on_car = ImageChops.multiply(clear, mask)
        # Add highlights back
        wrapped = ImageChops.add(wrapped, Image.merge("RGB", (clear_on_car, clear_on_car, clear_on_car)))

    # Metallic flake (gloss only)
    if finish == "gloss":
        flake = make_metallic_flake(original_rgb, amount=0.45)
        flake_on_car = ImageChops.multiply(flake, mask)
        wrapped = ImageChops.add(wrapped, Image.merge("RGB", (flake_on_car, flake_on_car, flake_on_car)))

    return wrapped


# ====== ROUTES ======
@app.get("/")
def root():
    return {"message": "RetroClean Wrap Visualizer API. Use /health or POST /render."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),   # kept for UI consistency (not used in deterministic render)
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.65),
):
    # cooldown (server side)
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
        # Keep decent size for good masking but not huge
        if max(w, h) > 2200:
            pil.thumbnail((2200, 2200))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Replicate env vars (segmentation only)
    token = os.getenv("REPLICATE_API_TOKEN")
    seg_version = os.getenv("REPLICATE_SEG_VERSION")

    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    if not seg_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION on server.")

    os.environ["REPLICATE_API_TOKEN"] = token

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

    # Download + cleanup mask
    try:
        mask_img = download_image(mask_url).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    mask_clean = cleanup_mask(pil, mask_img)

    # 2) Deterministic wrap render (no random car changes)
    wrapped = apply_wrap(
        original_rgb=pil,
        mask=mask_clean,
        color_key=color,
        finish=finish,
        strength=float(strength),
    )

    return {
        "before": img_to_data_url(pil, fmt="PNG"),
        "after": img_to_data_url(wrapped, fmt="PNG"),
        "mask": mask_url,
    }
