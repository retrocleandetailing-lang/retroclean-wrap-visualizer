import os
import io
import base64
import time
import urllib.request
import hashlib
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops

import replicate
from replicate.exceptions import ReplicateError

# One click = 1 prediction (segmentation). Keep cooldown conservative.
COOLDOWN_SECONDS = 25

Angle = Literal["front_3q", "side", "rear_3q"]
Finish = Literal["gloss", "satin", "matte"]

COLOR_NAME = {
    "nardo_grey": "Nardo Grey",
    "satin_black": "Satin Black",
    "gloss_black": "Gloss Black",
    "miami_blue": "Miami Blue",
    "british_racing_green": "British Racing Green",
    "ruby_red": "Deep Ruby Red",
    "pearl_white": "Pearl White",
}

# These hex colors drive the deterministic recolor.
# Tweak anytime for better “wrap-accurate” colors.
COLOR_HEX = {
    "nardo_grey": "#9FA4A9",
    "satin_black": "#1A1A1A",
    "gloss_black": "#0B0B0B",
    "miami_blue": "#00B7D6",
    "british_racing_green": "#0B3D2E",
    "ruby_red": "#7A0F1C",
    "pearl_white": "#F2F2F0",
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


def img_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def download_image(url: str, timeout: int = 30) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data))


def hex_to_rgb(hexstr: str):
    hexstr = hexstr.strip().lstrip("#")
    return tuple(int(hexstr[i:i + 2], 16) for i in (0, 2, 4))


def clamp_mask(mask_l: Image.Image, threshold: int = 110) -> Image.Image:
    # blur then threshold for smoother edges (less jaggies)
    m = mask_l.filter(ImageFilter.GaussianBlur(radius=1.2))
    return m.point(lambda p: 255 if p >= threshold else 0)


def refine_mask_reduce_windows_wheels(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Reduce common bleed into windows/wheels/tires by removing pixels that are BOTH:
    - low brightness AND low saturation
    inside the mask region.
    """
    mask = ImageOps.autocontrast(mask_l.resize(original_rgb.size).convert("L"))

    hsv = original_rgb.convert("HSV")
    _, s, v = hsv.split()

    # thresholds tuned for car photos
    v_ok = v.point(lambda px: 255 if px > 60 else 0)  # not very dark
    s_ok = s.point(lambda px: 255 if px > 32 else 0)  # not very desaturated

    keep_if = ImageChops.lighter(v_ok, s_ok)
    cleaned = ImageChops.multiply(mask, keep_if)

    return clamp_mask(cleaned, threshold=120)


def make_panel_lines(original_rgb: Image.Image, body_mask: Image.Image) -> Image.Image:
    """
    Create subtle “panel gap” lines by extracting edges from the original image,
    then applying them only inside the body mask.
    """
    gray = ImageOps.grayscale(original_rgb)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(2.2)

    # Keep only strong edges
    edges = edges.point(lambda p: 255 if p > 45 else 0)

    # Slight blur so it looks like real seam shadow, not a hard outline
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Apply only on body
    lines = ImageChops.multiply(edges, body_mask)
    return lines


def apply_wrap_recolor(original_rgb: Image.Image, body_mask: Image.Image, color_hex: str) -> Image.Image:
    """
    Deterministic recolor:
    - Keep original shading/reflections by using luminance
    - Recolor body panels only
    """
    r, g, b = hex_to_rgb(color_hex)

    # shading from original
    base_gray = ImageOps.grayscale(original_rgb)

    # Colorize: black->dark, white->target
    colored = ImageOps.colorize(base_gray, black=(0, 0, 0), white=(r, g, b)).convert("RGB")

    # Blend amount controls “wrap intensity”
    blended = Image.blend(original_rgb, colored, 0.78)

    # composite only on masked body
    out = Image.composite(blended, original_rgb, body_mask)
    return out


def add_clearcoat_highlights(img_rgb: Image.Image, body_mask: Image.Image, strength: float) -> Image.Image:
    """
    Boost highlights in a clearcoat-like way without changing geometry.
    """
    gray = ImageOps.grayscale(img_rgb)

    # highlight map = only bright areas
    highlights = gray.point(lambda p: max(0, p - 160) * 2)
    highlights = highlights.filter(ImageFilter.GaussianBlur(radius=2.2))

    # apply only on body
    highlights = ImageChops.multiply(highlights, body_mask)

    # convert to RGB overlay
    overlay = Image.merge("RGB", (highlights, highlights, highlights))
    out = ImageChops.screen(img_rgb, overlay)

    # control intensity
    out = Image.blend(img_rgb, out, strength)
    return out


def add_metallic_flake(img_rgb: Image.Image, body_mask: Image.Image, seed: int, strength: float) -> Image.Image:
    """
    Subtle metallic flake for gloss wraps:
    - Speckles only in highlight areas
    - Very light, premium, not glittery
    """
    import random
    rnd = random.Random(seed)

    w, h = img_rgb.size

    # Build monochrome noise
    noise = Image.new("L", (w, h))
    px = noise.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = rnd.randrange(256)

    # keep only rare bright specks
    flakes = noise.point(lambda p: 255 if p > 252 else 0)
    flakes = flakes.filter(ImageFilter.GaussianBlur(radius=0.6))

    # only in highlight areas of the image
    gray = ImageOps.grayscale(img_rgb)
    hi = gray.point(lambda p: 255 if p > 175 else 0).filter(ImageFilter.GaussianBlur(radius=1.0))

    flakes = ImageChops.multiply(flakes, hi)
    flakes = ImageChops.multiply(flakes, body_mask)

    # convert to RGB and screen it
    fl_rgb = Image.merge("RGB", (flakes, flakes, flakes))
    screened = ImageChops.screen(img_rgb, fl_rgb)
    out = Image.blend(img_rgb, screened, strength)
    return out


def finish_tuning(img_rgb: Image.Image, finish: Finish) -> Image.Image:
    if finish == "gloss":
        img_rgb = ImageEnhance.Contrast(img_rgb).enhance(1.07)
        img_rgb = ImageEnhance.Color(img_rgb).enhance(1.08)
    elif finish == "satin":
        img_rgb = ImageEnhance.Contrast(img_rgb).enhance(1.03)
        img_rgb = ImageEnhance.Color(img_rgb).enhance(1.03)
    elif finish == "matte":
        img_rgb = ImageEnhance.Contrast(img_rgb).enhance(0.97)
        img_rgb = ImageEnhance.Color(img_rgb).enhance(0.93)
    return img_rgb


@app.get("/")
def root():
    return {"message": "RetroClean Wrap Visualizer API. Use /health or POST /render."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),  # kept for UI consistency, not required by deterministic pipeline
    color: str = Form(...),
    finish: Finish = Form(...),
):
    # cooldown
    now = time.time()
    last = getattr(app.state, "last_call_ts", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail=f"Too many requests. Please wait ~{COOLDOWN_SECONDS}s and try again.")
    app.state.last_call_ts = now

    if color not in COLOR_NAME or color not in COLOR_HEX:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    # load image
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

    # env vars
    token = os.getenv("REPLICATE_API_TOKEN")
    seg_version = os.getenv("REPLICATE_SEG_VERSION")
    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    if not seg_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION on server.")

    os.environ["REPLICATE_API_TOKEN"] = token

    original_data_url = img_to_data_url(pil, fmt="PNG")

    # 1) segmentation
    try:
        seg_out = replicate.run(
            seg_version,
            input={
                "image": original_data_url,
                # Helps SAM focus a bit better:
                "text_prompt": "car body panels",
            },
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (segmentation): {str(e)}")

    # extract mask url
    mask_url = None
    if isinstance(seg_out, dict):
        mask_url = seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    elif isinstance(seg_out, list) and seg_out:
        mask_url = seg_out[0]
    elif isinstance(seg_out, str):
        mask_url = seg_out

    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")

    # download mask
    try:
        mask_img = download_image(mask_url).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    # refine mask (reduce window/wheel bleed)
    base_mask = ImageOps.autocontrast(mask_img.resize(pil.size).convert("L"))
    try:
        body_mask = refine_mask_reduce_windows_wheels(pil, base_mask)
    except Exception:
        body_mask = clamp_mask(base_mask, threshold=120)

    # 2) deterministic recolor
    wrapped = apply_wrap_recolor(pil, body_mask, COLOR_HEX[color])

    # 3) panel separation (subtle seam shadows)
    lines = make_panel_lines(pil, body_mask)
    # Darken along lines slightly
    lines_rgb = Image.merge("RGB", (lines, lines, lines))
    wrapped = ImageChops.multiply(wrapped, ImageEnhance.Brightness(lines_rgb).enhance(0.88))

    # 4) finish tuning + clearcoat + metallic (gloss only)
    wrapped = finish_tuning(wrapped, finish)

    if finish == "gloss":
        wrapped = add_clearcoat_highlights(wrapped, body_mask, strength=0.55)

        # seed based on image+settings so flakes are stable per photo
        seed_src = raw + f"{color}|{finish}".encode("utf-8")
        seed = int(hashlib.sha256(seed_src).hexdigest()[:8], 16)
        wrapped = add_metallic_flake(wrapped, body_mask, seed=seed, strength=0.12)

    after_data_url = img_to_data_url(wrapped, fmt="JPEG", quality=92)

    return {
        "before": original_data_url,
        "after": after_data_url,   # <-- DATA URL (not http)
        "mask": mask_url,
    }
