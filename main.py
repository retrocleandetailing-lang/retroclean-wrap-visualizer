import os
import io
import base64
import time
import urllib.request
import hashlib
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageFilter

import replicate
from replicate.exceptions import ReplicateError


# Only 1 Replicate call now (segmentation), so cooldown can be much shorter.
COOLDOWN_SECONDS = 12

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

# Hex targets for deterministic recolor
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


def hex_to_rgb(hexstr: str) -> tuple[int, int, int]:
    s = hexstr.strip().lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def clamp_mask(mask_l: Image.Image, size: tuple[int, int]) -> Image.Image:
    m = mask_l.resize(size).convert("L")
    m = ImageOps.autocontrast(m)
    # soften edges a bit to avoid “stair steps”
    m = m.filter(ImageFilter.GaussianBlur(radius=1.1))
    return m


def clean_mask_remove_windows_wheels(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Heuristic cleanup to reduce windows/wheels being recolored:
    - dark + low saturation areas inside the mask get removed
    """
    mask = clamp_mask(mask_l, original_rgb.size)

    hsv = original_rgb.convert("HSV")
    _, s, v = hsv.split()

    # thresholds tuned for typical car photos
    dark = v.point(lambda px: 255 if px < 60 else 0)
    lowsat = s.point(lambda px: 255 if px < 45 else 0)

    # areas likely to be windows/wheels = dark AND low sat
    win_wheel = ImageChops.multiply(dark, lowsat)

    # remove those areas from mask
    win_wheel_inv = ImageOps.invert(win_wheel)
    cleaned = ImageChops.multiply(mask, win_wheel_inv)

    # tighten + smooth
    cleaned = cleaned.filter(ImageFilter.GaussianBlur(radius=1.0))
    cleaned = cleaned.point(lambda px: 255 if px > 95 else 0)

    return cleaned


def make_panel_lines(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Panel separation: accentuate existing edges (door lines, hood lines) only on painted area.
    We DO NOT invent new lines; we just boost the ones already present.
    """
    gray = ImageOps.grayscale(original_rgb)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    # keep only “strong” edges
    edges = edges.point(lambda px: 255 if px > 140 else 0)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.6))

    # restrict to masked paint area
    edges = ImageChops.multiply(edges, mask_l)

    return edges


def make_highlight_map(original_rgb: Image.Image, mask_l: Image.Image) -> Image.Image:
    """
    Clearcoat highlights: extract bright specular regions from original, then use as a map.
    """
    gray = ImageOps.grayscale(original_rgb)
    # high-pass-ish: emphasize local contrast
    blur = gray.filter(ImageFilter.GaussianBlur(radius=6))
    hi = ImageChops.subtract(gray, blur)
    hi = ImageOps.autocontrast(hi)

    # focus on brighter features
    hi = hi.point(lambda px: 255 if px > 155 else 0)
    hi = hi.filter(ImageFilter.GaussianBlur(radius=2.0))

    # restrict to paint mask
    hi = ImageChops.multiply(hi, mask_l)
    return hi


def screen_blend(base: Image.Image, top: Image.Image, top_alpha_l: Image.Image) -> Image.Image:
    """
    Screen blend top over base, controlled by alpha (L image).
    """
    base = base.convert("RGB")
    top = top.convert("RGB")
    alpha = top_alpha_l.convert("L")

    # screen = 255 - ( (255-base)*(255-top) / 255 )
    inv_base = ImageOps.invert(base)
    inv_top = ImageOps.invert(top)
    mult = ImageChops.multiply(inv_base, inv_top)
    screen = ImageOps.invert(mult)

    return Image.composite(screen, base, alpha)


def recolor_body(
    original_rgb: Image.Image,
    mask_l: Image.Image,
    target_rgb: tuple[int, int, int],
    finish: Finish,
    intensity: float,
    seed: int,
) -> Image.Image:
    """
    Deterministic recolor preserving shading and reflections.
    Adds: panel line separation, clearcoat highlights, metallic flake (gloss only).
    """

    intensity = max(0.10, min(float(intensity), 0.95))

    # base shading from original
    base_gray = ImageOps.grayscale(original_rgb)

    # colorize shading to target
    colored = ImageOps.colorize(base_gray, black=(0, 0, 0), white=target_rgb).convert("RGB")

    # blend target onto original inside mask (preserves reflections shape)
    blended = Image.blend(original_rgb, colored, intensity)
    wrapped = Image.composite(blended, original_rgb, mask_l)

    # panel lines
    panel_lines = make_panel_lines(original_rgb, mask_l)
    # darken panel lines subtly
    panel_dark = ImageOps.colorize(panel_lines, black=(0, 0, 0), white=(0, 0, 0)).convert("RGB")
    wrapped = Image.composite(ImageChops.subtract(wrapped, panel_dark), wrapped, panel_lines)

    # clearcoat highlight map
    hi = make_highlight_map(original_rgb, mask_l)

    if finish == "gloss":
        # stronger clearcoat highlight
        hi_stronger = ImageEnhance.Brightness(hi).enhance(1.15)
        hi_stronger = ImageEnhance.Contrast(hi_stronger).enhance(1.35)

        # white highlight layer
        white = Image.new("RGB", wrapped.size, (255, 255, 255))
        wrapped = screen_blend(wrapped, white, hi_stronger)

        # metallic flake: subtle micro-sparkle only in highlights
        # procedural noise, seeded by image+color for consistency
        flake = Image.effect_noise(wrapped.size, 18)  # 0-255 noise
        flake = flake.filter(ImageFilter.GaussianBlur(radius=0.7))
        flake = ImageEnhance.Contrast(flake).enhance(2.1)
        flake = flake.point(lambda px: 255 if px > 235 else 0)  # tiny specks

        # only appear where highlights are (and inside mask)
        flake = ImageChops.multiply(flake, hi_stronger)
        flake = flake.point(lambda px: 255 if px > 40 else 0)

        wrapped = screen_blend(wrapped, white, flake)

        wrapped = ImageEnhance.Contrast(wrapped).enhance(1.06)
        wrapped = ImageEnhance.Color(wrapped).enhance(1.08)

    elif finish == "satin":
        hi_soft = ImageEnhance.Brightness(hi).enhance(0.95)
        hi_soft = hi_soft.filter(ImageFilter.GaussianBlur(radius=1.5))
        white = Image.new("RGB", wrapped.size, (245, 245, 245))
        wrapped = screen_blend(wrapped, white, hi_soft)
        wrapped = ImageEnhance.Contrast(wrapped).enhance(1.02)
        wrapped = ImageEnhance.Color(wrapped).enhance(1.03)

    else:  # matte
        # matte: suppress highlights a little
        wrapped = ImageEnhance.Contrast(wrapped).enhance(0.97)
        wrapped = ImageEnhance.Color(wrapped).enhance(0.95)

    return wrapped


@app.get("/")
def root():
    return {"message": "RetroClean Wrap Visualizer API. Use /health or POST /render."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),      # kept for your UI; not required for recolor
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.55),  # becomes recolor intensity (NOT diffusion strength)
):
    # cooldown
    now = time.time()
    last = getattr(app.state, "last_call_ts", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail=f"Too many requests. Please wait ~{COOLDOWN_SECONDS}s and try again.")
    app.state.last_call_ts = now

    if color not in COLOR_NAME or color not in COLOR_HEX:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    # Load image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = pil.size
        if w < 900 or h < 900:
            raise HTTPException(status_code=400, detail="Photo too small. Use at least ~900px on each side.")
        # keep reasonable CPU cost
        if w > 2200 or h > 2200:
            pil.thumbnail((2200, 2200))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Env vars
    token = os.getenv("REPLICATE_API_TOKEN")
    seg_version = os.getenv("REPLICATE_SEG_VERSION")

    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    if not seg_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION on server.")

    os.environ["REPLICATE_API_TOKEN"] = token

    # Original for response (before) + segmentation input
    before_data_url = img_to_data_url(pil, fmt="JPEG", quality=92)
    seg_input_data_url = img_to_data_url(pil, fmt="PNG")

    # 1) Segmentation
    try:
        seg_out = replicate.run(
            seg_version,
            input={
                "image": seg_input_data_url,
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

    # Download mask
    try:
        mask_img = download_image(mask_url).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    # Clean mask
    try:
        cleaned = clean_mask_remove_windows_wheels(pil, mask_img)
    except Exception:
        cleaned = clamp_mask(mask_img, pil.size)

    # seed for deterministic flake “feel”
    h = hashlib.sha256(raw + color.encode("utf-8") + finish.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)

    target_rgb = hex_to_rgb(COLOR_HEX[color])

    after_img = recolor_body(
        original_rgb=pil,
        mask_l=cleaned,
        target_rgb=target_rgb,
        finish=finish,
        intensity=float(strength),
        seed=seed,
    )

    after_data_url = img_to_data_url(after_img, fmt="JPEG", quality=92)

    return {
        "before": before_data_url,
        "after": after_data_url,
        "mask": mask_url,  # keep for debugging
    }
