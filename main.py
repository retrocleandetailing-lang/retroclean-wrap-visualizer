import os
import io
import base64
import time
from typing import Literal

import requests
from PIL import Image, ImageOps, ImageEnhance

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import replicate
from replicate.exceptions import ReplicateError


# -----------------------------
# SETTINGS
# -----------------------------
COOLDOWN_SECONDS = 20  # 1 Replicate call per request now; 20s is a safe cooldown

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

# Hex colors for photoreal wrap overlays (tweak later)
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
    angle: Angle = Form(...),     # currently unused in overlay mode, kept for UI compatibility
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.45), # we’ll map this to overlay intensity below
):
    # -----------------------------
    # Cooldown / basic rate limiting
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

    # Clamp strength to safe range (we'll reuse it as overlay amount)
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
    # Replicate token + model version
    # -----------------------------
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    os.environ["REPLICATE_API_TOKEN"] = token

    seg_version = os.getenv("REPLICATE_SEG_VERSION")
    if not seg_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION on server.")

    # Convert original image to data url (works with Replicate)
    original_data_url = img_to_data_url(pil, fmt="JPEG")

    # -----------------------------
    # 1) Segmentation (Replicate)
    # -----------------------------
    try:
        seg_out = replicate.run(
            seg_version,
            input={
                "image": original_data_url,
                "text_prompt": "car body",  # better than "car" for wraps
            },
        )
    except ReplicateError as e:
        # Bubble up as 429 so Webflow cooldown triggers nicely
        raise HTTPException(status_code=429, detail=f"Replicate segmentation error: {e}")

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

    # -----------------------------
    # 2) Download mask image
    # -----------------------------
    try:
        mask_bytes = requests.get(str(mask_url), timeout=30).content
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

    # Ensure mask matches photo size
    mask_img = mask_img.resize(pil.size)

    # Improve mask contrast
    mask = ImageOps.autocontrast(mask_img)

    # Optional: soften edges slightly to reduce harsh boundaries
    # (comment this out if you prefer sharper edges)
    # mask = mask.filter(ImageFilter.GaussianBlur(radius=1))

    # -----------------------------
    # 3) Wrap color overlay (photoreal, preserves lighting)
    # -----------------------------
    target_hex = COLOR_HEX[color]
    r, g, b = hex_to_rgb(target_hex)

    # Take luminance from original so reflections/shadows stay real
    base_gray = ImageOps.grayscale(pil)

    # Colorize grayscale into target color while keeping shading
    colored = ImageOps.colorize(base_gray, black=(0, 0, 0), white=(r, g, b)).convert("RGB")

    # Map "strength" to overlay intensity (lower strength = more subtle, higher = more color)
    # Good range is ~0.55 to 0.85
    overlay = 0.55 + (0.30 * strength)

    blended = Image.blend(pil, colored, overlay)

    # Composite ONLY on masked area
    wrapped = Image.composite(
        blended,
        pil,
        mask  # white areas apply wrap, black areas keep original
    )

    # Finish tweaks
    if finish == "gloss":
        wrapped = ImageEnhance.Contrast(wrapped).enhance(1.08)
        wrapped = ImageEnhance.Color(wrapped).enhance(1.12)
    elif finish == "satin":
        wrapped = ImageEnhance.Contrast(wrapped).enhance(1.03)
        wrapped = ImageEnhance.Color(wrapped).enhance(1.05)
    elif finish == "matte":
        wrapped = ImageEnhance.Contrast(wrapped).enhance(0.96)
        wrapped = ImageEnhance.Color(wrapped).enhance(0.92)

    after_data_url = img_to_data_url(wrapped, fmt="JPEG")

    # Return data URLs so Webflow can display instantly
    return {"before": original_data_url, "after": after_data_url, "mask": str(mask_url)}
