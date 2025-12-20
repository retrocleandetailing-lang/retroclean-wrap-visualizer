import os
import io
import base64
import time

COOLDOWN_SECONDS = 60  # 6/min means 1 every 10s max; set 12 to be safe

from starlette.requests import Request
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import replicate
from replicate.exceptions import ReplicateError

# IMPORTANT:
# 1) Put your Replicate token into Render (Environment Variables), NOT in this file.
# 2) Put your Replicate model VERSION IDs into Render too (we’ll do that).

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

def img_to_data_url(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def build_prompt(color_name: str, finish: Finish, angle: Angle) -> str:
    finish_text = {
        "gloss": "glossy vinyl wrap",
        "satin": "satin vinyl wrap",
        "matte": "matte vinyl wrap",
    }[finish]

    angle_hint = {
        "front_3q": "front three-quarter angle",
        "side": "side profile angle",
        "rear_3q": "rear three-quarter angle",
    }[angle]

    return (
        f"Photorealistic car wrap preview. Keep the same car, same body shape, "
        f"same camera angle ({angle_hint}), and same background. "
        f"ONLY change the painted body panels to a {color_name} {finish_text}. "
        f"Do not alter wheels, headlights, windows, trim, badges, reflections shape, "
        f"panel gaps, environment, or perspective. Preserve realistic reflections and shading."
    )

def negative_prompt() -> str:
    return (
        "cartoon, anime, illustration, CGI, distorted body, changed wheels, changed background, "
        "warped panels, melted shapes, extra parts, text, watermark, logo, blurry, low quality"
    )

app = FastAPI(title="RetroClean Wrap Visualizer API")

# Allow your Webflow domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
  "https://retrocleandetailing.com",
  "https://www.retrocleandetailing.com",
],     # testing
    allow_credentials=False, # IMPORTANT
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
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.45),
):
    # Cooldown to reduce Replicate throttling (2 predictions per request)
    now = time.time()
    if not hasattr(app.state, "last_call_ts"):
        app.state.last_call_ts = 0.0

    if now - app.state.last_call_ts < COOLDOWN_SECONDS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait about 60 seconds and try again."
        )

    app.state.last_call_ts = now

    # Validate
    if color not in COLOR_MAP:
        raise HTTPException(status_code=400, detail="Unsupported color.")

    # Load image
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

    # Secrets from environment
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_API_TOKEN on server.")
    os.environ["REPLICATE_API_TOKEN"] = token

    seg_version = os.getenv("REPLICATE_SEG_VERSION")
    img_version = os.getenv("REPLICATE_IMG_VERSION")
    if not seg_version or not img_version:
        raise HTTPException(status_code=500, detail="Missing REPLICATE_SEG_VERSION or REPLICATE_IMG_VERSION on server.")

    original_data_url = img_to_data_url(pil, fmt="JPEG")

    # 1) Segmentation
    try:
        seg_out = replicate.run(
            seg_version,
            input={"image": original_data_url, "text_prompt": "car body"},
        )
    except ReplicateError as e:
        raise HTTPException(status_code=429, detail=f"Replicate error (step 1/2): {e}")

    mask_url = None
    if isinstance(seg_out, dict):
        mask_url = seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    elif isinstance(seg_out, list) and seg_out:
        mask_url = seg_out[0]
    elif isinstance(seg_out, str):
        mask_url = seg_out

    if not mask_url:
        raise HTTPException(status_code=500, detail="Segmentation failed (no mask returned).")
        
import requests
from PIL import ImageOps, ImageEnhance

def hex_to_rgb(hexstr: str):
    hexstr = hexstr.strip().lstrip("#")
    return tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))

# Add hex colors for more accurate wraps
COLOR_HEX = {
    "nardo_grey": "#9FA4A9",
    "satin_black": "#1A1A1A",
    "gloss_black": "#0B0B0B",
    "miami_blue": "#00B7D6",
    "british_racing_green": "#0B3D2E",
    "ruby_red": "#7A0F1C",
    "pearl_white": "#F2F2F0",
}

# --- download mask image from Replicate ---
try:
    mask_bytes = requests.get(mask_url, timeout=30).content
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
except Exception:
    raise HTTPException(status_code=500, detail="Could not download segmentation mask.")

# Ensure mask matches photo size
mask_img = mask_img.resize(pil.size)

# IMPORTANT:
# Most segmentation masks are white=object, black=background.
# We'll treat white areas as "car area to recolor".
mask = ImageOps.autocontrast(mask_img)

# --- Create a “shaded” recolor overlay ---
# Take luminance from original so reflections/shadows stay real
base_gray = ImageOps.grayscale(pil)

# Colorize grayscale into target color while keeping shading
target_hex = COLOR_HEX.get(color)
if not target_hex:
    raise HTTPException(status_code=400, detail="Unsupported color.")

r, g, b = hex_to_rgb(target_hex)

# Colorize grayscale: black->dark version, white->light version
# This preserves highlights + depth
colored = ImageOps.colorize(base_gray, black=(0,0,0), white=(r,g,b)).convert("RGB")

# Blend strength tuning (wrap intensity)
blend_strength = 0.75  # 0.6-0.85 works well

# Composite ONLY on the masked area
wrapped = Image.composite(
    Image.blend(pil, colored, blend_strength),
    pil,
    ImageOps.invert(mask)  # invert so mask white affects car, black leaves background alone
)

# Finish tweaks
if finish == "gloss":
    wrapped = ImageEnhance.Contrast(wrapped).enhance(1.08)
    wrapped = ImageEnhance.Color(wrapped).enhance(1.10)
elif finish == "satin":
    wrapped = ImageEnhance.Contrast(wrapped).enhance(1.03)
    wrapped = ImageEnhance.Color(wrapped).enhance(1.04)
elif finish == "matte":
    wrapped = ImageEnhance.Contrast(wrapped).enhance(0.96)
    wrapped = ImageEnhance.Color(wrapped).enhance(0.92)

after_data_url = img_to_data_url(wrapped, fmt="JPEG")

return {"before": original_data_url, "after": after_data_url, "mask": mask_url}
