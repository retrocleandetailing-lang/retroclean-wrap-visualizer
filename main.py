import os
import io
import base64
import time

LAST_CALL = 0
COOLDOWN_SECONDS = 12  # 6/min means 1 every 10s max; set 12 to be safe

from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import replicate

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




@app.get("/health")
def health():
    return {"ok": True}

@app.post("/render")
async def render(
    
    global LAST_CALL_TS
    now = time.time()
    if now - LAST_CALL_TS < COOLDOWN_SECONDS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait 10–15 seconds and try again."
        )
    LAST_CALL_TS = now

    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    strength: float = Form(0.45),
):
    # 1) Validate
    if color not in COLOR_MAP:
        raise HTTPException(400, "Unsupported color.")

    # 2) Load image
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = pil.size
        if w < 900 or h < 900:
            raise HTTPException(400, "Photo too small. Use at least ~900px on each side.")
        if w > 5000 or h > 5000:
            pil.thumbnail((5000, 5000))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    # 3) Read secrets from environment (set these in Render)
    # Replicate token
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(500, "Missing REPLICATE_API_TOKEN on server.")
    os.environ["REPLICATE_API_TOKEN"] = token

    # Replicate model version IDs (you will paste these from Replicate)
    seg_version = os.getenv("REPLICATE_SEG_VERSION")
    img_version = os.getenv("REPLICATE_IMG_VERSION")
    if not seg_version or not img_version:
        raise HTTPException(500, "Missing REPLICATE_SEG_VERSION or REPLICATE_IMG_VERSION on server.")

    original_data_url = img_to_data_url(pil, fmt="JPEG")

    # 4) Segmentation (make a mask)
    # NOTE: different models return different output formats. We handle common cases.
    seg_out = replicate.run(
    seg_version,
    input={
        "image": original_data_url,
        "text_prompt": "car"
    }
)

    mask_url = None
    if isinstance(seg_out, dict):
        mask_url = seg_out.get("mask") or seg_out.get("output") or seg_out.get("mask_url")
    elif isinstance(seg_out, list) and seg_out:
        mask_url = seg_out[0]
    elif isinstance(seg_out, str):
        mask_url = seg_out

    if not mask_url:
        raise HTTPException(500, "Segmentation failed (no mask returned).")

    # 5) Image-to-image with mask (recolor wrap)
    prompt = build_prompt(COLOR_MAP[color], finish, angle)

    out = replicate.run(
        img_version,
        input={
            "image": original_data_url,
            "mask": mask_url,
            "prompt": prompt,
            "negative_prompt": negative_prompt(),
            "strength": float(strength),
            "num_outputs": 1,
        },
    )

    result_url = None
    if isinstance(out, list) and out:
        result_url = out[0]
    elif isinstance(out, str):
        result_url = out

    if not result_url:
        raise HTTPException(500, "Render failed (no image returned).")

    return {
        "before": original_data_url,
        "after": result_url,
        "mask": mask_url,
    }
