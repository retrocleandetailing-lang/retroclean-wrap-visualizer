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

def to_64(x: int) -> int:
    return max(64, (x // 64) * 64)

def img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read()))

def build_prompt(color: str, finish: Finish, angle: Angle) -> str:
    finish_text = {
        "gloss": "high-gloss vinyl wrap with subtle clearcoat highlights",
        "satin": "satin vinyl wrap with soft reflections",
        "matte": "matte vinyl wrap with diffused reflections",
    }[finish]

    return (
        f"Photorealistic inpainting of the SAME car. "
        f"DO NOT change the car, camera angle, background, lighting, or perspective. "
        f"ONLY recolor the painted body panels to {color} {finish_text}. "
        f"Preserve wheels, windows, headlights, trim, badges, and environment. "
        f"Maintain exact proportions and realism."
    )

def negative_prompt() -> str:
    return (
        "different car, different angle, changed vehicle, changed background, "
        "cartoon, CGI, illustration, low quality, blurry, warped"
    )

@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
):
    now = time.time()
    last = getattr(app.state, "last_call", 0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(429, "Please wait and try again.")
    app.state.last_call = now

    if color not in COLOR_MAP:
        raise HTTPException(400, "Unsupported color.")

    raw = await image.read()
    base = Image.open(io.BytesIO(raw)).convert("RGB")

    # Resize to SD-safe dimensions
    w, h = base.size
    base = base.resize((to_64(w), to_64(h)))

    token = os.getenv("REPLICATE_API_TOKEN")
    seg = os.getenv("REPLICATE_SEG_VERSION")
    img = os.getenv("REPLICATE_IMG_VERSION")

    if not token or not seg or not img:
        raise HTTPException(500, "Missing Replicate env vars.")

    os.environ["REPLICATE_API_TOKEN"] = token

    base_url = img_to_data_url(base)

    # --- SEGMENT ---
    try:
        seg_out = replicate.run(seg, {
            "image": base_url,
            "text_prompt": "car body panels",
        })
    except ReplicateError as e:
        raise HTTPException(429, str(e))

    mask_url = seg_out[0] if isinstance(seg_out, list) else seg_out
    mask = download_image(mask_url).convert("L").resize(base.size)

    # HARD threshold mask (critical)
    mask = ImageOps.autocontrast(mask)
    mask = mask.point(lambda x: 255 if x > 200 else 0)

    mask_url = img_to_data_url(mask)

    time.sleep(10)

    # --- INPAINT ---
    try:
        out = replicate.run(img, {
            "image": base_url,
            "mask": mask_url,
            "prompt": build_prompt(COLOR_MAP[color], finish, angle),
            "negative_prompt": negative_prompt(),
            "strength": 0.20,          # VERY IMPORTANT
            "guidance_scale": 5.0,     # VERY IMPORTANT
            "num_inference_steps": 28,
            "num_outputs": 1,
        })
    except ReplicateError as e:
        raise HTTPException(429, str(e))

    return {
        "before": base_url,
        "after": out[0] if isinstance(out, list) else out,
    }
