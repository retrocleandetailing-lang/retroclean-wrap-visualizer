import os
import io
import base64
import time
import urllib.request
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter

import replicate
from replicate.exceptions import ReplicateError

# Load environment variables from .env file
load_dotenv()

COOLDOWN_SECONDS = 10

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

# Default inpainting model - stability-ai/sdxl with inpainting support
DEFAULT_INPAINT_MODEL = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

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
    """Round dimension to nearest multiple of 64 for SD compatibility."""
    return max(64, (x // 64) * 64)


def img_to_data_url(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def download_image(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read()))


def build_prompt(color: str, finish: Finish, angle: Angle) -> str:
    """Build the inpainting prompt for the wrap color."""
    finish_text = {
        "gloss": "high-gloss vinyl wrap with reflective shine",
        "satin": "satin vinyl wrap with soft sheen",
        "matte": "matte vinyl wrap with no reflections",
    }[finish]

    # Keep prompt simple and focused on just the color/finish
    return (
        f"{color} {finish_text} on car body panels, "
        f"professional automotive photography, photorealistic, high quality, "
        f"perfect lighting, smooth paint finish"
    )


def negative_prompt() -> str:
    """Return negative prompt to avoid unwanted changes."""
    return (
        "different car, different vehicle, changed shape, changed angle, "
        "changed background, changed perspective, distorted, deformed, "
        "cartoon, CGI, illustration, low quality, blurry, artifacts"
    )


def prepare_mask_for_inpainting(mask: Image.Image) -> Image.Image:
    """
    Prepare mask for inpainting. Standard SD inpainting convention:
    - WHITE (255) = area to INPAINT/CHANGE
    - BLACK (0) = area to KEEP unchanged
    """
    # Maximize contrast in the mask
    mask = ImageOps.autocontrast(mask)
    
    # Apply threshold to create binary mask
    mask = mask.point(lambda x: 255 if x > 128 else 0)
    
    # Apply slight feathering for smoother edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Re-threshold but keep some gradient for blending
    mask = mask.point(lambda x: min(255, max(0, int((x - 50) * 1.5))))
    
    return mask


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    debug: bool = Form(False),
):
    """
    Render a car wrap visualization.
    
    This endpoint:
    1. Segments the car body panels from the uploaded image
    2. Uses inpainting to change ONLY the body panel colors
    3. Returns the before/after images
    """
    now = time.time()
    last = getattr(app.state, "last_call", 0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(429, "Please wait and try again.")
    app.state.last_call = now

    if color not in COLOR_MAP:
        raise HTTPException(400, "Unsupported color.")

    # Load and prepare the uploaded image
    raw = await image.read()
    base = Image.open(io.BytesIO(raw)).convert("RGB")

    # Resize to SD-safe dimensions (multiples of 64)
    w, h = base.size
    new_w, new_h = to_64(w), to_64(h)
    base = base.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Get API credentials
    token = os.getenv("REPLICATE_API_TOKEN")
    seg_model = os.getenv("REPLICATE_SEG_VERSION")
    inpaint_model = os.getenv("REPLICATE_IMG_VERSION", DEFAULT_INPAINT_MODEL)

    if not token or not seg_model:
        raise HTTPException(500, "Missing Replicate env vars.")

    os.environ["REPLICATE_API_TOKEN"] = token

    base_url = img_to_data_url(base)

    # --- STEP 1: SEGMENT CAR BODY PANELS ---
    try:
        seg_out = replicate.run(seg_model, {
            "image": base_url,
            "text_prompt": "car body panels, car doors, car hood, car fenders, car bumpers",
        })
    except ReplicateError as e:
        raise HTTPException(429, f"Segmentation failed: {str(e)}")

    # Download and process the segmentation mask
    seg_mask_url = seg_out[0] if isinstance(seg_out, list) else seg_out
    mask = download_image(seg_mask_url).convert("L").resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Prepare mask for inpainting (white = areas to change)
    mask = prepare_mask_for_inpainting(mask)
    mask_data_url = img_to_data_url(mask)

    # Return debug info if requested
    if debug:
        return {
            "before": base_url,
            "mask": mask_data_url,
            "message": "Debug mode - showing mask only"
        }

    # Brief delay to avoid rate limiting
    time.sleep(5)

    # --- STEP 2: INPAINT WITH NEW COLOR ---
    try:
        inpaint_out = replicate.run(inpaint_model, {
            "image": base_url,
            "mask": mask_data_url,
            "prompt": build_prompt(COLOR_MAP[color], finish, angle),
            "negative_prompt": negative_prompt(),
            "prompt_strength": 0.45,  # Lower to preserve more of original image
            "num_inference_steps": 40,  # More steps for better quality
            "guidance_scale": 9.0,  # Higher to follow prompt more closely
            "num_outputs": 1,
            "output_format": "png",
        })
    except ReplicateError as e:
        raise HTTPException(429, f"Inpainting failed: {str(e)}")

    result_url = inpaint_out[0] if isinstance(inpaint_out, list) else inpaint_out

    return {
        "before": base_url,
        "after": result_url,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
