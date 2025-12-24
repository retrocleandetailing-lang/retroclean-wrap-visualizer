import os
import io
import base64
import time
import urllib.request
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter

import replicate
from replicate.exceptions import ReplicateError

# Load .env locally (Render ignores unless you add env vars in dashboard)
load_dotenv()

# -----------------------------
# Config
# -----------------------------
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

# Default fallback if REPLICATE_IMG_VERSION isn't set (works, but not ControlNet)
DEFAULT_INPAINT_MODEL = (
    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
)

# You can set this in Render as REPLICATE_SEG_VERSION
DEFAULT_SEG_MODEL: Optional[str] = None


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="RetroClean Wrap Visualizer API")

# CORS: Updated configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://retrocleandetailing.com",
        "https://www.retrocleandetailing.com",
        "http://localhost:3000",  # for local testing
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.webflow\.io|https://.*\.webflow\.com",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


# -----------------------------
# Helpers
# -----------------------------
def to_64(x: int) -> int:
    """Round dimension down to nearest multiple of 64 (SDXL-friendly)."""
    return max(64, (x // 64) * 64)


def img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read()))


def build_prompt(color: str, finish: Finish, angle: Angle) -> str:
    finish_text = {
        "gloss": "high-gloss vinyl wrap with realistic reflections",
        "satin": "satin vinyl wrap with soft sheen",
        "matte": "matte vinyl wrap with minimal reflections",
    }[finish]

    angle_text = {
        "front_3q": "front three-quarter view",
        "side": "side profile view",
        "rear_3q": "rear three-quarter view",
    }[angle]

    return (
        f"Recolor ONLY the masked painted body panels to {color}. "
        f"Apply {finish_text}. "
        f"Preserve everything else exactly: same {angle_text}, same background, "
        f"same wheels, tires, windows, grille, lights, badges, logos, trim, "
        f"same panel gaps, same perspective. Photorealistic."
    )


def negative_prompt() -> str:
    return (
        "different car, different vehicle, change wheels, change rims, change tires, "
        "change windows, change grille, change headlights, change taillights, "
        "change badges, change logo, change background, change lighting, "
        "change camera angle, change perspective, deformed, distorted, warped, "
        "cartoon, CGI, illustration, low quality, blurry, artifacts"
    )


def prepare_mask_for_inpainting(mask: Image.Image) -> Image.Image:
    """
    WHITE = change, BLACK = keep. Keep it crisp to avoid bleed.
    """
    m = mask.convert("L")
    m = ImageOps.autocontrast(m)
    m = m.point(lambda x: 255 if x > 128 else 0)
    m = m.filter(ImageFilter.GaussianBlur(radius=1))
    m = m.point(lambda x: 255 if x > 128 else 0)
    return m


def get_models() -> tuple[str, str]:
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(500, "Missing REPLICATE_API_TOKEN.")

    seg_model = os.getenv("REPLICATE_SEG_VERSION") or DEFAULT_SEG_MODEL
    if not seg_model:
        raise HTTPException(500, "Missing REPLICATE_SEG_VERSION.")

    inpaint_model = os.getenv("REPLICATE_IMG_VERSION", DEFAULT_INPAINT_MODEL)

    # Replicate SDK reads from env
    os.environ["REPLICATE_API_TOKEN"] = token
    return seg_model, inpaint_model


# -----------------------------
# Debug / Health
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/cors-test")
async def cors_test():
    return {"ok": True}


@app.get("/debug-origin")
async def debug_origin(request: Request):
    """
    Call this from the Webflow page console to see what Origin is being sent.
    If fetch fails, it's still CORS/network before FastAPI is reached.
    """
    return {
        "origin": request.headers.get("origin"),
        "host": request.headers.get("host"),
        "user_agent": request.headers.get("user-agent"),
    }


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    debug: bool = Form(False),
):
    # Cooldown
    now = time.time()
    last = getattr(app.state, "last_call", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(429, "Please wait and try again.")
    app.state.last_call = now

    if color not in COLOR_MAP:
        raise HTTPException(400, "Unsupported color.")

    seg_model, inpaint_model = get_models()

    # Read uploaded image
    raw = await image.read()
    try:
        base = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    # Resize for SDXL-style inpainting models
    w, h = base.size
    new_w, new_h = to_64(w), to_64(h)
    base = base.resize((new_w, new_h), Image.Resampling.LANCZOS)

    base_url = img_to_data_url(base)

    # STEP 1: Segmentation (tight prompt)
    seg_prompt = (
        "car painted body panels only, doors, hood, fenders, quarter panels, roof, bumpers, "
        "exclude windows, windshield, grille, headlights, taillights, wheels, tires, badges, logo"
    )

    try:
        seg_out = replicate.run(
            seg_model,
            {
                "image": base_url,
                "text_prompt": seg_prompt,
            },
        )
    except ReplicateError as e:
        raise HTTPException(429, f"Segmentation failed: {str(e)}")

    seg_mask_url = seg_out[0] if isinstance(seg_out, list) else seg_out

    try:
        mask = download_image(seg_mask_url).convert("L").resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )
    except Exception:
        raise HTTPException(500, "Failed to download segmentation mask.")

    mask = prepare_mask_for_inpainting(mask)
    mask_url = img_to_data_url(mask)

    if debug:
        return {
            "before": base_url,
            "mask": mask_url,
            "seg_model": seg_model,
            "inpaint_model": inpaint_model,
        }

    # Brief delay to avoid rate limiting
    time.sleep(10)
    
    # STEP 2: Inpaint
    try:
        inpaint_out = replicate.run(
            inpaint_model,
            {
                "image": base_url,
                "mask": mask_url,
                "prompt": build_prompt(COLOR_MAP[color], finish, angle),
                "negative_prompt": negative_prompt(),
                "prompt_strength": 0.45,
                "num_inference_steps": 40,
                "guidance_scale": 9.0,
                "num_outputs": 1,
                "output_format": "png",
            },
        )
    except ReplicateError as e:
        raise HTTPException(429, f"Inpainting failed: {str(e)}")

    result_url = inpaint_out[0] if isinstance(inpaint_out, list) else inpaint_out

    return {"before": base_url, "after": result_url}
