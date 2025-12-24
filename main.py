import os
import io
import base64
import time
import urllib.request
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter

import replicate
from replicate.exceptions import ReplicateError

# Load environment variables from .env file (local dev)
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

# Use env var REPLICATE_IMG_VERSION for your model; fallback is SDXL inpaint
DEFAULT_INPAINT_MODEL = (
    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
)

# If you don’t set REPLICATE_SEG_VERSION, you can optionally set a default here.
# Leaving None so you are forced to configure it properly in Render.
DEFAULT_SEG_MODEL: Optional[str] = None


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="RetroClean Wrap Visualizer API")

# CORS: allow your live domains + Webflow staging/preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://retrocleandetailing.com",
        "https://www.retrocleandetailing.com",
    ],
    allow_origin_regex=r"^https:\/\/.*\.webflow\.io$|^https:\/\/preview\.webflow\.com$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


# -----------------------------
# Helpers
# -----------------------------
def to_64(x: int) -> int:
    """Round dimension down to nearest multiple of 64 for SD compatibility."""
    return max(64, (x // 64) * 64)


def img_to_data_url(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def download_image(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read()))


def build_prompt(color: str, finish: Finish, angle: Angle) -> str:
    """Build a tight inpainting prompt for wrap color."""
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
    """Negative prompt to avoid unwanted changes."""
    return (
        "different car, different vehicle, change wheels, change rims, change tires, "
        "change windows, change grille, change headlights, change taillights, "
        "change badges, change logo, change background, change lighting, "
        "change camera angle, change perspective, deformed, distorted, warped, "
        "cartoon, CGI, illustration, low quality, blurry, artifacts"
    )


def prepare_mask_for_inpainting(mask: Image.Image) -> Image.Image:
    """
    Prepare mask for inpainting. Standard convention:
    - WHITE (255) = area to CHANGE
    - BLACK (0) = area to KEEP
    Keep mask crisp to prevent “bleed” onto trim/badges.
    """
    m = mask.convert("L")
    m = ImageOps.autocontrast(m)
    m = m.point(lambda x: 255 if x > 128 else 0)
    m = m.filter(ImageFilter.GaussianBlur(radius=1))
    m = m.point(lambda x: 255 if x > 128 else 0)
    return m


def get_models() -> tuple[str, str]:
    """
    Returns (seg_model, inpaint_model).
    Enforces required env vars to avoid silent misconfig.
    """
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(500, "Missing REPLICATE_API_TOKEN.")

    seg_model = os.getenv("REPLICATE_SEG_VERSION") or DEFAULT_SEG_MODEL
    if not seg_model:
        raise HTTPException(500, "Missing REPLICATE_SEG_VERSION.")

    inpaint_model = os.getenv("REPLICATE_IMG_VERSION", DEFAULT_INPAINT_MODEL)

    # Replicate SDK reads token from env; set it explicitly
    os.environ["REPLICATE_API_TOKEN"] = token
    return seg_model, inpaint_model


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/cors-test")
async def cors_test():
    return {"ok": True}


@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    debug: bool = Form(False),
):
    """
    1) Segment car body panels
    2) Inpaint ONLY those panels to chosen color/finish
    """
    # Cooldown
    now = time.time()
    last = getattr(app.state, "last_call", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(429, "Please wait and try again.")
    app.state.last_call = now

    if color not in COLOR_MAP:
        raise HTTPException(400, "Unsupported color.")

    seg_model, inpaint_model = get_models()

    # Read image
    raw = await image.read()
    try:
        base = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    # Resize to SD-safe multiples of 64 (for SDXL-style models)
    w, h = base.size
    new_w, new_h = to_64(w), to_64(h)
    base = base.resize((new_w, new_h), Image.Resampling.LANCZOS)

    base_url = img_to_data_url(base)

    # STEP 1: segmentation (tight prompt)
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
    
    # STEP 2: inpainting
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

    return {
        "before": base_url,
        "after": result_url,
    }
