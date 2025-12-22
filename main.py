import os
import io
import base64
import time
import urllib.request
from typing import Literal, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter

import replicate
from replicate.exceptions import ReplicateError

# -----------------------------
# Config
# -----------------------------
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

# STRICTEST option requested: FLUX inpainting + ControlNet
# Latest version hash from Replicate versions page
# (Input schema includes controlnet_conditioning_scale, guidance_scale, true_guidance_scale, etc.)
DEFAULT_INPAINT_MODEL = (
    "zsxkib/flux-dev-inpainting-controlnet:"
    "f9cb02cfd6b131af7ff9166b4bac5fdd2ed68bc282d2c049b95a23cea485e40d"
)

# Segmentation model (prompted). This outputs a mask URL we download.
DEFAULT_SEG_MODEL = (
    "tmappdev/lang-segment-anything:"
    "f4cbdd8c8ce5deac41ae87b9c77e2f950c08edfb1ca77fe763057d84fd4608fd"
)

# For FLUX ControlNet models, best results are often around ~768px.
# We’ll scale the longest edge to this (or smaller if the image is already smaller).
TARGET_LONG_EDGE = 1024  # you can set 768 if you want maximum consistency


app = FastAPI(title="RetroClean Wrap Visualizer API (Strict Inpaint + ControlNet)")

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


# -----------------------------
# Helpers
# -----------------------------
def round_to_multiple(x: int, m: int = 8) -> int:
    return max(m, (x // m) * m)


def resize_for_flux(img: Image.Image, long_edge: int = TARGET_LONG_EDGE) -> Image.Image:
    """Resize keeping aspect ratio; FLUX generally doesn’t require /64 like SDXL.
    We round to multiples of 8 for safer tensor shapes.
    """
    w, h = img.size
    if max(w, h) <= long_edge:
        new_w, new_h = round_to_multiple(w, 8), round_to_multiple(h, 8)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    scale = long_edge / float(max(w, h))
    new_w = round_to_multiple(int(w * scale), 8)
    new_h = round_to_multiple(int(h * scale), 8)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read()))


def build_prompt(color_name: str, finish: Finish, angle: Angle) -> str:
    """Keep prompt strict: repaint only; preserve all geometry/branding."""
    finish_text = {
        "gloss": "high-gloss vinyl wrap with realistic specular reflections",
        "satin": "satin vinyl wrap with soft sheen",
        "matte": "matte vinyl wrap with minimal reflections",
    }[finish]

    # Angle isn’t strictly needed, but you might want it to help the model not “recompose”.
    angle_text = {
        "front_3q": "front three-quarter view",
        "side": "side profile view",
        "rear_3q": "rear three-quarter view",
    }[angle]

    return (
        f"Recolor ONLY the masked painted body panels to {color_name}. "
        f"Apply {finish_text}. "
        f"Preserve everything else exactly: same {angle_text}, same background, "
        f"same wheels, same tires, same windows, same grille, same headlights, "
        f"same badges, same trim, same logo, same panel gaps, same reflections direction. "
        f"Photorealistic automotive photo. No redesign."
    )


def negative_prompt() -> str:
    """Aggressively forbid the stuff you said must never change."""
    return (
        "change wheels, change rims, change tires, change windows, change windshield, "
        "change grille, change headlights, change taillights, change badges, change logo, "
        "change body kit, change bumper, change hood, change roof, change mirrors, "
        "change background, change lighting, change camera angle, change perspective, "
        "different car, different vehicle, deformed, distorted, warped, messy edges, "
        "cartoon, CGI, illustration, low quality, blurry, artifacts"
    )


def prepare_mask_strict(mask_l: Image.Image) -> Image.Image:
    """
    STRICT mask for inpainting:
      - WHITE (255) = edit
      - BLACK (0) = keep
    For strictest preservation, keep edges crisp (minimal feather).
    """
    m = mask_l.convert("L")
    m = ImageOps.autocontrast(m)

    # Hard threshold for crisp separation
    m = m.point(lambda x: 255 if x > 128 else 0)

    # Very light blur (optional) to reduce jagged edges.
    # Keep this small; big feathering causes bleed onto badges/trim.
    m = m.filter(ImageFilter.GaussianBlur(radius=1))

    # Re-binarize after blur to keep it strict
    m = m.point(lambda x: 255 if x > 128 else 0)

    return m


def stable_seed(angle: Angle, color_key: str, finish: Finish) -> int:
    """Deterministic seed for consistent output per selection."""
    s = f"{angle}|{color_key}|{finish}"
    # Simple stable hash -> 32-bit int
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


# -----------------------------
# API
# -----------------------------
@app.post("/render")
async def render(
    image: UploadFile = File(...),
    angle: Angle = Form(...),
    color: str = Form(...),
    finish: Finish = Form(...),
    debug: bool = Form(False),
    # Optional overrides
    seed: Optional[int] = Form(None),
):
    """
    Strict wrap recolor:
      1) Segment paintable body panels (mask)
      2) Inpaint ONLY masked region using FLUX Inpainting + ControlNet
    """

    # Rate limit / cooldown
    now = time.time()
    last = getattr(app.state, "last_call", 0.0)
    if now - last < COOLDOWN_SECONDS:
        raise HTTPException(429, "Please wait and try again.")
    app.state.last_call = now

    if color not in COLOR_MAP:
        raise HTTPException(400, "Unsupported color.")

    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(500, "Missing REPLICATE_API_TOKEN env var.")
    os.environ["REPLICATE_API_TOKEN"] = token

    # Models (allow env overrides)
    seg_model = os.getenv("REPLICATE_SEG_VERSION", DEFAULT_SEG_MODEL)
    inpaint_model = os.getenv("REPLICATE_INPAINT_VERSION", DEFAULT_INPAINT_MODEL)

    # Load image
    raw = await image.read()
    try:
        base = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image upload.")

    # Resize for FLUX
    base = resize_for_flux(base, TARGET_LONG_EDGE)
    w, h = base.size
    base_url = img_to_data_url(base)

    # -----------------------------
    # STEP 1: SEGMENT BODY PANELS
    # -----------------------------
    # IMPORTANT: You may need to tune the text prompt.
    # Goal: ONLY painted panels. Not glass, wheels, lights, badges.
    seg_prompt = (
        "car painted body panels only, paint, doors, hood, fenders, quarter panels, roof, "
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

    # lang-segment-anything typically returns a URL (or list containing URL)
    seg_mask_url = seg_out[0] if isinstance(seg_out, list) else seg_out
    try:
        mask = download_image(seg_mask_url).convert("L").resize((w, h), Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(500, "Failed to download/parse segmentation mask output.")

    mask = prepare_mask_strict(mask)
    mask_url = img_to_data_url(mask)

    if debug:
        return {
            "before": base_url,
            "mask": mask_url,
            "seg_model": seg_model,
            "inpaint_model": inpaint_model,
            "note": "Debug mode: verify mask covers ONLY paintable body panels.",
        }

    # -----------------------------
    # STEP 2: STRICT INPAINT + CONTROLNET
    # -----------------------------
    # Strictest settings:
    # - controlnet_conditioning_scale near recommended 0.9–0.95
    # - guidance relatively low to avoid “creative” edits
    # - more steps for quality, but not too high
    chosen_seed = seed if seed is not None else stable_seed(angle, color, finish)

    try:
        out = replicate.run(
            inpaint_model,
            {
                "image": base_url,
                "mask": mask_url,
                "prompt": build_prompt(COLOR_MAP[color], finish, angle),
                "negative_prompt": negative_prompt(),
                # Strict control:
                "controlnet_conditioning_scale": 0.95,
                # FLUX defaults are often ~3.5; keep low to preserve original pixels.
                "guidance_scale": 3.5,
                "true_guidance_scale": 3.5,
                "num_inference_steps": 32,
                "num_outputs": 1,
                "seed": chosen_seed,
                "output_format": "png",
                "output_quality": 100,
            },
        )
    except ReplicateError as e:
        raise HTTPException(429, f"Inpainting failed: {str(e)}")

    result_url = out[0] if isinstance(out, list) else out

    return {
        "before": base_url,
        "after": result_url,
        "seed": chosen_seed,
        "models": {"seg": seg_model, "inpaint": inpaint_model},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
