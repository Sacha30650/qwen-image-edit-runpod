"""
Qwen Image Edit — RunPod Serverless Handler (v2)
Direct diffusers inference with bitsandbytes 8-bit quantization.
No ComfyUI. ~17GB VRAM → runs on RTX 4090 (24GB) or RTX 5090 (32GB).
"""

import runpod
import torch
import base64
import io
import os
import time
import requests
from PIL import Image

# Global model reference — loaded once, reused across requests
PIPELINE = None
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/huggingface")


def load_model():
    """Load Qwen Image Edit with 8-bit quantization for low VRAM usage."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print(f"[init] Loading {MODEL_ID} with 8-bit quantization...")
    print(f"[init] Cache dir: {CACHE_DIR}")
    print(f"[init] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[init] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    start = time.time()

    from transformers import BitsAndBytesConfig
    from diffusers import FluxPipeline

    # 8-bit quantization → ~17GB VRAM instead of ~40GB+
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    PIPELINE = FluxPipeline.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )

    elapsed = time.time() - start
    print(f"[init] Model loaded in {elapsed:.1f}s")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"[init] VRAM used: {allocated:.1f} GB")
    return PIPELINE


def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def decode_base64_image(data: str) -> Image.Image:
    """Decode base64 (with or without data URI prefix) to PIL Image."""
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def handler(event):
    """
    RunPod handler.

    Input:
        prompt: str — editing instruction (required)
        image: str — base64 data URI of reference image (optional)
        image_url: str — URL of reference image (optional, alternative to image)
        seed: int (default 0)
        num_inference_steps: int (default 28)
        true_cfg_scale: float (default 4.0)

    Output:
        image: str — base64 data URI of generated image
        generation_time: float — seconds
    """
    try:
        inp = event.get("input", {})

        # Quick ping for health check / testing
        if inp.get("ping"):
            return {"status": "ok", "message": "pong"}

        prompt = inp.get("prompt", "")
        image_b64 = inp.get("image", "")
        image_url = inp.get("image_url", "")
        seed = inp.get("seed", 0)
        num_inference_steps = inp.get("num_inference_steps", 28)
        true_cfg_scale = inp.get("true_cfg_scale", 4.0)

        if not prompt:
            return {"error": "Missing prompt"}

        # Load model (cached after first call)
        pipe = load_model()

        # Get reference image
        input_image = None
        if image_b64:
            input_image = decode_base64_image(image_b64)
        elif image_url:
            input_image = download_image(image_url)

        if input_image:
            input_image = input_image.resize((1024, 1024), Image.LANCZOS)

        # Generate
        start = time.time()

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": " ",
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": 1.0,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
            "generator": torch.manual_seed(seed),
        }

        if input_image is not None:
            gen_kwargs["image"] = [input_image]

        with torch.inference_mode():
            result = pipe(**gen_kwargs)

        output_image = result.images[0]
        elapsed = time.time() - start
        print(f"[generate] Done in {elapsed:.1f}s (steps={num_inference_steps})")

        return {
            "image": encode_image(output_image),
            "generation_time": round(elapsed, 2),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Pre-load model at worker startup
print("[startup] Pre-loading model...")
try:
    load_model()
    print("[startup] Model ready!")
except Exception as e:
    print(f"[startup] Pre-load failed: {e}")
    print("[startup] Will retry on first request")

runpod.serverless.start({"handler": handler})
