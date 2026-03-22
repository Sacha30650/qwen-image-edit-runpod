"""
RunPod Serverless Handler for Qwen-Image-Edit
Receives a prompt + reference image (base64), returns edited image URL.
"""

import runpod
import torch
import base64
import io
import os
from PIL import Image

# Global model reference — loaded once at cold start
pipe = None


def load_model():
    """Load Qwen Image Edit pipeline once."""
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import FluxPipeline

    print("Loading Qwen-Image-Edit model...")
    pipe = FluxPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    print("Model loaded successfully.")
    return pipe


def decode_base64_image(data_uri: str) -> Image.Image:
    """Decode a base64 data URI to PIL Image."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def encode_image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def handler(event):
    """
    RunPod handler function.

    Input (event["input"]):
        - prompt: str — editing instruction
        - image: str — base64 data URI of reference image
        - seed: int (optional) — random seed
        - guidance_scale: float (optional, default 2.5)
        - num_inference_steps: int (optional, default 28)

    Output:
        - image: str — base64 data URI of generated image
    """
    try:
        inp = event["input"]
        prompt = inp.get("prompt", "")
        image_b64 = inp.get("image", "")
        seed = inp.get("seed", -1)
        guidance_scale = inp.get("guidance_scale", 2.5)
        num_inference_steps = inp.get("num_inference_steps", 28)

        if not prompt:
            return {"error": "Missing prompt"}

        model = load_model()

        # Decode reference image if provided
        input_image = None
        if image_b64:
            input_image = decode_base64_image(image_b64)
            input_image = input_image.resize((1024, 1024), Image.LANCZOS)

        # Set seed
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate
        if input_image is not None:
            result = model(
                prompt=prompt,
                image=input_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        else:
            result = model(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        output_image = result.images[0]
        output_b64 = encode_image_to_base64(output_image)

        return {"image": output_b64}

    except Exception as e:
        print(f"Error in handler: {e}")
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
