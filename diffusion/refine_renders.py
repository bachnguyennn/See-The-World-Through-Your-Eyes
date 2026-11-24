"""Diffusion post-refinement of EyeNeRF renders (Stage 1: post-hoc, no retraining).

Takes images rendered from a trained EyeNeRF model (e.g. with `ns-render`) and
restores them with a diffusion model. Runs as a separate process from NeRF
training so the two never compete for GPU memory.

Methods
-------
upscale : Stable Diffusion x4 upscaler. Best for the small, low-res renders
          EyeNeRF produces (~64-256 px). Output is 4x the input size.
tile    : SD 1.5 + ControlNet Tile, img2img at low strength. Keeps the layout
          of the render and only re-synthesises texture/detail. `--strength`
          controls how much the model may change (lower = more faithful).

Consistency
-----------
Every frame uses the same seed, so a sequence (e.g. a camera-path video) gets
identical starting noise per frame, which noticeably reduces flicker.
Diffusion can still invent details that were not in the eye reflection: always
compare against NeRF-only renders with `evaluate.py` before drawing conclusions.

Example
-------
    python diffusion/refine_renders.py --input renders/ --output refined/ --method upscale
    python diffusion/refine_renders.py --input renders/ --output refined/ --method tile --strength 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

UPSCALER_ID = "stabilityai/stable-diffusion-x4-upscaler"
SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TILE_CONTROLNET_ID = "lllyasviel/control_v11f1e_sd15_tile"

DEFAULT_PROMPT = "a sharp, detailed photo of an indoor scene, natural lighting"
DEFAULT_NEGATIVE = "blurry, noisy, low quality, artifacts, oversaturated, cartoon, painting"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Image file or folder of NeRF renders.")
    p.add_argument("--output", type=Path, required=True, help="Folder to write refined images to.")
    p.add_argument("--method", choices=["upscale", "tile"], default="upscale")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Describe the captured scene (e.g. 'a desk with a laptop and a lamp'); a good prompt helps a lot.")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--steps", type=int, default=30, help="Denoising steps.")
    p.add_argument("--guidance", type=float, default=None,
                   help="CFG scale (default: 7.5 for upscale, 5.0 for tile).")
    p.add_argument("--seed", type=int, default=0, help="Same seed for every frame (reduces flicker).")
    # upscale-specific
    p.add_argument("--noise-level", type=int, default=20,
                   help="[upscale] Noise added to the low-res input; higher = more hallucinated detail.")
    p.add_argument("--max-input-side", type=int, default=256,
                   help="[upscale] Downscale inputs larger than this before 4x upscaling (VRAM guard).")
    # tile-specific
    p.add_argument("--strength", type=float, default=0.3,
                   help="[tile] img2img strength, 0-1. 0.2-0.4 keeps the render faithful.")
    p.add_argument("--target-side", type=int, default=768,
                   help="[tile] Long side the render is resized to before refinement (multiple of 8).")
    p.add_argument("--controlnet-scale", type=float, default=1.0, help="[tile] ControlNet conditioning scale.")
    p.add_argument("--pre-smooth", type=float, default=None,
                   help="[tile] Gaussian blur radius applied after resizing, before refinement. Stops the model "
                        "turning render noise into fake texture. Default: 0.5 * upscale factor; 0 disables.")
    # misc
    p.add_argument("--stride", type=int, default=1,
                   help="Only refine every Nth image (sorted by name) - for quick looks at long camera paths.")
    p.add_argument("--save-compare", action="store_true",
                   help="Also save side-by-side input|refined images to <output>/compare/.")
    p.add_argument("--cpu-offload", action="store_true",
                   help="Offload model parts to CPU between steps (use if you hit CUDA out-of-memory).")
    return p.parse_args()


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {path}")
    return files


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    # 16-bit TIFFs (as recommended by the EyeNeRF capture guide) open as I;16 / RGB;16.
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        img = img.point(lambda v: v * (1 / 256)).convert("L")
    return img.convert("RGB")


def round_to(x: int, m: int = 8) -> int:
    return max(m, int(round(x / m)) * m)


def build_pipeline(args: argparse.Namespace, device: str, dtype: torch.dtype):
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionUpscalePipeline,
    )

    if args.method == "upscale":
        pipe = StableDiffusionUpscalePipeline.from_pretrained(UPSCALER_ID, torch_dtype=dtype)
        pipe.vae.enable_tiling()  # the x4 upscaler's VAE decode is the VRAM peak
    else:
        controlnet = ControlNetModel.from_pretrained(TILE_CONTROLNET_ID, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SD15_ID, controlnet=controlnet, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )

    if device == "cuda" and args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def refine_upscale(pipe, img: Image.Image, args, generator) -> Image.Image:
    w, h = img.size
    if max(w, h) > args.max_input_side:
        s = args.max_input_side / max(w, h)
        img = img.resize((round_to(int(w * s)), round_to(int(h * s))), Image.LANCZOS)
    else:
        img = img.resize((round_to(w), round_to(h)), Image.LANCZOS)
    return pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=img,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance if args.guidance is not None else 7.5,
        noise_level=args.noise_level,
        generator=generator,
    ).images[0]


def refine_tile(pipe, img: Image.Image, args, generator) -> Image.Image:
    w, h = img.size
    s = args.target_side / max(w, h)
    img = img.resize((round_to(int(w * s)), round_to(int(h * s))), Image.BICUBIC)
    radius = args.pre_smooth if args.pre_smooth is not None else 0.5 * max(s, 1.0)
    if radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=img,
        control_image=img,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance if args.guidance is not None else 5.0,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator,
    ).images[0]


def side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    a = a.resize(b.size, Image.NEAREST)  # nearest keeps the original render's pixels honest
    out = Image.new("RGB", (a.width + b.width, b.height))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    return out


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (very slow).", file=sys.stderr)

    files = list_images(args.input)[:: max(1, args.stride)]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_compare:
        (args.output / "compare").mkdir(exist_ok=True)

    print(f"Loading '{args.method}' pipeline on {device} ...")
    pipe = build_pipeline(args, device, dtype)
    refine = refine_upscale if args.method == "upscale" else refine_tile

    for i, f in enumerate(files, 1):
        img = load_rgb(f)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        out = refine(pipe, img, args, generator)
        out_path = args.output / f"{f.stem}.png"
        out.save(out_path)
        if args.save_compare:
            side_by_side(img, out).save(args.output / "compare" / f"{f.stem}.png")
        print(f"[{i}/{len(files)}] {f.name} {img.size} -> {out_path.name} {out.size}")


if __name__ == "__main__":
    main()
