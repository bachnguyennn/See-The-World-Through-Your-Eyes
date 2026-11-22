# See The World Through Your Eyes

Reconstructing the scene a person is looking at from the reflections in their eyes,
built on [EyeNeRF](https://github.com/hadizayer/eyenerf) ("Seeing the World through
Your Eyes", CVPR 2024), with diffusion-based refinement of the reconstructions.

## Layout

| Path | What |
|---|---|
| `eyenerf/` | Git submodule → our fork [`bachnguyennn/eyenerf`](https://github.com/bachnguyennn/eyenerf) (upstream remote: `hadizayer/eyenerf`). NeRF training. |
| `diffusion/refine_renders.py` | Stage 1: diffusion post-refinement of rendered views. |
| `diffusion/evaluate.py` | PSNR / SSIM / LPIPS of NeRF-only vs refined against ground truth. |

## Setup

```bash
git clone --recurse-submodules https://github.com/bachnguyennn/See-The-World-Through-Your-Eyes.git
# already cloned without submodules:
git submodule update --init
```

Two environments, on purpose: NeRF training (tiny-cuda-nn) and Stable Diffusion
each want most of the GPU, and their package pins differ.

1. **EyeNeRF** — follow `eyenerf/README.md` (conda env `eyenerf`, CUDA 11.8, tiny-cuda-nn,
   GroundingDINO, SAM, ExifTool).
2. **Diffusion**:
   ```bash
   python -m venv .venv-diffusion
   # torch + torchvision must both come from the CUDA index (PyPI Windows wheels are CPU-only)
   .venv-diffusion/Scripts/pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
   .venv-diffusion/Scripts/pip install -r diffusion/requirements.txt
   .venv-diffusion/Scripts/python -c "import torch; print(torch.cuda.is_available())"   # must print True
   ```
   Model weights (~3–6 GB) download from Hugging Face on first run.

## Pipeline

```bash
# 1. Train EyeNeRF (eyenerf env)
ns-train cornea --data $DATA --output_dir outputs --experiment_name $EXP \
    --pipeline.model.rot-loss-mult 0.1 --pipeline.model.use-texture-field True

# 2. Render views of the reconstructed scene. Build a camera path in the viewer
#    (http://localhost:7007), export it, then:
ns-render camera-path --load-config outputs/$EXP/cornea/<timestamp>/config.yml \
    --camera-path-filename camera_path.json --output-format images --output-path renders/

# 3. Diffusion refinement (diffusion venv) — run only after training has exited
python diffusion/refine_renders.py --input renders/ --output refined/upscale --method upscale \
    --prompt "a desk with a monitor and a lamp" --save-compare
python diffusion/refine_renders.py --input renders/ --output refined/tile --method tile \
    --strength 0.3 --prompt "a desk with a monitor and a lamp" --save-compare

# 4. Measure it (ground-truth photos named like the renders, from matching viewpoints)
python diffusion/evaluate.py --gt gt/ --nerf renders/ --refined refined/upscale --csv upscale.csv
```

### Choosing a method

- **`upscale`** (SD x4 upscaler) — first choice for EyeNeRF's small renders. `--noise-level`
  (default 20) trades faithfulness for invented detail.
- **`tile`** (SD 1.5 + ControlNet Tile, img2img) — keeps the render's layout and re-synthesises
  texture. Keep `--strength` at 0.2–0.4; above ~0.5 it starts painting a new scene. The input
  is blurred slightly first (`--pre-smooth`) because the model otherwise turns render noise into
  fake texture (dots, letter-like patterns). The ControlNet Tile weights are only published as a
  pickle `.bin` (official `lllyasviel` repo), so diffusers prints an "unsafe serialization" notice.

Smoke test (128×96 noisy/blurred synthetic "renders" vs 512×384 ground truth, RTX 3060,
~1 min per method including model load):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| input (NeRF stand-in) | 23.21 | 0.465 | 0.729 |
| `upscale` | 20.02 | 0.357 | **0.497** |
| `tile`, strength 0.35 | **22.09** | 0.491 | 0.650 |
| `tile`, strength 0.65 | 21.54 | **0.492** | 0.544 |

`upscale` looks by far the sharpest but invents the most (lowest PSNR); `tile` at low strength
is the most faithful. Real EyeNeRF renders will behave differently — rerun `evaluate.py` on them.
- **The prompt matters.** Describe what is actually in the scene; generic prompts give generic
  detail.
- Every frame uses the same `--seed`, which reduces flicker across a camera path, but frames
  are still refined independently. Multi-view-consistent refinement is the planned Stage 2
  (diffusion inside NeRF training, Instruct-NeRF2NeRF-style iterative dataset updates).
- On a 12 GB GPU both methods fit in fp16; add `--cpu-offload` if you hit out-of-memory.

### Honesty check

Diffusion can produce sharp, plausible detail that was never in the eye reflection. Always
report NeRF-only alongside NeRF + diffusion, with `evaluate.py` numbers. LPIPS improving while
PSNR drops is a warning sign of hallucinated content.
