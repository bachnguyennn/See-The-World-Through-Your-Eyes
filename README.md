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
| `eyenerf/eyenerf/diffusion_guidance.py` | Stage 2: diffusion inside training (`ns-train cornea-diffusion`). |

## Setup

```bash
git clone --recurse-submodules https://github.com/bachnguyennn/See-The-World-Through-Your-Eyes.git
# already cloned without submodules:
git submodule update --init
```

Two environments, on purpose: NeRF training (tiny-cuda-nn) and Stable Diffusion
each want most of the GPU, and their package pins differ.

1. **EyeNeRF** (runs in WSL2 — native Windows builds of tiny-cuda-nn/GroundingDINO are fragile):
   ```powershell
   # once, in an admin PowerShell, then reboot
   wsl --install -d Ubuntu-22.04
   ```
   ```bash
   # inside Ubuntu, from the repo root (e.g. /mnt/h/Bach/Bach\ Projects/CV1/See-The-World-Through-Your-Eyes)
   bash scripts/setup_wsl_eyenerf.sh      # CUDA 12.1 toolkit, conda env "eyenerf", tiny-cuda-nn,
                                          # nerfstudio 1.0.2, EyeNeRF, SAM, GroundingDINO, ExifTool
   conda activate eyenerf
   bash scripts/train_sample.sh           # downloads the authors' sample capture and trains
   ```
   Use Ubuntu **22.04** (CUDA 12.1 needs gcc ≤ 12). Keep data/outputs under the WSL home
   (`~/eyenerf-data`, the default) — `/mnt/h/...` works but is much slower.
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

- **`upscale`** (SD x4 upscaler) — **not recommended on real EyeNeRF renders** (see below).
  `--noise-level` (default 20) trades faithfulness for invented detail.
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
is the most faithful.

**On real EyeNeRF renders** (sample `capture_3`: two plush toys on a table, authors' camera
path `2024-06-17_182843`, every 8th frame, prompt describing the scene):

- `tile`, strength 0.3 — **use this.** Keeps the layout and both objects in place, sharpens the
  white plush; but it gives the table a fur texture (borrowed from the "plush" prompt) and
  leaves the orange toy blurry.
- `upscale` — unusable as is. The renders are mostly black with a small lit region, so after
  downscaling to 256 px the content is tiny and the model invents a different picture (frame 0
  became a close-up of a fur object). Frames also disagree with each other → flicker.
- The sample captures ship no ground-truth scene photo, so these are visual comparisons only.
  Per-frame inconsistency is the main argument for Stage 2 (diffusion inside NeRF training).
- **The prompt matters.** Describe what is actually in the scene; generic prompts give generic
  detail.
- Every frame uses the same `--seed`, which reduces flicker across a camera path, but frames
  are still refined independently. Stage 2 (below) puts diffusion inside training instead.
- On a 12 GB GPU both methods fit in fp16; add `--cpu-offload` if you hit out-of-memory.

## Stage 2: diffusion inside training (`cornea-diffusion`)

EyeNeRF's training images are photos of *eyes*, so the Instruct-NeRF2NeRF trick (replace
training images with diffusion-edited versions) does not apply. Instead
[`eyenerf/eyenerf/diffusion_guidance.py`](eyenerf/eyenerf/diffusion_guidance.py) supervises the
scene through **virtual views**:

1. From `--pipeline.guidance-start-step` (default 10000), every `--pipeline.guidance-every`
   steps (default 5000), render 8 virtual 512 px views of the scene. Cameras are derived
   automatically from the reflected rays (placed at the corneas, looking along the mean
   reflected direction) or taken from `--pipeline.guidance-camera-path`.
2. Refine them with `diffusion/refine_renders.py --method tile` in the diffusion environment
   (a subprocess: from WSL it runs the Windows `.venv-diffusion` through interop, so no
   second copy of torch or the model weights is needed).
3. Every step, add an L1 + LPIPS loss on 32×32 patches of those refined views. The loss is
   **rescaled every step to `--pipeline.guidance-weight` × the eye-image loss** (default 0.1).

```bash
# inside WSL, conda activate eyenerf
METHOD=cornea-diffusion PROMPT="a photo of toys on a table in a dark room" bash scripts/train_sample.sh
# or directly (from eyenerf/):
ns-train cornea-diffusion --data $DATA --pipeline.model.rot-loss-mult 0.1 \
    --pipeline.model.use-texture-field True --pipeline.guidance-weight 0.5 \
    --pipeline.guidance-prompt "..."
```

Every round is saved to `<run>/diffusion_guidance/step_XXXXXX/{render,refined}` for inspection.
Each round takes ~40 s on an RTX 3060 (12 GB), and training + diffusion fit side by side.

### Results on `capture_3`

Fine-tuning the 30k-step baseline for 2500 more steps, with guidance rounds every 500 steps,
then rendering the authors' camera path (96 frames). There is no ground truth, so these are
measured against the unguided control run. For scale: plain continued training moves the
renders by ~49.8 dB (baseline vs control).

| Run | Eye-image PSNR | PSNR vs control | Sharpness* |
|---|---|---|---|
| control (no guidance) | 51.08 | – | 2.26 |
| guidance-weight 0.1 (adaptive) | 52.05 | 49.16 | 2.16 |
| guidance-weight 0.5 (adaptive) | 52.25 | 47.64 | 2.25 |
| fixed weights, L1+LPIPS ×0.05 (first attempt) | **36.63** | **31.06** | 2.98 |

\* variance of the Laplacian in the lit region, ×10⁴.

- **Fixed loss weights overwhelm the eye data.** Eye reflections are faint, so the eye-image
  loss is ~1e-5; a guidance loss weighted 0.05 was ~100× larger. The NeRF then copied the
  diffusion output: higher "sharpness", but it is invented fur-like texture on the table, a blue
  cast on the white plush, and background speckle, while the fit to the real eye images
  dropped 15 dB. That is why the weight is now adaptive.
- **With adaptive weights, guidance is safe but does not visibly help.** Renders change
  about as much as continued training alone does, and sharpness is unchanged. The eye fit is
  unharmed (slightly higher, within noise).
- **Prompt words become texture.** A prompt mentioning "plush toys" made the refiner paint
  fur everywhere; keep it to the layout ("toys on a table in a dark room").
- Takeaway so far: the limit is how much information the eye reflection holds, not the
  regulariser. Diffusion can only add detail that is not in the data, and in these tests
  that detail was either too weak to matter or wrong.

### Honesty check

Diffusion can produce sharp, plausible detail that was never in the eye reflection. Always
report NeRF-only alongside NeRF + diffusion, with `evaluate.py` numbers. LPIPS improving while
PSNR drops is a warning sign of hallucinated content.
