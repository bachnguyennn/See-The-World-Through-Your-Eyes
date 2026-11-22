"""Compare NeRF-only renders vs diffusion-refined renders against ground truth.

Diffusion can make results *look* sharper while inventing content that was never
in the eye reflection. This script reports PSNR / SSIM / LPIPS for both variants
against ground-truth photos of the real scene so the gain (or hallucination) is
measured, not eyeballed.

Images are matched by file stem across the three folders. Ground truth must be
taken from (approximately) the same viewpoint as the render. All images are
resized to the ground-truth resolution before scoring.

Example
-------
    python diffusion/evaluate.py --gt gt/ --nerf renders/ --refined refined/ --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def index(folder: Path) -> dict[str, Path]:
    return {f.stem: f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS}


def load(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True, help="Folder of ground-truth photos.")
    p.add_argument("--nerf", type=Path, required=True, help="Folder of NeRF-only renders.")
    p.add_argument("--refined", type=Path, required=True, help="Folder of diffusion-refined renders.")
    p.add_argument("--csv", type=Path, default=None, help="Optional per-image CSV output.")
    args = p.parse_args()

    import lpips  # imported late so --help works without it

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)

    gt, nerf, refined = index(args.gt), index(args.nerf), index(args.refined)
    stems = sorted(set(gt) & set(nerf) & set(refined))
    if not stems:
        sys.exit("No matching file stems across --gt, --nerf and --refined.")
    missing = (set(nerf) | set(refined)) - set(stems)
    if missing:
        print(f"Skipping {len(missing)} image(s) without a full GT/NeRF/refined triple.", file=sys.stderr)

    def score(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
        psnr = peak_signal_noise_ratio(ref, pred, data_range=1.0)
        ssim = structural_similarity(ref, pred, channel_axis=2, data_range=1.0)
        to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(to_t(pred), to_t(ref)).item()
        return psnr, ssim, lp

    rows = []
    for s in stems:
        ref = load(gt[s])
        size = (ref.shape[1], ref.shape[0])
        rows.append((s, *score(load(nerf[s], size), ref), *score(load(refined[s], size), ref)))

    header = ["image", "nerf_psnr", "nerf_ssim", "nerf_lpips", "refined_psnr", "refined_ssim", "refined_lpips"]
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    m = np.array([r[1:] for r in rows]).mean(axis=0)
    print(f"\n{len(rows)} image(s)          PSNR(+)  SSIM(+)  LPIPS(-)   (+ higher is better, - lower is better)")
    print(f"NeRF only              {m[0]:7.2f}  {m[1]:7.4f}  {m[2]:7.4f}")
    print(f"NeRF + diffusion       {m[3]:7.2f}  {m[4]:7.4f}  {m[5]:7.4f}")
    print(f"delta                  {m[3]-m[0]:+7.2f}  {m[4]-m[1]:+7.4f}  {m[5]-m[2]:+7.4f}")
    print("\nNote: LPIPS often improves while PSNR drops - that pattern can mean plausible-but-invented detail.")


if __name__ == "__main__":
    main()
