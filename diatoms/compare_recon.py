"""Compare originals and reconstructions in a grid made by sample_diffusion.py --mode recon.

The grid has 8 columns and alternating rows: originals, reconstructions,
originals, ... (256x256 tiles, 2px padding). For every pair this prints SSIM and
PSNR on grayscale and the reconstruction's color strength (0 = pure gray; the
data is grayscale, so color means the model invented it). It also writes an
enlarged image with columns original | reconstruction | difference (x3).

Note: SSIM/PSNR are dominated by the uniform background and padding, so a
diatom that disappears can still score high. Look at the pairs image.

Example:
  python diatoms/compare_recon.py results/diffusion/recon_imagenet.png \
      --out results/diffusion/recon_imagenet_pairs.png --pairs 6 9 13 14 15 3
"""
import argparse

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

TILE, PAD, COLS = 256, 2, 8


def tile(grid, r, c):
    y, x = PAD + r * (TILE + PAD), PAD + c * (TILE + PAD)
    return grid[y:y + TILE, x:x + TILE]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('grid')
    p.add_argument('--out', required=True)
    p.add_argument('--pairs', type=int, nargs='*', help='pair indices to show (default: all)')
    args = p.parse_args()

    grid = np.asarray(Image.open(args.grid).convert('RGB')).astype(float)
    n_rows = (grid.shape[0] - PAD) // (TILE + PAD)
    n = (n_rows // 2) * COLS

    strips = []
    print('pair  grid_pos      SSIM   PSNR(dB)  recon_color')
    for k in range(n):
        r, c = (k // COLS) * 2, k % COLS
        orig, rec = tile(grid, r, c), tile(grid, r + 1, c)
        og, rg = orig.mean(2), rec.mean(2)
        ssim = structural_similarity(og, rg, data_range=255)
        psnr = 10 * np.log10(255 ** 2 / max(((og - rg) ** 2).mean(), 1e-8))
        color = np.abs(rec[..., 0] - rec[..., 2]).mean()
        print(f'{k:4d}  row {r + 1}, col {c + 1}   {ssim:.2f}   {psnr:6.1f}    {color:5.1f}')
        diff = np.repeat(np.clip(np.abs(og - rg) * 3, 0, 255)[..., None], 3, axis=2)
        strips.append(np.concatenate([orig, rec, diff], axis=1))

    show = args.pairs if args.pairs else range(n)
    Image.fromarray(np.concatenate([strips[i] for i in show]).astype(np.uint8)).save(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
