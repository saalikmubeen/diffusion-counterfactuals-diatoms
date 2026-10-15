"""Reconstruction metrics for one or more recon grids (sample_diffusion.py --mode recon).

Each grid has 8 columns and alternating rows (originals, reconstructions, ...).
All grids must come from the same test images and seed, so pairs line up.

Per pair and model:
  lpips          perceptual distance (AlexNet LPIPS), lower = better
  psnr / ssim    whole image (dominated by background and padding)
  m_psnr / m_ssim  only inside the diatom mask, higher = better
The diatom mask is computed from the ORIGINAL image (pixels that differ from
the background value, closed and hole-filled), so it's the same for all models.

With two or more grids it also prints means and a paired Wilcoxon signed-rank
test of every model against the first one.

Example:
  python diatoms/recon_metrics.py \
      --grid results/diffusion/recon_imagenet.png --name imagenet \
      --grid results/diffusion/recon_ft_v1_030000.png --name finetuned \
      --masks_out results/diffusion/recon_masks.png
"""
import argparse

import lpips
import numpy as np
import torch
from PIL import Image
from scipy import ndimage, stats
from skimage.metrics import structural_similarity

TILE, PAD, COLS = 256, 2, 8


def tiles(path):
    g = np.asarray(Image.open(path).convert('RGB')).astype(np.float64) / 255
    n_rows = (g.shape[0] - PAD) // (TILE + PAD)
    get = lambda r, c: g[PAD + r * (TILE + PAD):PAD + r * (TILE + PAD) + TILE,
                         PAD + c * (TILE + PAD):PAD + c * (TILE + PAD) + TILE]
    origs, recs = [], []
    for r in range(0, n_rows, 2):
        for c in range(COLS):
            origs.append(get(r, c))
            recs.append(get(r + 1, c))
    return origs, recs


def diatom_mask(orig, thresh=0.04, min_size=150):
    """Pixels of the diatom: differ from the background (border median), closed and filled.
    Thin or faint diatoms break into pieces, so keep every piece >= min_size pixels."""
    gray = orig.mean(2)
    border = np.concatenate([gray[:3].ravel(), gray[-3:].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()])
    fg = np.abs(ndimage.gaussian_filter(gray, 1.0) - np.median(border)) > thresh
    fg = ndimage.binary_closing(fg, iterations=6)
    fg = ndimage.binary_fill_holes(fg)
    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size] or [1 + int(np.argmax(sizes))]
        fg = np.isin(lab, keep)
    return ndimage.binary_dilation(fg, iterations=3)


def size_ratios(orig_mask, rec):
    """Diatom length (vertical extent) and area of the reconstruction relative to the
    original, from their masks. 1.0 = same size; >1 = the reconstruction is bigger."""
    rm = diatom_mask(rec)
    rows_o, rows_r = np.where(orig_mask.any(1))[0], np.where(rm.any(1))[0]
    if len(rows_o) == 0 or len(rows_r) == 0:
        return float('nan'), float('nan')
    length = (rows_r[-1] - rows_r[0] + 1) / (rows_o[-1] - rows_o[0] + 1)
    return float(length), float(rm.sum() / max(orig_mask.sum(), 1))


def metrics(orig, rec, mask, lp):
    og, rg = orig.mean(2), rec.mean(2)
    ssim, ssim_map = structural_similarity(og, rg, data_range=1.0, full=True)
    mse = ((og - rg) ** 2).mean()
    m_mse = ((og - rg)[mask] ** 2).mean()
    to_t = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).float()[None] * 2 - 1
    with torch.no_grad():
        d = lp(to_t(orig), to_t(rec)).item()
    length, area = size_ratios(mask, rec)
    return dict(lpips=d, psnr=10 * np.log10(1 / max(mse, 1e-10)), ssim=ssim,
                m_psnr=10 * np.log10(1 / max(m_mse, 1e-10)), m_ssim=ssim_map[mask].mean(),
                len_ratio=length, area_ratio=area)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', action='append', required=True)
    p.add_argument('--name', action='append')
    p.add_argument('--masks_out', help='save originals with their diatom masks, to check them')
    args = p.parse_args()
    names = args.name or [f'model{i}' for i in range(len(args.grid))]

    lp = lpips.LPIPS(net='alex', verbose=False).eval()
    data = [tiles(g) for g in args.grid]
    origs = data[0][0]
    masks = [diatom_mask(o) for o in origs]

    keys = ['lpips', 'psnr', 'ssim', 'm_psnr', 'm_ssim']
    better = {'lpips': 'lower', 'psnr': 'higher', 'ssim': 'higher', 'm_psnr': 'higher', 'm_ssim': 'higher'}
    res = {n: [metrics(o, r, m, lp) for o, r, m in zip(origs, recs, masks)] for n, (_, recs) in zip(names, data)}

    print(f'{len(origs)} pairs; diatom mask covers {100 * np.mean([m.mean() for m in masks]):.0f}% of the image on average\n')
    print('pair  ' + '  '.join(f'{k:>{9 * len(names)}s}' for k in keys))
    print('      ' + '  '.join(' '.join(f'{n[:8]:>8s}' for n in names) for _ in keys))
    for i in range(len(origs)):
        print(f'{i:4d}  ' + '  '.join(' '.join(f'{res[n][i][k]:8.3f}' for n in names) for k in keys))

    print('\nmean ± std' + ' ' * 6 + '  '.join(f'{n:>18s}' for n in names) + '    better / Wilcoxon p (vs ' + names[0] + ')')
    for k in keys:
        cols = [np.array([r[k] for r in res[n]]) for n in names]
        line = f'{k:8s} ({better[k]:6s})' + '  '.join(f'{c.mean():9.3f} ± {c.std():6.3f}' for c in cols)
        for n, c in zip(names[1:], cols[1:]):
            wins = (c < cols[0]) if better[k] == 'lower' else (c > cols[0])
            pval = stats.wilcoxon(c, cols[0]).pvalue if np.any(c != cols[0]) else 1.0
            line += f'    {n}: {int(wins.sum())}/{len(c)} better, p={pval:.1e}'
        print(line)

    print('\nsize of the diatom in the reconstruction vs the original (1.00 = unchanged)')
    for n in names:
        L = np.array([r['len_ratio'] for r in res[n]]); A = np.array([r['area_ratio'] for r in res[n]])
        print(f'  {n:12s} length ratio mean {np.nanmean(L):.2f} (min {np.nanmin(L):.2f}, max {np.nanmax(L):.2f}) | '
              f'area ratio mean {np.nanmean(A):.2f} | pairs with length off by >10%: {int(np.sum(np.abs(L - 1) > 0.1))}/{len(L)}')

    if args.masks_out:
        rows = []
        for o, m in zip(origs, masks):
            overlay = o.copy()
            edge = m ^ ndimage.binary_erosion(m, iterations=2)
            overlay[edge] = [1, 0, 0]
            rows.append(overlay)
        grid = np.concatenate([np.concatenate(rows[i:i + 8], 1) for i in range(0, len(rows), 8)], 0)
        Image.fromarray((grid * 255).astype(np.uint8)).save(args.masks_out)
        print('\nmask check image:', args.masks_out)


if __name__ == '__main__':
    main()
