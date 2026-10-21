"""Break DVCE results down by source-specimen properties (cropped or not, length, atlas).

Joins the per-counterfactual metrics.csv (written by dvce_postprocess.py, rows in the
order of the --pth files) with each source image's metadata from splits.csv; rows are
matched to source images by pixel content, not by index.

Example:
  python diatoms/failure_breakdown.py --pth <wid_0 pth> <wid_1 pth> \
      --metrics results/dvce/full_r50eps1_ftv2/metrics.csv \
      --pairs evaluation/vce_pairs_paper_all.json --out results/dvce/full_r50eps1_ftv2/extra/breakdown.txt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import DATA_NATIVE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', nargs='+', required=True)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    entries = json.load(open(args.pairs))['entries']
    imgs = {e['path']: np.asarray(Image.open(os.path.join(DATA_NATIVE, e['path'])).convert('L'), np.float32) / 255
            for e in entries}
    with open(os.path.join(DATA_NATIVE, 'splits.csv')) as f:
        meta = {r['path']: r for r in csv.DictReader(f)}
    rows, worst = [], 0.0
    for p in args.pth:
        d = torch.load(p, map_location='cpu')
        for k in range(len(d['gt_imgs'])):
            x = d['gt_imgs'][k].float().mean(0).numpy()
            path = min(imgs, key=lambda q: np.abs(imgs[q] - x).mean())
            worst = max(worst, float(np.abs(imgs[path] - x).mean()))
            r = meta[path]
            rows.append(dict(path=path, species=r['species_code'], cropped=int(r['cropped']), source=r['source'],
                             length=max(int(r['orig_w']), int(r['orig_h']))))
    m = pd.read_csv(args.metrics)
    assert len(m) == len(rows), 'metrics rows and .pth images differ'
    d = pd.concat([m.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    L = [f'{len(d)} counterfactuals, {d.path.nunique()} source specimens, {d.species.nunique()} source species, '
         f'{d.source.nunique()} atlases (max pixel mismatch when matching rows: {worst:.1e})', '']
    L.append('BY CROPPING (source longer than 224 px)')
    for c, g in d.groupby('cropped'):
        L.append(f'  cropped={c}: n={len(g)}  reached {g.success.mean():.1%}  LPIPS {g.lpips.mean():.3f}  '
                 f'transfer robust R18 {g.independent_robust_success.mean():.1%}')
    tab = pd.crosstab(d.cropped, d.success).values
    L.append(f'  Fisher exact (reached vs cropped): p = {fisher_exact(tab)[1]:.3f}; '
             f'LPIPS Mann-Whitney p = {mannwhitneyu(d.lpips[d.cropped == 0], d.lpips[d.cropped == 1]).pvalue:.1e}')
    fails = d[d.success == 0]
    L.append(f'  failures: {len(fails)}, of which cropped {int(fails.cropped.sum())} ({fails.cropped.mean():.0%})')
    L.append('')
    L.append('BY SOURCE LENGTH (px)')
    d['bin'] = pd.cut(d.length, [0, 100, 150, 224, 400, 10 ** 5], labels=['<=100', '101-150', '151-224', '225-400', '>400'])
    for b, g in d.groupby('bin'):
        L.append(f'  {b:>8s}: n={len(g):3d}  reached {g.success.mean():.1%}  LPIPS {g.lpips.mean():.3f}  '
                 f'transfer robust R18 {g.independent_robust_success.mean():.1%}')
    L.append('')
    L.append('BY ATLAS (source id)')
    for s, g in sorted(d.groupby('source'), key=lambda kv: int(kv[0])):
        L.append(f'  atlas {s:>2s}: n={len(g):3d}  reached {g.success.mean():.1%}  cropped {g.cropped.mean():.0%}')
    txt = '\n'.join(L)
    print(txt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(txt + '\n')
    d.drop(columns=['bin']).to_csv(args.out.replace('.txt', '.csv'), index=False)


if __name__ == '__main__':
    main()
