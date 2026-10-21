"""Do two DVCE runs give the same explanations for the same (source, target) pairs?

Compares a reference run with one or more other runs on identical pairs:
  lpips_between   LPIPS between the two counterfactuals (0 = identical)
  change_corr     Pearson correlation of the two change maps |x' - x| inside the
                  specimen mask (1 = changed the same places by the same amounts)
  trait_agree     target-specific trait agreement: for each source, the difference
                  between its two counterfactuals (target B1 minus target B2) in each
                  measured trait (length, width, elongation, stria period); share of
                  differences with the same sign in both runs (only differences
                  > --min_change relative in both). Using B1 - B2 cancels changes that
                  diffusion makes regardless of the target.
Include a run of the same classifier with another seed as the noise floor: a
different classifier gives "the same explanations" if it is as close to the
reference as the reseeded run is.

Example:
  python diatoms/compare_explanations.py --ref main:<pths> --run seed2:<pths> --run page_clf:<pths> \
      --out results/dvce/explanation_agreement.txt
"""
import argparse
import os
import sys

import lpips
import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_metrics import diatom_mask  # noqa: E402
from trait_check import TRAITS, load_run, measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='name:pth,pth')
    ap.add_argument('--run', action='append', required=True, help='name:pth,pth')
    ap.add_argument('--min_change', type=float, default=0.03, help='relative trait change counted as a change')
    ap.add_argument('--out')
    args = ap.parse_args()

    ref_name, ref_p = args.ref.split(':', 1)
    gt, cf_ref, y, t = load_run(ref_p)
    n = len(t)
    lp = lpips.LPIPS(net='alex', verbose=False).eval()
    to_t = lambda a: torch.from_numpy(a)[:, None].repeat(1, 3, 1, 1) * 2 - 1
    masks = [diatom_mask(np.repeat(gt[i][..., None], 3, 2)) for i in range(n)]
    crop = [np.isfinite(measure(gt[i])['length']) and measure(gt[i])['length'] >= 222 for i in range(n)]
    tr_o = [measure(gt[i], crop[i]) for i in range(n)]
    tr_ref = [measure(cf_ref[i], crop[i]) for i in range(n)]

    lines = [f'{n} counterfactuals; reference run: {ref_name}', '',
             f'{"run":14s} {"LPIPS between":>14s} {"change-map corr":>16s} {"target-specific trait agree":>28s}']
    per_run = {}
    for spec in args.run:
        name, p = spec.split(':', 1)
        gt2, cf, y2, t2 = load_run(p)
        assert (y2 == y).all() and (t2 == t).all() and np.abs(gt2 - gt).max() < 1e-3, f'{name}: different pairs'
        with torch.no_grad():
            d = torch.cat([lp(to_t(cf_ref[i:i + 20]), to_t(cf[i:i + 20])).flatten() for i in range(0, n, 20)]).numpy()
        corr = []
        for i in range(n):
            a = np.abs(cf_ref[i] - gt[i])[masks[i]]
            b = np.abs(cf[i] - gt[i])[masks[i]]
            corr.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
        corr = np.array(corr)
        agree = tot = 0
        tr_c = [measure(cf[i], crop[i]) for i in range(n)]
        for i in range(0, n - 1, 2):                 # the two targets of one source
            for k in TRAITS:
                o = tr_o[i][k]
                d1 = tr_ref[i][k] - tr_ref[i + 1][k]
                d2 = tr_c[i][k] - tr_c[i + 1][k]
                if not np.isfinite([o, d1, d2]).all() or o == 0:
                    continue
                if abs(d1) / o < args.min_change or abs(d2) / o < args.min_change:
                    continue
                tot += 1
                agree += int(np.sign(d1) == np.sign(d2))
        per_run[name] = (d, corr)
        lines.append(f'{name:14s} {d.mean():14.3f} {np.nanmean(corr):16.2f} {agree / max(tot, 1):20.0%} ({agree}/{tot})')
    names = list(per_run)
    if len(names) >= 2:
        lines.append('')
        a, b = names[0], names[1:]
        for nb in b:
            p_l = wilcoxon(per_run[a][0], per_run[nb][0]).pvalue
            ok = np.isfinite(per_run[a][1]) & np.isfinite(per_run[nb][1])
            p_c = wilcoxon(per_run[a][1][ok], per_run[nb][1][ok]).pvalue
            lines.append(f'{nb} vs {a} (paired Wilcoxon): LPIPS between p = {p_l:.1e}, change-map corr p = {p_c:.1e}')
    txt = '\n'.join(lines)
    print(txt)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(txt + '\n')


if __name__ == '__main__':
    main()
