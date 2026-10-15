"""After a DVCE run: one figure per source diatom (diatoms/dvce_figure.py layout)
and one metrics table + summary (diatoms/eval_dvce.py) for all .pth files of the run.

Example:
  python diatoms/dvce_postprocess.py --pth run_a/50.pth run_b/50.pth \
      --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt --out_dir results/dvce/main_r50eps1_ftv2
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pth', nargs='+', required=True)
    p.add_argument('--ckpt', required=True, help='classifier used for the DVCEs')
    p.add_argument('--transfer', nargs='*', default=['DiatlasModels/resnet18_native_eps0.5/best.pt', 'independent'],
                   help='independent classifiers for the transfer analysis (not shown on the figures)')
    p.add_argument('--transfer_name', nargs='*', default=['independent_robust', 'independent_nonrobust'])
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)

    tables = []
    fig_no = 0
    for part, pth in enumerate(args.pth):
        n_sources = len(torch.load(pth, map_location='cpu')['gt_imgs']) // 2
        for s in range(n_sources):
            out = os.path.join(args.out_dir, 'figures', f'{fig_no:03d}.png')
            subprocess.run([sys.executable, os.path.join(HERE, 'dvce_figure.py'), pth, '--ckpt', args.ckpt,
                            '--source', str(s), '--out', out],
                           check=True, stderr=subprocess.DEVNULL)
            fig_no += 1
        csv_part = os.path.join(args.out_dir, f'metrics_part{part}.csv')
        subprocess.run([sys.executable, os.path.join(HERE, 'eval_dvce.py'), pth, '--eval_ckpt', *args.transfer,
                        '--eval_name', *args.transfer_name, '--out', csv_part],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(csv_part) as f:
            tables += list(csv.DictReader(f))
        os.remove(csv_part)

    with open(os.path.join(args.out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tables[0]))
        w.writeheader()
        w.writerows(tables)

    col = lambda k: np.array([float(r[k]) for r in tables])
    lines = [
        f'{len(tables)} counterfactuals from {fig_no} source diatoms',
        '',
        'EXPLAINED CLASSIFIER',
        f'  target reached:   {col("success").mean():.1%}   p(target) {col("p_target_before").mean():.3f} -> '
        f'{col("p_target_after").mean():.3f} (mean), median after {np.median(col("p_target_after")):.3f}',
        f'  closeness:        LPIPS {col("lpips").mean():.3f} (median {np.median(col("lpips")):.3f}), L1 {col("l1").mean():.4f}',
        f'  where it changed: inside the diatom {col("l1_in").mean():.4f} vs background {np.nanmean(col("l1_out")):.4f} '
        f'({col("l1_in").mean() / max(np.nanmean(col("l1_out")), 1e-9):.0f}x more on the diatom)',
        '',
        'TRANSFER: do independent classifiers (never used for guidance) also see the target?',
    ]
    for name in args.transfer_name:
        ok = col(name + '_orig_correct').astype(bool)
        lines.append(f'  {name:16s} target seen {col(name + "_success").mean():.1%} '
                     f'(when it classified the original correctly: {col(name + "_success")[ok].mean():.1%}, n={int(ok.sum())})   '
                     f'p(target) {col(name + "_p_target_before").mean():.3f} -> {col(name + "_p_target_after").mean():.3f}')
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'figures: {args.out_dir}/figures/000.png ... {fig_no - 1:03d}.png | table: {args.out_dir}/metrics.csv')


if __name__ == '__main__':
    main()
