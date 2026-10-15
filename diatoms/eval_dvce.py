"""Evaluate DVCE counterfactuals saved by diatlas_VCEs.py (DiatlasVCEs/.../<num_imgs>.pth).

Per counterfactual (original image x, counterfactual x', target class t):
  success      the classifier predicts t on x'
  p_target     the classifier's probability of t on x' (and on x, before)
  lpips        perceptual distance x -> x' (lower = closer to the original)
  l1           mean absolute pixel change
  l1_in / l1_out  mean change inside / outside the diatom mask (from x): a good
               counterfactual changes the diatom, not the background
The guiding classifier's probabilities come from the .pth file. --eval_ckpt adds
an independent classifier (one not used for guidance): if it also predicts the
target, the change is less likely to be an artefact of the guiding model.

Example:
  python diatoms/eval_dvce.py DiatlasVCEs/<run dir>/6.pth \
      --eval_ckpt DiatlasModels/r50madry_native_eps0.5/best.pt --out results/dvce/<name>.csv
"""
import argparse
import csv
import os
import sys

import lpips
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_metrics import diatom_mask  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pth', help='data_dict saved by diatlas_VCEs.py')
    p.add_argument('--eval_ckpt', nargs='*', default=[],
                   help='independent diatom classifiers (checkpoints or "independent"), each gets its own columns')
    p.add_argument('--eval_name', nargs='*', default=[], help='short names for --eval_ckpt (column prefixes)')
    p.add_argument('--out', help='per-counterfactual CSV')
    args = p.parse_args()

    d = torch.load(args.pth, map_location='cpu')
    x = d['gt_imgs'].float()                         # (N, 3, 256, 256) in [0, 1]
    xcf = d['counterfactuals'][:, 0, 0].float()      # (N, 3, 256, 256)
    t = d['targets'].long()
    y = d['gt_labels'][:len(x)].long()
    p_out = d['out_probabilities'][:, 0, 0]          # guiding classifier on x'
    p_in = d['in_probabilities'][:, 0, 0]            # guiding classifier on x
    n = len(x)

    lp = lpips.LPIPS(net='alex', verbose=False).eval()
    with torch.no_grad():
        d_lpips = lp(x * 2 - 1, xcf.clamp(0, 1) * 2 - 1).flatten()
    diff = (xcf.clamp(0, 1) - x).abs().mean(1)       # (N, 256, 256)
    rows = []
    masks = [torch.from_numpy(diatom_mask(x[i].permute(1, 2, 0).numpy())) for i in range(n)]

    evals = []
    names = args.eval_name or [f'indep{i}' for i in range(len(args.eval_ckpt))]
    if args.eval_ckpt:
        from diatom_classifier import load_diatom_classifier
        for name, ck in zip(names, args.eval_ckpt):
            clf, classes, desc = load_diatom_classifier(ck)
            print(f'independent classifier {name}:', desc)
            with torch.no_grad():
                pin = torch.cat([clf(x[i:i + 50]).softmax(1) for i in range(0, n, 50)])
                pout = torch.cat([clf(xcf[i:i + 50].clamp(0, 1)).softmax(1) for i in range(0, n, 50)])
            evals.append((name, pin, pout))

    for i in range(n):
        m = masks[i]
        r = dict(idx=i, true=int(y[i]), target=int(t[i]),
                 success=int(p_out[i].argmax() == t[i]),
                 p_target_before=float(p_in[i, t[i]]), p_target_after=float(p_out[i, t[i]]),
                 p_true_after=float(p_out[i, y[i]]),
                 lpips=float(d_lpips[i]), l1=float(diff[i].mean()),
                 l1_in=float(diff[i][m].mean()), l1_out=float(diff[i][~m].mean()) if (~m).any() else float('nan'))
        for name, pin, pout in evals:
            r.update({f'{name}_orig_correct': int(pin[i].argmax() == y[i]), f'{name}_success': int(pout[i].argmax() == t[i]),
                      f'{name}_p_target_before': float(pin[i, t[i]]), f'{name}_p_target_after': float(pout[i, t[i]])})
        rows.append(r)

    def mean(k):
        return np.mean([r[k] for r in rows])

    print(f'{n} counterfactuals from {args.pth}')
    print(f'  guiding classifier:     success {mean("success"):.2f}   p(target) {mean("p_target_before"):.3f} -> '
          f'{mean("p_target_after"):.3f}   p(true) after {mean("p_true_after"):.3f}')
    for name, _, _ in evals:
        print(f'  {name}: target seen {mean(name + "_success"):.2f}   p(target) '
              f'{mean(name + "_p_target_before"):.3f} -> {mean(name + "_p_target_after"):.3f}')
    print(f'  closeness: LPIPS {mean("lpips"):.3f}   L1 {mean("l1"):.4f}   '
          f'change inside diatom {mean("l1_in"):.4f} vs background {np.nanmean([r["l1_out"] for r in rows]):.4f}')
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('per-counterfactual table:', args.out)


if __name__ == '__main__':
    main()
