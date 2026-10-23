"""Side-by-side figure: classifier push vs. actual change, robust vs. non-robust classifier.

Both runs must use the same pair list in the same order (diatlas_VCEs.py --log_gradients,
--world_size 1), so counterfactual k is the same source and target in both.
Columns per row: original | robust: change | robust: push | non-robust: change | non-robust: push

  python diatoms/gradient_compare_fig.py --robust <run dir>/gradient_logs --nonrobust <run dir>/gradient_logs \
      --rows 0 6 14 --out results/dvce/gradient_maps/robust_vs_nonrobust.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402


def load(log_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(log_dir, 'batch_*.pth'))):
        d = torch.load(f, map_location='cpu')
        G = d['grad_class'].float()
        G = (G / G.flatten(2).norm(dim=2)[..., None, None].clamp_min(1e-20)).abs().sum(0).numpy()   # (B,H,W)
        for b in range(G.shape[0]):
            init, fin = d['init_image'][b].mean(0).numpy(), d['final_image'][b].mean(0).numpy()
            out.append(dict(init=init, change=ndimage.gaussian_filter(np.abs(fin - init), 4),
                            push=ndimage.gaussian_filter(G[b], 4)))
    return out


def n01(a):
    hi = np.quantile(a, 0.99)
    return np.clip(a / hi, 0, 1) if hi > 0 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robust', required=True)
    ap.add_argument('--nonrobust', required=True)
    ap.add_argument('--rows', type=int, nargs='+', default=[0, 6, 14])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    R, N = load(args.robust), load(args.nonrobust)
    assert len(R) == len(N)
    for k in args.rows:
        assert np.abs(R[k]['init'] - N[k]['init']).max() < 1e-3, 'runs not aligned'
    titles = ['original', 'robust:\nactual change', 'robust:\nclassifier push',
              'non-robust:\nactual change', 'non-robust:\nclassifier push']
    fig, ax = plt.subplots(len(args.rows), 5, figsize=(5 * 2.2, len(args.rows) * 2.3), squeeze=False)
    for r, k in enumerate(args.rows):
        init = R[k]['init']
        ax[r, 0].imshow(init, cmap='gray', vmin=0, vmax=1)
        for c, (run, key) in enumerate([(R, 'change'), (R, 'push'), (N, 'change'), (N, 'push')]):
            if key == 'change':
                ax[r, c + 1].imshow(n01(run[k][key]), cmap='magma')
            else:
                ax[r, c + 1].imshow(init, cmap='gray', vmin=0, vmax=1)
                ax[r, c + 1].imshow(n01(run[k][key]), cmap='jet', alpha=0.5)
        for c in range(5):
            ax[r, c].axis('off')
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print('saved', args.out)


if __name__ == '__main__':
    main()
