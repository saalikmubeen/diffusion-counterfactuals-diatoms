"""One figure per source diatom: original, its two counterfactuals, and a real
training photo of the source and of each target species for comparison.

Layout (one column per class: source | target 1 | target 2):
  row 1  original              | counterfactual -> T1  | counterfactual -> T2
  rows 2-4  real photos of source | real photos of T1  | real photos of T2  (3 examples each)
(native size, same scale)

Input: the .pth saved by diatlas_VCEs.py, the classifier it used (for class
names) and the data root. Reference photos come from the training split given
by --ref_split_mode (random split: what the classifier learned from).

Example:
  python diatoms/dvce_figure.py DiatlasVCEs/<run>/<settings>/6.pth \
      --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt --source 0 --out results/dvce/figure_0.png
"""
import argparse
import csv
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pth')
    p.add_argument('--ckpt', required=True, help='classifier used for the DVCEs (class names)')
    p.add_argument('--source', type=int, default=0, help='which source image in the .pth (0, 1, 2, ...)')
    p.add_argument('--n_ref', type=int, default=3, help='real photos per species (rows below the counterfactuals)')
    p.add_argument('--ref_split_mode', choices=['random', 'page'], default='random')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--indep_ckpt', default='none',
                   help="optional: another classifier whose verdict is added to the titles (default: none)")
    p.add_argument('--out', required=True)
    args = p.parse_args()

    d = torch.load(args.pth, map_location='cpu')
    _, classes, _ = load_diatom_classifier(args.ckpt)
    k = 2 * args.source                      # counterfactuals come in pairs per source image
    x0 = d['gt_imgs'][k].float()
    y = int(d['gt_labels'][k])
    cfs = [d['counterfactuals'][k + j, 0, 0].float().clamp(0, 1) for j in range(2)]
    tgts = [int(d['targets'][k + j]) for j in range(2)]
    p_after = [float(d['out_probabilities'][k + j, 0, 0, tgts[j]]) for j in range(2)]
    p_before = [float(d['in_probabilities'][k + j, 0, 0, tgts[j]]) for j in range(2)]
    p_src = float(d['in_probabilities'][k, 0, 0, y])

    verdict = [None, None]
    if args.indep_ckpt != 'none':
        indep, indep_classes, _ = load_diatom_classifier(args.indep_ckpt)
        assert list(indep_classes) == list(classes), 'independent classifier needs the same classes'
        with torch.no_grad():
            pr = indep(torch.stack(cfs)).softmax(1)
        verdict = [(classes[int(pr[j].argmax())], float(pr[j, tgts[j]])) for j in range(2)]

    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        rows = list(csv.DictReader(f))
    col = 'split_random' if args.ref_split_mode == 'random' else 'split_page'
    genus_of = {r['species_code']: r['genus'] for r in rows if r['species_code']}
    rng = random.Random(args.seed)

    def refs(species):
        pool = sorted(r['path'] for r in rows if r['species_code'] == species and r[col] == 'train')
        pick = rng.sample(pool, min(args.n_ref, len(pool)))
        return [np.asarray(Image.open(os.path.join(DATA_NATIVE, q)).convert('L')) for q in pick], len(pool)

    species = [classes[y], classes[tgts[0]], classes[tgts[1]]]
    ref_imgs = [refs(s) for s in species]

    to_np = lambda t: (t.permute(1, 2, 0).numpy().mean(2) * 255).astype(np.uint8)
    n_rows = 1 + args.n_ref
    fig, ax = plt.subplots(n_rows, 3, figsize=(3 * 3.1, n_rows * 3.3), squeeze=False)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    ax[0, 0].imshow(to_np(x0), cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title(f'ORIGINAL: {species[0]}\nclassifier p = {p_src:.2f}', fontsize=10)
    for j in range(2):
        ax[0, j + 1].imshow(to_np(cfs[j]), cmap='gray', vmin=0, vmax=255)
        title = f'COUNTERFACTUAL -> {species[j + 1]}\nour classifier: p({species[j + 1]}) {p_before[j]:.2f} -> {p_after[j]:.2f}'
        if verdict[j] is not None:
            pred, pt = verdict[j]
            ok = 'agrees' if pred == species[j + 1] else f'says {pred}'
            title += f'\nindependent classifier: {ok} (p {pt:.2f})'
        ax[0, j + 1].set_title(title, fontsize=9.5)
    for c in range(3):
        imgs, n_pool = ref_imgs[c]
        for r in range(args.n_ref):
            if r < len(imgs):
                ax[1 + r, c].imshow(imgs[r], cmap='gray', vmin=0, vmax=255)
        role = 'source' if c == 0 else f'target {c}'
        ax[1, c].set_title(f'REAL {species[c]} ({role})\nexample training photos', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print('saved', args.out)


if __name__ == '__main__':
    main()
