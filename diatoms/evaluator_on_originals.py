"""How a classifier not used for guidance judges the original images and their counterfactuals.

Reports, for one DVCE run: accuracy on the originals, how often it already predicts the
target on the originals, target / source predictions on the counterfactuals, mean
p(target) before and after, and the share of counterfactual predictions in the target's genus.

  python diatoms/evaluator_on_originals.py --pth <wid_0 pth> <wid_1 pth> \
      --eval_ckpt DiatlasModels/r50imagenet_page_eps0/best.pt --out <file.txt>
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--pth', nargs='+', required=True)
ap.add_argument('--eval_ckpt', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
ds = [torch.load(p, map_location='cpu') for p in args.pth]
x = torch.cat([d['gt_imgs'].float() for d in ds])
c = torch.cat([d['counterfactuals'][:, 0, 0].float().clamp(0, 1) for d in ds])
y = torch.cat([d['gt_labels'][:len(d['gt_imgs'])] for d in ds])
t = torch.cat([d['targets'] for d in ds])
clf, classes, desc = load_diatom_classifier(args.eval_ckpt)
with torch.no_grad():
    po = torch.cat([clf(x[i:i + 50]).softmax(1) for i in range(0, len(x), 50)])
    pc = torch.cat([clf(c[i:i + 50]).softmax(1) for i in range(0, len(c), 50)])
n, r = len(t), torch.arange(len(t))
genus = {row['species_code']: row['genus'] for row in csv.DictReader(open(os.path.join(DATA_NATIVE, 'splits_random.csv')))
         if row['species_code']}
same_genus = np.mean([genus.get(classes[int(a)]) == genus.get(classes[int(b)]) for a, b in zip(pc.argmax(1), t)])
ok = po.argmax(1) == y
lines = [f'evaluator: {desc}', f'{n} counterfactuals from {args.pth}', '',
         f'originals:        source species correct {ok.float().mean():.1%} | already predicts target {(po.argmax(1) == t).float().mean():.1%} | mean p(target) {po[r, t].mean():.3f}',
         f'counterfactuals:  predicts target {(pc.argmax(1) == t).float().mean():.1%} | still source {(pc.argmax(1) == y).float().mean():.1%} | mean p(target) {pc[r, t].mean():.3f} | prediction in target genus {same_genus:.1%}',
         f'among originals it classified correctly (n={int(ok.sum())}): predicts target on counterfactual {(pc.argmax(1) == t)[ok].float().mean():.1%}']
print('\n'.join(lines))
open(args.out, 'w').write('\n'.join(lines) + '\n')
