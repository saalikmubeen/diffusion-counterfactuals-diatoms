"""Choose source -> target pairs for DVCE on DIATLAS, following the DVCE paper's idea
(Augustin et al. 2022): targets are semantically close to the source class. The
paper uses "the same WordNet cluster"; for diatoms the equivalent is the taxonomy,
so targets are species of the SAME GENUS. No A/B crossover (only needed for FID).

Source images:
  * test image in BOTH splits (page split and random split), so neither the
    diffusion model (trained on page-split train+val) nor the classifier
    (trained on random-split train) has seen it
  * the classifier predicts the true species with p >= --min_conf
  * the species has >= --min_train training images (random split)
  * at most one image per species and --max_per_genus species per genus
Targets: 2 other species of the same genus with >= --min_train training images,
chosen at random (fixed seed).

Writes the JSON diatlas_VCEs.py reads ({"entries": [{"path", "targets", ...}]}).

Example:
  CUDA_VISIBLE_DEVICES=0 python diatoms/select_pairs_paper.py \
      --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt --out evaluation/vce_pairs_paper50.json
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from train_classifier import DATA_NATIVE  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='classifier to explain')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=50, help='number of source images (0 = all eligible)')
    p.add_argument('--all_per_species', action='store_true',
                   help='allow several images per species and no per-genus limit (for a full evaluation)')
    p.add_argument('--min_conf', type=float, default=0.5)
    p.add_argument('--min_train', type=int, default=20)
    p.add_argument('--max_per_genus', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, classes, desc = load_diatom_classifier(args.ckpt)
    model = model.to(device)
    idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['species_code'].strip() and r['species_code'] in idx]
    n_train = collections.Counter(r['species_code'] for r in rows if r['split_random'] == 'train')
    genus_of = {r['species_code']: r['genus'] for r in rows}
    known = {s for s in n_train if n_train[s] >= args.min_train}
    species_in_genus = collections.defaultdict(list)
    for s in sorted(known):
        species_in_genus[genus_of[s]].append(s)

    cands = [r for r in rows if r['split_page'] == 'test' and r['split_random'] == 'test'
             and r['species_code'] in known and len(species_in_genus[genus_of[r['species_code']]]) >= 3]
    print(f'classifier: {desc}')
    print(f'candidates: {len(cands)} test images (both splits) of species with >= {args.min_train} training images '
          f'in genera with >= 3 such species')

    to_t = transforms.ToTensor()
    probs = []
    with torch.no_grad():
        for i in range(0, len(cands), 64):
            x = torch.stack([to_t(Image.open(os.path.join(DATA_NATIVE, r['path'])).convert('RGB')) for r in cands[i:i + 64]])
            probs.append(model(x.to(device)).softmax(1).cpu())
    probs = torch.cat(probs)
    correct = [(probs[i].argmax().item() == idx[r['species_code']]) for i, r in enumerate(cands)]
    print(f'classifier correct on {sum(correct)}/{len(cands)} candidates ({100 * sum(correct) / len(cands):.1f}%)')

    order = list(range(len(cands)))
    rng.shuffle(order)
    used, per_genus, entries = set(), collections.Counter(), []
    for i in order:
        r = cands[i]
        sp, g = r['species_code'], r['genus']
        y = idx[sp]
        if not args.all_per_species and (sp in used or per_genus[g] >= args.max_per_genus):
            continue
        if not correct[i] or probs[i, y].item() < args.min_conf:
            continue
        others = [s for s in species_in_genus[g] if s != sp]
        targets = rng.sample(others, 2)
        entries.append({'path': r['path'], 'targets': targets, 'source_species': sp, 'genus': g,
                        'source_conf': round(probs[i, y].item(), 3),
                        'target_conf_before': [round(probs[i, idx[t]].item(), 4) for t in targets],
                        'n_train': [n_train[sp]] + [n_train[t] for t in targets],
                        'atlas_source': r['source'], 'page': r['page']})
        used.add(sp)
        per_genus[g] += 1
        if args.n and len(entries) == args.n:
            break

    meta = {'classifier': desc, 'ckpt': args.ckpt, 'rule': 'targets = 2 random species of the same genus (no crossover)',
            'settings': vars(args), 'n_entries': len(entries), 'n_genera': len(per_genus)}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'entries': entries}, f, indent=1)
    print(f'{len(entries)} source images from {len(per_genus)} genera -> {args.out}')
    for e in entries[:10]:
        print(f"  {e['source_species']:6s} ({e['genus']}, p={e['source_conf']:.2f}) -> {e['targets']}   {e['path']}")


if __name__ == '__main__':
    main()
