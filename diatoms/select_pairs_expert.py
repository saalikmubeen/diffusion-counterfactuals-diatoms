"""DVCE pairs from an expert's list of morphologically similar species.

For every source specimen of a listed species (test image in BOTH splits, so neither the
classifier nor the diffusion model has seen it), two targets:
  1. its expert look-alike (the other species of the pair)
  2. a random other species of the same genus with >= --min_train training images
     (the rule of the main experiment), as a comparison
Pairs are used in both directions (A -> B and B -> A). Species missing from the
classifier's classes are reported and skipped.

  python diatoms/select_pairs_expert.py --out evaluation/vce_pairs_expert.json
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import DATA_NATIVE  # noqa: E402

EXPERT_PAIRS = [('ADPY', 'ADMO'), ('ADRI', 'ADSU'), ('APED', 'AMID'), ('CEUG', 'CLNT'), ('CMEN', 'CMED'),
                ('DVUL', 'DEHR'), ('NCRY', 'NCPR'), ('NGRE', 'NGER'), ('NPAE', 'NIAR'), ('NPAL', 'NCPL')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='DiatlasModels/r50madry_random_eps1.0/best.pt')
    ap.add_argument('--min_train', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    classes = set(torch.load(args.ckpt, map_location='cpu')['classes'])
    with open(os.path.join(DATA_NATIVE, 'splits_random.csv')) as f:
        rows = list(csv.DictReader(f))
    n_train = collections.Counter(r['species_code'] for r in rows if r['split_random'] == 'train')
    genus = {r['species_code']: r['genus'] for r in rows if r['species_code']}
    in_genus = collections.defaultdict(list)
    for sp, g in genus.items():
        if sp in classes and n_train[sp] >= args.min_train:
            in_genus[g].append(sp)

    entries, skipped = [], []
    for a, b in EXPERT_PAIRS:
        if a not in classes or b not in classes:
            skipped.append(f'{a}<->{b}: ' + ', '.join(c for c in (a, b) if c not in classes) + ' not in the classifier/data')
            continue
        for src, look_alike in ((a, b), (b, a)):
            imgs = sorted(r['path'] for r in rows if r['species_code'] == src
                          and r['split_page'] == 'test' and r['split_random'] == 'test')
            others = sorted(s for s in in_genus[genus[src]] if s not in (src, look_alike))
            for p in imgs:
                rand = rng.choice(others)
                entries.append(dict(path=p, targets=[look_alike, rand], source_species=src, genus=genus[src],
                                    expert_pair=f'{a}<->{b}', target_kinds=['expert_look_alike', 'random_same_genus'],
                                    n_train=[n_train[src], n_train[look_alike], n_train[rand]]))
    meta = dict(rule='target 1 = expert look-alike, target 2 = random species of the same genus '
                     f'(>= {args.min_train} training images); sources = test images in both splits',
                pairs=[f'{a}<->{b}' for a, b in EXPERT_PAIRS], skipped=skipped)
    with open(args.out, 'w') as f:
        json.dump(dict(meta=meta, entries=entries), f, indent=1)
    print(f'{len(entries)} source specimens -> {args.out}')
    for s in skipped:
        print('skipped', s)
    print(collections.Counter(e['expert_pair'] for e in entries))


if __name__ == '__main__':
    main()
