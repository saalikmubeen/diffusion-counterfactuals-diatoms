"""Choose source -> target pairs for DVCE on DIATLAS.

  * confusions are counted on the VAL split, pairs are picked from the TEST
    split, so the test images aren't selected using test results
  * source image: a test image the classifier gets right with p >= --min_conf,
    from a species with >= --min_train train images; at most one image per
    species and --max_per_genus species per genus (variety)
  * targets: two other species of the SAME genus with >= --min_train train
    images, ranked by how often the classifier confuses them with the source
    species on val (either direction); ties / no confusions -> larger species first
Writes the JSON diatlas_VCEs.py reads ({"entries": [{"path", "targets", ...}]}),
with extra fields (source species, genus, confidence, confusion counts) for reference.

Example (CPU is fine for a ResNet-18, ~2 min):
  CUDA_VISIBLE_DEVICES= python diatoms/select_pairs.py \
      --ckpt DiatlasModels/resnet18_native_eps0.5/best.pt --out evaluation/vce_pairs_native.json
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diatom_classifier import load_diatom_classifier  # noqa: E402
from train_classifier import DATA_NATIVE, DiatlasSpecies  # noqa: E402


def predict(model, ds, device, bs=100):
    model = model.to(device)
    probs = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, bs, num_workers=8):
            probs.append(model(x.to(device)).softmax(1).cpu())
    return torch.cat(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help="classifier checkpoint or 'independent'")
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=50, help='number of source images')
    p.add_argument('--min_conf', type=float, default=0.5)
    p.add_argument('--min_train', type=int, default=20)
    p.add_argument('--max_per_genus', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, classes, desc = load_diatom_classifier(args.ckpt)
    idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(DATA_NATIVE, 'splits.csv')) as f:
        rows = list(csv.DictReader(f))
    genus_of = {r['species_code']: r['genus'] for r in rows if r['species_code'].strip()}
    n_train = collections.Counter(r['species_code'] for r in rows if r['split'] == 'train')

    val = DiatlasSpecies('val', classes, augment=False, preproc='nativesize')
    test = DiatlasSpecies('test', classes, augment=False, preproc='nativesize')
    print(f'classifier: {desc}\npredicting val ({len(val)}) and test ({len(test)}) on {device} ...', flush=True)
    pv, pt = predict(model, val, device), predict(model, test, device)
    print(f'val acc {(pv.argmax(1) == torch.tensor(val.targets)).float().mean():.3f}, '
          f'test acc {(pt.argmax(1) == torch.tensor(test.targets)).float().mean():.3f}')

    confusion = collections.Counter()
    for y, yhat in zip(val.targets, pv.argmax(1).tolist()):
        if y != yhat:
            confusion[frozenset((y, yhat))] += 1

    def targets_for(src):
        g = genus_of[classes[src]]
        cands = [i for i, c in enumerate(classes)
                 if c.strip() and i != src and genus_of.get(c) == g and n_train[c] >= args.min_train]
        cands.sort(key=lambda t: (-confusion[frozenset((src, t))], -n_train[classes[t]], classes[t]))
        return cands[:2]

    test_rows = {r['path']: r for r in rows}
    order = list(range(len(test)))
    random.Random(args.seed).shuffle(order)
    used_species, per_genus, entries = set(), collections.Counter(), []
    for i in order:
        y = test.targets[i]
        sp = classes[y]
        if sp in used_species or n_train[sp] < args.min_train:
            continue
        if pt[i].argmax().item() != y or pt[i, y].item() < args.min_conf:
            continue
        g = genus_of[sp]
        if per_genus[g] >= args.max_per_genus:
            continue
        tg = targets_for(y)
        if len(tg) < 2:
            continue
        rel = os.path.relpath(test.paths[i], DATA_NATIVE)
        entries.append({'path': rel, 'targets': [classes[t] for t in tg],
                        'source_species': sp, 'genus': g, 'source_conf': round(pt[i, y].item(), 3),
                        'target_conf_before': [round(pt[i, t].item(), 4) for t in tg],
                        'val_confusions_with_targets': [confusion[frozenset((y, t))] for t in tg],
                        'n_train': [n_train[sp]] + [n_train[classes[t]] for t in tg],
                        'atlas': test_rows[rel]['source'], 'cropped': int(test_rows[rel]['cropped'])})
        used_species.add(sp)
        per_genus[g] += 1
        if len(entries) == args.n:
            break

    meta = {'classifier': desc, 'ckpt': args.ckpt, 'settings': vars(args),
            'n_genera': len(per_genus), 'n_with_confused_target': sum(any(e['val_confusions_with_targets']) for e in entries)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'entries': entries}, f, indent=1)
    print(f'{len(entries)} source images from {len(per_genus)} genera; '
          f'{meta["n_with_confused_target"]} have at least one target the classifier confuses on val -> {args.out}')
    for e in entries[:8]:
        print(f"  {e['source_species']:6s} ({e['genus']}, p={e['source_conf']:.2f}) -> {e['targets']} "
              f"confusions {e['val_confusions_with_targets']}  {e['path']}")


if __name__ == '__main__':
    main()
