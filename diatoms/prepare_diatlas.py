"""Preprocess DIATLAS into 256x256 grayscale images plus a label-independent split.

Steps (see DIATOM_PLAN.md, section 2):
  * drop connective (girdle) views
  * split ~80/10/10 into train/val/test, grouped by (source, page) so that
    near-duplicates from one atlas page never cross splits
  * per image: grayscale, trim the atlas frame, make the long axis vertical,
    resize the long side to 256, pad to a square with the image's border median

Outputs (under --out):
  256/<split>/<taxon>/<atlas>/<name>.png   single-channel PNGs (loaders convert to RGB)
  splits.csv                               path,orig_path,taxon,genus,source,page,split,orig_w,orig_h
  split_report.txt                         image counts and per-class test coverage
  montage.png                              random processed samples

Usage:
  ~/miniconda3/envs/python_38_dvces_cu111/bin/python diatoms/prepare_diatlas.py
"""
import argparse
import collections
import csv
import os
import random
from multiprocessing import Pool

import numpy as np
from PIL import Image

SIZE = 256
TRIM = 2  # pixels removed from each edge; some crops carry a 1px atlas frame


def assign_splits(rows, seed, val_frac, test_frac):
    """Stratified group split: whole (source, page) groups go to one split.

    Atlas pages are organised by taxon (almost every page holds one taxon), and
    one page typically holds 15-30% of a taxon's images, so neither random nor
    proportional page assignment gives every taxon test images. Instead, each
    page belongs to its majority taxon, and per taxon the smallest pages go to
    test until test_frac is reached, then the next ones to val until val_frac,
    the rest to train. A taxon with >= 3 pages gets at least one page in each
    split; with 2 pages the smaller one alternates between test and val; a
    single page goes to train.
    """
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['source'], r['page'])].append(r)
    by_taxon = collections.defaultdict(list)
    for k in sorted(groups):
        majority = collections.Counter(r['taxon'] for r in groups[k]).most_common(1)[0][0]
        by_taxon[majority].append(k)
    two_page_turn = 0
    for taxon in sorted(by_taxon):
        keys = by_taxon[taxon]
        rng.shuffle(keys)
        keys.sort(key=lambda k: len(groups[k]))  # smallest first, random among ties
        n = sum(len(groups[k]) for k in keys)
        split_of = {k: 'train' for k in keys}
        if len(keys) == 2:
            split_of[keys[0]] = ('test', 'val')[two_page_turn % 2]
            two_page_turn += 1
        elif len(keys) >= 3:
            i = 0
            for split, frac in (('test', test_frac), ('val', val_frac)):
                taken = 0
                # take at least one page, keep at least one page for train
                while i < len(keys) - 1 and (taken == 0 or taken < frac * n):
                    if taken > 0 and taken + len(groups[keys[i]]) > 2 * frac * n:
                        break
                    split_of[keys[i]] = split
                    taken += len(groups[keys[i]])
                    i += 1
        for k in keys:
            for r in groups[k]:
                r['split'] = split_of[k]


def process(img):
    a = np.asarray(img.convert('L'))
    if min(a.shape) > 4 * TRIM + 8:
        a = a[TRIM:-TRIM, TRIM:-TRIM]
    if a.shape[1] > a.shape[0]:
        a = np.rot90(a)
    h, w = a.shape
    border = np.concatenate([a[:2].ravel(), a[-2:].ravel(), a[:, :2].ravel(), a[:, -2:].ravel()])
    fill = int(np.median(border))
    scale = SIZE / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray(np.ascontiguousarray(a)).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('L', (SIZE, SIZE), fill)
    canvas.paste(small, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas


def work(job):
    src, dst = job
    img = Image.open(src)
    w, h = img.size
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        process(img).save(dst)
    return w, h


def write_report(rows, path):
    lines = []
    split_n = collections.Counter(r['split'] for r in rows)
    lines.append(f'images: {len(rows)}  ' + '  '.join(f'{s}={split_n[s]}' for s in ('train', 'val', 'test')))
    lines.append(f"page groups: {len(set((r['source'], r['page']) for r in rows))}")
    for key, min_n in (('taxon', 100), ('genus', 200)):
        total = collections.Counter(r[key] for r in rows)
        classes = sorted(c for c, n in total.items() if n >= min_n)
        per = {s: collections.Counter(r[key] for r in rows if r['split'] == s and r[key] in classes)
               for s in ('train', 'val', 'test')}
        test_counts = sorted(per['test'][c] for c in classes)
        lines.append('')
        lines.append(f'{key} with >= {min_n} images: {len(classes)} classes')
        lines.append(f'  test images per class: min={test_counts[0]} median={test_counts[len(test_counts) // 2]} '
                     f'max={test_counts[-1]}')
        for t in (0, 5, 10):
            lines.append(f'  classes with <= {t} test images: {sum(x <= t for x in test_counts)}')
        weak = [c for c in classes if per['test'][c] < 5]
        if weak:
            lines.append('  classes with < 5 test images: ' + ', '.join(
                f"{c}(tr={per['train'][c]},va={per['val'][c]},te={per['test'][c]})" for c in weak))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def write_montage(rows, out_root, path, n=64, seed=0):
    sample = random.Random(seed).sample(rows, n)
    cols, t = 16, 128
    sheet = Image.new('L', (cols * t, (n // cols) * t), 255)
    for i, r in enumerate(sample):
        im = Image.open(os.path.join(out_root, r['path'])).resize((t, t), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * t, (i // cols) * t))
    sheet.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/mnt/diatoms/datasets/diatlas')
    p.add_argument('--out', default='data/diatlas')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()

    with open(os.path.join(args.root, 'diatoms.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['connective'] != 'True']
    for r in rows:
        r['taxon'], r['genus'] = r['official_taxon'], r['genus_code']
    assign_splits(rows, args.seed, args.val_frac, args.test_frac)

    jobs = []
    for r in rows:
        rel = r['path'].lstrip('/')  # diatoms/<taxon>/<atlas>/<name>.<ext>
        _, taxon_dir, atlas, name = rel.split('/')
        r['orig_path'] = rel
        r['path'] = os.path.join('256', r['split'], taxon_dir, atlas, os.path.splitext(name)[0] + '.png')
        jobs.append((os.path.join(args.root, rel), os.path.join(args.out, r['path'])))

    with Pool(args.workers) as pool:
        sizes = pool.map(work, jobs, chunksize=64)
    for r, (w, h) in zip(rows, sizes):
        r['orig_w'], r['orig_h'] = w, h

    fields = ['path', 'orig_path', 'taxon', 'genus', 'source', 'page', 'split', 'orig_w', 'orig_h']
    with open(os.path.join(args.out, 'splits.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    write_report(rows, os.path.join(args.out, 'split_report.txt'))
    write_montage(rows, args.out, os.path.join(args.out, 'montage.png'))


if __name__ == '__main__':
    main()
