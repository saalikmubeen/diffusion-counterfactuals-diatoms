"""Second preprocessing of DIATLAS: native pixel size instead of stretch-to-fit.

Uses the the PadCropDiatom transform (diatoms/padcrop_transform.py)
without physical rescaling (src_scale = tgt_scale), with his 224 px crop but a
256x256 canvas, because:
  * his classifier is trained on this kind of input (84.8% val accuracy vs
    ~30% on our stretch-to-fit images, see diatoms/COMMANDS.md section 8)
  * size is a real species trait, and stretching every diatom to fill the
    image destroys it

Per image: atlas crop -> long axis vertical (transpose) -> center-crop to 224
if larger -> seamless clone (flipped upside down, as in his code) onto a
256x256 canvas filled with the image median -> grayscale PNG.
The center 224x224 of each image is pixel-identical to his 224 input, so DVCE
should give the classifier a center crop (not a resize): checked, 84.2% on
1,000 val images either way, while feeding the full 256 image gives 11% and
resizing 256 -> 224 gives 70%.

Same images and the same train/val/test split as data/diatlas (connective
views dropped, page-grouped split), read from data/diatlas/splits.csv.

Outputs (under --out, default data/diatlas_nativesize_crop224_canvas256):
  256/<split>/<taxon>/<atlas>/<name>.png
  splits.csv   path,orig_path,taxon,genus,species_code,source,page,split,orig_w,orig_h,cropped
  montage.png  64 random processed images
  README.md    what was done (written by hand, not by this script)

Usage:
  ~/miniconda3/envs/python_38_dvces_cu111/bin/python diatoms/prepare_diatlas_nativesize.py
"""
import argparse
import csv
import os
import random
import sys
from multiprocessing import Pool

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from padcrop_transform import pad_crop_diatom  # noqa: E402

SIZE = 256   # canvas (diffusion model input)
CROP = 224   # his crop size (classifier input = center 224 crop)
TGT_SCALE = 0.0826  # irrelevant without rescaling (src_scale = tgt_scale), kept for clarity


def work(job):
    src, dst = job
    img = Image.open(src)
    w, h = img.size
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pad_crop_diatom(img, TGT_SCALE, TGT_SCALE, tgt_size=SIZE, to_gray=True, crop_size=CROP).save(dst)
    return w, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/mnt/diatoms/datasets/diatlas')
    p.add_argument('--splits', default='data/diatlas/splits.csv', help='reuse this split')
    p.add_argument('--out', default='data/diatlas_nativesize_crop224_canvas256')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()

    with open(os.path.join(args.root, 'diatoms.csv')) as f:
        species_of = {r['path'].lstrip('/'): r['species_code'] for r in csv.DictReader(f)}
    with open(args.splits) as f:
        rows = list(csv.DictReader(f))

    jobs = []
    for r in rows:
        r['species_code'] = species_of[r['orig_path']]
        # same relative layout as data/diatlas: 256/<split>/<taxon>/<atlas>/<name>.png
        jobs.append((os.path.join(args.root, r['orig_path']), os.path.join(args.out, r['path'])))

    with Pool(args.workers) as pool:
        sizes = pool.map(work, jobs, chunksize=64)
    for r, (w, h) in zip(rows, sizes):
        r['cropped'] = int(max(w, h) > CROP)

    fields = ['path', 'orig_path', 'taxon', 'genus', 'species_code', 'source', 'page', 'split',
              'orig_w', 'orig_h', 'cropped']
    with open(os.path.join(args.out, 'splits.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    n_crop = sum(r['cropped'] for r in rows)
    print(f'{len(rows)} images; {n_crop} ({100 * n_crop / len(rows):.1f}%) longer than {CROP}px and center-cropped')

    sample = random.Random(0).sample(rows, 64)
    t = 128
    sheet = Image.new('L', (16 * t, 4 * t), 255)
    for i, r in enumerate(sample):
        sheet.paste(Image.open(os.path.join(args.out, r['path'])).resize((t, t), Image.LANCZOS), ((i % 16) * t, (i // 16) * t))
    sheet.save(os.path.join(args.out, 'montage.png'))
    print('montage:', os.path.join(args.out, 'montage.png'))


if __name__ == '__main__':
    main()
