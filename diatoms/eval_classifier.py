"""Evaluate diatom classifiers on a split of the native-size data: clean top-1 /
top-5 accuracy and accuracy under L2 PGD attacks (fp32), same input as DVCE
(256 native-size image -> center 224 crop -> model with ImageNet normalisation).

--ckpt takes checkpoints saved by diatoms/train_classifier.py, or 'independent'
for the original independent ResNet-18. Images without species_code are skipped.

Example:
  CUDA_VISIBLE_DEVICES=0 python diatoms/eval_classifier.py --split test \
      --ckpt independent --ckpt DiatlasModels/resnet18_native_eps0.5/best.pt \
      --ckpt DiatlasModels/resnet18_native_eps1.0/best.pt --eps 0.5 1.0
"""
import argparse
import os
import sys

import torch
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_classifier import (DiatlasSpecies, NormalizedClassifier, ResizeToClassifier,  # noqa: E402
                              load_independent_resnet18, pgd_l2, independent_classes)


def load(ckpt):
    if ckpt == 'independent':
        return ResizeToClassifier(NormalizedClassifier(load_independent_resnet18()), mode='centercrop'), 'independent classifier (original)'
    ck = torch.load(ckpt, map_location='cpu')
    net = getattr(torchvision.models, ck.get('arch', 'resnet18'))(num_classes=ck['num_classes'])
    m = NormalizedClassifier(net)
    m.load_state_dict(ck['state_dict'])
    mode = 'centercrop' if ck.get('preproc', 'nativesize') == 'nativesize' else 'resize'
    return ResizeToClassifier(m, mode=mode), f"{os.path.basename(os.path.dirname(ckpt))} (epoch {ck.get('epoch')})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--eps', type=float, nargs='*', default=[0.5, 1.0])
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--robust_n', type=int, default=0, help='images for the attacks (0 = all)')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--split_mode', choices=['page', 'random'], default='page')
    args = p.parse_args()

    classes = independent_classes()
    ds = DiatlasSpecies(args.split, classes, augment=False, preproc='nativesize', split_mode=args.split_mode)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if args.robust_n:
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:args.robust_n].tolist()
        rob_dl = DataLoader(torch.utils.data.Subset(ds, idx), args.batch_size, num_workers=8, pin_memory=True)
    else:
        rob_dl = dl
    print(f'{args.split} split ({args.split_mode}): {len(ds)} images, {len(set(ds.targets))} species present; '
          f'attacks: L2 PGD-{args.steps} (fp32) on {len(rob_dl.dataset)} images\n')

    header = f'{"model":40s} {"top-1":>7s} {"top-5":>7s}' + ''.join(f' {"eps " + str(e):>9s}' for e in args.eps)
    print(header)
    for ck in args.ckpt:
        model, name = load(ck)
        model = model.cuda().eval()
        top1 = top5 = n = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.cuda(), y.cuda()
                t5 = model(x).topk(5, 1).indices
                top1 += (t5[:, 0] == y).sum().item()
                top5 += (t5 == y[:, None]).any(1).sum().item()
                n += len(y)
        robs = []
        for eps in args.eps:
            torch.manual_seed(1234)
            ok = m = 0
            for x, y in rob_dl:
                x, y = x.cuda(), y.cuda()
                xa = pgd_l2(model, x, y, eps, args.steps)
                with torch.no_grad():
                    ok += (model(xa).argmax(1) == y).sum().item()
                m += len(y)
            robs.append(ok / m)
        print(f'{name:40s} {top1 / n:7.3f} {top5 / n:7.3f}' + ''.join(f' {r:9.3f}' for r in robs), flush=True)


if __name__ == '__main__':
    main()
