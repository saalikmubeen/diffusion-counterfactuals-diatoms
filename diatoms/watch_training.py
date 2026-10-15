"""Summarise a (running) diffusion fine-tuning run from its progress.csv.

Prints the latest step, loss, speed and ETA, lists saved checkpoints, and
writes a loss-curve plot (training_curve.png) into the run directory.
Safe to run at any time while training; it only reads files. CPU only.

  python diatoms/watch_training.py checkpoints/diatlas/ft_v1
  watch -n 300 python diatoms/watch_training.py checkpoints/diatlas/ft_v1   # refresh every 5 min
"""
import argparse
import csv
import glob
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def smooth(y, k):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode='valid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--smooth', type=int, default=10, help='moving-average window, in log rows')
    args = p.parse_args()

    path = os.path.join(args.run_dir, 'progress.csv')
    if not os.path.exists(path):
        print(f'no progress.csv yet in {args.run_dir} (training may still be starting)')
        return
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get('step')]
    if not rows:
        print('progress.csv is empty so far')
        return

    def col(name):
        return np.array([float(r[name]) if r.get(name) not in (None, '') else np.nan for r in rows])

    step, loss = col('step'), col('loss')
    last = rows[-1]
    age_min = (datetime.now().timestamp() - os.path.getmtime(path)) / 60

    print(f'run:          {args.run_dir}')
    print(f'latest step:  {int(step[-1])}   (progress.csv updated {age_min:.0f} min ago)')
    print(f'loss:         {loss[-1]:.4f}   (mean of last {min(args.smooth, len(loss))} logs: '
          f'{np.nanmean(loss[-args.smooth:]):.4f}; first log: {loss[0]:.4f})')
    if last.get('sec_per_step'):
        eta_h = float(last['eta_h'])
        finish = datetime.fromtimestamp(os.path.getmtime(path)) + timedelta(hours=eta_h)
        print(f'speed:        {float(last["sec_per_step"]):.2f} s/step   lr {float(last["lr"]):.2e}')
        print(f'elapsed:      {float(last["elapsed_h"]):.1f} h   ETA {eta_h:.1f} h  (finishes ~{finish:%a %H:%M})')
    log_txt = os.path.join(args.run_dir, 'log.txt')
    finished = os.path.exists(log_txt) and 'training finished' in open(log_txt).read()
    if not finished:
        # runs from before the "training finished" line: final step saved => done
        total = float(last['eta_h']) * 3600 / float(last['sec_per_step']) + step[-1] if last.get('eta_h') else None
        finished = total is not None and os.path.exists(
            os.path.join(args.run_dir, f'model{int(round(total)):06d}.pt'))
    if finished:
        print('status:       FINISHED (final checkpoint saved)')
    elif age_min > 15:
        print('WARNING: no new log line for >15 min. Check that the job is still running (nvidia-smi).')

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, 'ema_*.pt')))
    print('EMA checkpoints:', ', '.join(os.path.basename(c) for c in ckpts) if ckpts else 'none yet')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step, loss, color='0.8', lw=0.8, label='loss (per log)')
    k = args.smooth
    if len(loss) >= k:
        ax.plot(step[k - 1:], smooth(loss, k), color='C0', lw=1.8, label=f'moving average ({k})')
    ax.set_xlabel('step')
    ax.set_ylabel('training loss (MSE + VB)')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(os.path.basename(os.path.normpath(args.run_dir)))
    fig.tight_layout()
    out = os.path.join(args.run_dir, 'training_curve.png')
    fig.savefig(out, dpi=120)
    print('plot:        ', out)


if __name__ == '__main__':
    main()
