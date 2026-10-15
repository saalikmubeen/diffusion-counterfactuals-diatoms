"""Fine-tune the OpenAI 256x256 unconditional diffusion model on DIATLAS.

Thin wrapper around the vendored guided-diffusion TrainLoop
(blended_diffusion/guided_diffusion). Differences from scripts/image_train.py:
  * images come from data/diatlas/splits.csv (train+val only, never test)
  * augmentation: random horizontal AND vertical flips (diatoms are roughly
    symmetric along both axes); images are already 256x256, so no crop/resize
  * the architecture is fixed to the one DVCE loads
    (blended_diffusion/optimization/dff_attack.py), so a checkpoint from here
    can be dropped into the DVCE pipeline
  * the distributed setup respects CUDA_VISIBLE_DEVICES and runs on localhost
  * works around a DVCE-authors change that breaks training (_patch_vb_terms)
    and a guided-diffusion multi-GPU resume deadlock (_patch_checkpoint_loading)

Launch (one process per GPU) with diatoms/train_diffusion.sh.
"""
import argparse
import csv
import os
import socket
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch as th
import torch.distributed as dist
from mpi4py import MPI
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'blended_diffusion', 'guided_diffusion'))

from guided_diffusion import dist_util, logger  # noqa: E402
from guided_diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from guided_diffusion.resample import create_named_schedule_sampler  # noqa: E402
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults  # noqa: E402
from guided_diffusion.train_util import TrainLoop  # noqa: E402

# Must match dff_attack.py:270-288 (plus use_checkpoint, which only affects memory).
DVCE_MODEL_FLAGS = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    num_head_channels=64,
    attention_resolutions='32,16,8',
    resblock_updown=True,
    use_scale_shift_norm=True,
    learn_sigma=True,
    class_cond=False,
    diffusion_steps=1000,
    noise_schedule='linear',
)


def _patch_vb_terms():
    """The DVCE authors' p_mean_variance always passes middle_out=False to the
    model, but training_losses calls _vb_terms_bpd with a stand-in
    `lambda *args, r=frozen_out: r` that takes no keyword arguments, so
    training crashes. Drop the (no-op) flag before it reaches the model.
    Only used for training; the shared DVCE code is left unchanged."""
    original = GaussianDiffusion._vb_terms_bpd

    def _vb_terms_bpd(self, model, *args, **kwargs):
        def tolerant(*m_args, middle_out=False, **m_kwargs):
            assert not middle_out
            return model(*m_args, **m_kwargs)
        return original(self, tolerant, *args, **kwargs)

    GaussianDiffusion._vb_terms_bpd = _vb_terms_bpd


def _patch_checkpoint_loading():
    """guided-diffusion bug: TrainLoop loads the resume checkpoint (and EMA) only
    on rank 0, but dist_util.load_state_dict MPI-broadcasts the file and needs
    every rank to join, so with >1 GPU rank 0 waits forever and rank 1 waits in
    sync_params (deadlock). Load straight from disk instead; TrainLoop's
    sync_params still copies rank 0's weights to the other ranks."""
    dist_util.load_state_dict = lambda path, **kwargs: th.load(path, **kwargs)


def model_config(use_fp16, use_checkpoint):
    cfg = model_and_diffusion_defaults()
    cfg.update(DVCE_MODEL_FLAGS)
    cfg.update(use_fp16=use_fp16, use_checkpoint=use_checkpoint)
    return cfg


class ProgressTrainLoop(TrainLoop):
    """TrainLoop that also logs timing: every log_interval steps it adds
    lr / sec_per_step / elapsed_h / eta_h to progress.csv and writes one
    readable status line (step, % done, loss, speed, ETA) to log.txt/stdout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_start = None  # set after the first step (excludes startup and the step-0 save)
        self.steps_at_start = 0

    def log_step(self):
        super().log_step()
        done = self.step + self.resume_step
        total = self.lr_anneal_steps
        logger.logkv('lr', self.opt.param_groups[0]['lr'])
        if self.t_start is None:
            if self.step == 1:
                self.t_start, self.steps_at_start = time.time(), done
            return
        steps = done - self.steps_at_start
        elapsed = time.time() - self.t_start
        sec_per_step = elapsed / max(steps, 1)
        eta = sec_per_step * max(total - done, 0) if total else float('nan')
        logger.logkv('sec_per_step', sec_per_step)
        logger.logkv('elapsed_h', elapsed / 3600)
        logger.logkv('eta_h', eta / 3600)
        if self.step % self.log_interval == 0 and dist.get_rank() == 0:
            loss = logger.get_current().name2val.get('loss', float('nan'))
            now = datetime.now()
            finish = (now + timedelta(seconds=eta)).strftime('%a %H:%M') if total else '?'
            pct = f' ({100 * done / total:.1f}%)' if total else ''
            logger.log(f'[{now:%Y-%m-%d %H:%M}] step {done}/{total or "?"}{pct} | loss {loss:.4f} | '
                       f'{sec_per_step:.2f} s/step | elapsed {elapsed / 3600:.1f} h | '
                       f'ETA {eta / 3600:.1f} h (finishes ~{finish})')


class DiatlasDataset(Dataset):
    def __init__(self, data_root, splits, shard, num_shards):
        with open(os.path.join(data_root, 'splits.csv')) as f:
            paths = [os.path.join(data_root, r['path']) for r in csv.DictReader(f) if r['split'] in splits]
        self.paths = sorted(paths)[shard::num_shards]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.asarray(Image.open(self.paths[idx]).convert('RGB'))
        if np.random.rand() < 0.5:
            arr = arr[:, ::-1]
        if np.random.rand() < 0.5:
            arr = arr[::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        return np.ascontiguousarray(arr.transpose(2, 0, 1)), {}


def load_data(data_root, splits, batch_size, num_workers):
    dataset = DiatlasDataset(data_root, splits, MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
    logger.log(f'rank {MPI.COMM_WORLD.Get_rank()}: {len(dataset)} images')
    while True:
        yield from loader


def setup_dist():
    """Like guided_diffusion.dist_util.setup_dist, but keeps CUDA_VISIBLE_DEVICES and uses localhost."""
    comm = MPI.COMM_WORLD
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    visible = visible.split(',') if visible else [str(i) for i in range(th.cuda.device_count())]
    os.environ['CUDA_VISIBLE_DEVICES'] = visible[comm.rank % len(visible)]
    if comm.rank == 0:
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
    else:
        port = None
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(comm.bcast(port, root=0)),
                      RANK=str(comm.rank), WORLD_SIZE=str(comm.size))
    dist.init_process_group(backend='nccl', init_method='env://')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.path.join(REPO, 'data', 'diatlas'))
    p.add_argument('--splits', default='train,val')
    p.add_argument('--resume_checkpoint', default=os.path.join(REPO, 'checkpoints', '256x256_diffusion_uncond.pt'))
    p.add_argument('--batch_size', type=int, default=8, help='per GPU')
    p.add_argument('--microbatch', type=int, default=4, help='per GPU forward/backward chunk; -1 = batch_size')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--ema_rate', default='0.9995')
    p.add_argument('--total_steps', type=int, default=50000, help='training stops here (LR anneals linearly to 0)')
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=5000)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_fp16', action='store_true')
    p.add_argument('--no_checkpointing', action='store_true')
    args = p.parse_args()

    _patch_vb_terms()
    _patch_checkpoint_loading()
    setup_dist()
    logger.configure()  # directory from OPENAI_LOGDIR
    logger.log(f'args: {vars(args)}')

    model, diffusion = create_model_and_diffusion(**model_config(not args.no_fp16, not args.no_checkpointing))
    model.to(th.device('cuda'))
    data = load_data(args.data_root, args.splits.split(','), args.batch_size, args.num_workers)

    ProgressTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=not args.no_fp16,
        schedule_sampler=create_named_schedule_sampler('uniform', diffusion),
        lr_anneal_steps=args.total_steps,
    ).run_loop()
    if dist.get_rank() == 0:
        logger.log(f'[{datetime.now():%Y-%m-%d %H:%M}] training finished: step {args.total_steps}/{args.total_steps} '
                   f'(100%), final checkpoints saved')


if __name__ == '__main__':
    main()
