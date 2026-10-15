# Diatom DVCE: commands

> **Cleanup on 2026-10-15:** intermediate diffusion checkpoints (only step 30000
> kept), the stretched images of `data/diatlas/256/`, early test outputs and the
> old run logs were deleted to free ~106 GB. Commands below that write to
> `logs/` still work; results and final models are unchanged.

Every command used so far, in order, so you can rerun everything yourself.
Run all of them **from the repo root**:

```bash
cd ~/CS8813/DVCEs
conda activate python_38_dvces_cu111     # or call ~/miniconda3/envs/python_38_dvces_cu111/bin/python directly
```

Status markers: ✅ done and verified · ⏸ prepared, not run yet.

---

## 1. Preprocess the dataset ✅

Builds the 256×256 grayscale images, the train/val/test split, a report and a
montage. Takes about 1 minute on CPU. Details: `data/diatlas/README.md`.

```bash
python diatoms/prepare_diatlas.py
```

Outputs in `data/diatlas/`: `256/`, `splits.csv`, `split_report.txt`, `montage.png`.

To rebuild from scratch (e.g. after changing the split), delete the images first:

```bash
rm -rf data/diatlas/256 && python diatoms/prepare_diatlas.py
```

---

## 2. Check the training setup ✅ (CPU only)

Confirms the model loads the ImageNet checkpoint and the loader sees only
train+val images. Takes about 1 minute.

```bash
CUDA_VISIBLE_DEVICES= python diatoms/check_setup.py
```

Expected output:

```
checkpoint loads strictly: 552.8M params
train+val images: 32191, sample (3, 256, 256) in [-0.94, 0.95]
no test images in training set
```

---

## 3. Baseline: what the unmodified ImageNet model does to diatoms ✅

Takes 16 test images, noises them to DVCE's starting point (respacing 200,
skip 100), and denoises with the **original ImageNet model**. No classifier is
involved and nothing is trained. About 3 minutes on one GPU.

```bash
mkdir -p results/diffusion
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py \
    --ckpt checkpoints/256x256_diffusion_uncond.pt \
    --mode recon \
    --out results/diffusion/recon_imagenet.png
```

`recon_imagenet.png` has 8 columns. Rows 1 and 3 are originals; rows 2 and 4
are their reconstructions.

Close-up comparison and per-pair numbers:

```bash
python diatoms/compare_recon.py results/diffusion/recon_imagenet.png \
    --out results/diffusion/recon_imagenet_pairs.png \
    --pairs 6 9 13 14 15 3
```

`recon_imagenet_pairs.png` shows columns original | reconstruction |
difference. Leave out `--pairs` to show all 16. Change `--seed` (default 0) or
`--n` in `sample_diffusion.py` to try other test images; keep them the same to
compare models on identical inputs.

Useful options of `sample_diffusion.py`:

| Option | Meaning |
|---|---|
| `--mode recon` / `--mode sample` | reconstruct test images / generate new images from pure noise |
| `--skip_timesteps 100` | how much of the original survives (higher = closer to the original; DVCE default 100 of 200) |
| `--n 16 --batch_size 8` | number of images, images per GPU pass |
| `--seed 0` | which test images and which noise |

---

## 4. Trial training run ✅

50 training steps on one GPU to check memory and measure speed. About 5
minutes, about 2 of them spent writing checkpoints. The training loop always
saves at step 0 and at the last step (~17 GB in total), so delete them
afterwards.

```bash
GPUS=0 RUN=probe bash diatoms/train_diffusion.sh --total_steps 50 --save_interval 1000000 --log_interval 10 \
    > logs/probe.out 2>&1
rm -rf checkpoints/diatlas/probe
```

Look at: the loss in `checkpoints/diatlas/probe/progress.csv` (before deleting),
and GPU memory with `watch -n 2 nvidia-smi` in another terminal while it
runs. If it runs out of memory, add `--microbatch 2` (same effective batch,
less memory) or `--batch_size 4`.

---

## 5. Full diffusion fine-tuning ✅ (ft_v1: 2026-10-12 20:36 → 2026-10-13 15:31, 18.9 h)

Both GPUs. Per-GPU batch 8 (global 16), LR 2e-5 annealed to 0, EMA 0.9995,
30k steps (~18.5 h; measured 2.24 s/step on both GPUs), a checkpoint every 5k
steps (~3.5 h apart). No `conda activate` needed: the launcher calls the
`python_38_dvces_cu111` env directly.

**Before starting:**

```bash
cd ~/CS8813/DVCEs
nvidia-smi        # both GPUs should show almost no memory used (~0–100 MiB)
df -h .           # need ~62 GB free for checkpoints
```

**Start:**

```bash
GPUS=0,1 RUN=ft_v1 nohup bash diatoms/train_diffusion.sh --total_steps 30000 > logs/ft_v1.out 2>&1 &
```

- `GPUS=0,1` uses both GPUs; `RUN=ft_v1` names the run, so everything goes to
  `checkpoints/diatlas/ft_v1/`.
- `nohup … &` runs it in the background and keeps it going after you close
  the terminal or log out. All output goes to `logs/ft_v1.out`.

**What to expect, and when** (in `logs/ft_v1.out`):

| Time after start | What you see |
|---|---|
| ~0–1 min | `run=ft_v1 gpus=0,1 nproc=2 ...`, `loading model from checkpoint ...`, `rank 0: 16096 images` (each GPU gets half of the 32,191). `Authorization required, but no authorization protocol specified` is harmless. |
| ~1–3 min | `saving model 0...` / `saving model 0.9995...`: the training code always saves once at step 0 (~9 GB). Normal. |
| ~6–8 min | First status line (step 100), then one every ~4 min (see below). |
| every ~3.5 h | `saving model ...`: a checkpoint every 5,000 steps; training pauses ~2 min for each. |
| ~18.5 h | Last save at step 30000, then the process exits. |

A status line looks like:

```
[2026-10-12 21:05] step 1200/30000 (4.0%) | loss 0.0031 | 2.45 s/step | elapsed 0.8 h | ETA 19.6 h (finishes ~Tue 16:41)
```

Normal values while running:

- GPU memory ~22–24 GB on each GPU, utilization near 100% (`nvidia-smi`).
- Speed about 2.2–2.3 s/step (measured on the first run, ft_v1).
- Loss roughly 0.001–0.01, jumping around from line to line. Each batch uses
  random noise levels, so judge the trend in the plot's moving average.
  Starting from a trained model, expect a fairly flat curve drifting slowly
  down, not a big drop. The real quality check is section 6.

**Watch it:**

```bash
grep "ETA" logs/ft_v1.out | tail -3                                    # latest status lines
tail -f logs/ft_v1.out | grep --line-buffered "ETA\|saving\|Error"    # live (Ctrl+C stops watching, not training)

python diatoms/watch_training.py checkpoints/diatlas/ft_v1             # summary + checkpoints/diatlas/ft_v1/training_curve.png
watch -n 300 python diatoms/watch_training.py checkpoints/diatlas/ft_v1   # refresh every 5 min

watch -n 5 nvidia-smi                                                  # GPU use
```

**If it runs out of memory.** This appears within the first minute or two as
`CUDA out of memory` in the log, and the job exits. Rerun with a smaller
microbatch (same effective batch, less memory):

```bash
rm -rf checkpoints/diatlas/ft_v1
GPUS=0,1 RUN=ft_v1 nohup bash diatoms/train_diffusion.sh --total_steps 30000 --microbatch 2 > logs/ft_v1.out 2>&1 &
```

**Stop it:**

```bash
pkill -f train_diffusion.py
```

**Resume** after a crash or stop, from the last saved checkpoint (example:
step 20000). The optimizer and EMA files next to it are loaded automatically.
Keep `--total_steps 30000` the same so the learning-rate schedule continues
where it left off:

```bash
ls checkpoints/diatlas/ft_v1/          # find the latest modelNNNNNN.pt
GPUS=0,1 RUN=ft_v1 nohup bash diatoms/train_diffusion.sh --total_steps 30000 \
    --resume_checkpoint checkpoints/diatlas/ft_v1/model020000.pt > logs/ft_v1_resume.out 2>&1 &
```

**If it hangs at `loading model from checkpoint ...`** (GPUs at 0%, no status
line after ~5 min): that was a multi-GPU bug, fixed on 2026-10-12. If it ever
returns, stop it and check that `train_diffusion.py` still calls
`_patch_checkpoint_loading()`, and that `train_diffusion.sh` still sets
`NCCL_P2P_DISABLE=1` (these two GPUs can't copy directly to each other).

Other options: `--lr`, `--batch_size`, `--microbatch`, `--save_interval`,
`--ema_rate`. See `python diatoms/train_diffusion.py -h`.

---

## 6. Check a fine-tuned checkpoint ✅ (ft_v1, step 30000)

Use the **EMA** checkpoint (`ema_0.9995_NNNNNN.pt`); it gives better samples
than `modelNNNNNN.pt`. Example for the final step, 30000; for an intermediate
check use e.g. `010000`. Both GPUs are busy during training, so run this
after training finishes or is stopped.

```bash
CK=checkpoints/diatlas/ft_v1/ema_0.9995_030000.pt

# same 16 test images as the baseline -> direct before/after comparison
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py --ckpt $CK --mode recon \
    --out results/diffusion/recon_ft_v1_030000.png
python diatoms/compare_recon.py results/diffusion/recon_ft_v1_030000.png \
    --out results/diffusion/recon_ft_v1_030000_pairs.png --pairs 6 9 13 14 15 3 \
    | tee results/diffusion/recon_ft_v1_030000_metrics.txt

# brand-new diatoms from pure noise
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py --ckpt $CK --mode sample --n 32 --batch_size 16 \
    --save_individual --out results/diffusion/samples_ft_v1_030000.png

# the same from the original ImageNet model, for comparison (run on GPU 1 in parallel)
CUDA_VISIBLE_DEVICES=1 python diatoms/sample_diffusion.py --ckpt checkpoints/256x256_diffusion_uncond.pt \
    --mode sample --n 32 --batch_size 16 --save_individual --out results/diffusion/samples_imagenet.png
```

Metrics comparing both models on the same 16 reconstructions (CPU, ~1 min):

```bash
CUDA_VISIBLE_DEVICES= python diatoms/recon_metrics.py \
    --grid results/diffusion/recon_imagenet.png --name imagenet \
    --grid results/diffusion/recon_ft_v1_030000.png --name finetuned \
    --masks_out results/diffusion/recon_masks.png \
    | tee results/diffusion/recon_metrics_imagenet_vs_ft.txt
```

LPIPS (perceptual distance, lower = better), PSNR/SSIM over the whole image
and inside the diatom only (`m_psnr`, `m_ssim`; masks drawn in red in
`recon_masks.png`), plus a paired Wilcoxon test. Result: the fine-tuned model
is better on every metric (LPIPS 0.223 → 0.112, 16/16 pairs, p = 3e-5;
diatom-only SSIM 0.51 → 0.64).

`--save_individual` also writes each image separately, to
`results/diffusion/samples_ft_v1_030000/00.png … 31.png` and
`results/diffusion/samples_imagenet/00.png … 31.png`. The same seed gives the
same images on a rerun.

About 3 min for the reconstructions and 7 min for the 32 samples, on one GPU.
Compare `recon_ft_v1_030000*.png` with `recon_imagenet*.png` from section 3
(same 16 test images, same seed). Result: on average SSIM 0.84 → 0.88,
PSNR 24.7 → 26.0 dB, invented color 3.4 → 0.1; no diatom vanished or turned
into an object.

---

## 7. Independent classifier: load and check accuracy ✅ (CPU only, partial)

The independent classifier's species classifier (ResNet-18, PyTorch Lightning format),
copied from your laptop with:

```bash
# on your laptop
ssh morgoth "mkdir -p ~/CS8813/DVCEs/DiatlasModels/independent_resnet18"
scp ~/Downloads/"last-v4.ckpt 2" morgoth:~/CS8813/DVCEs/DiatlasModels/independent_resnet18/last-v4.ckpt
```

Measure its accuracy (CPU, 8 threads, ~20 s per 1,000 images). The default
preprocessing is his `PadCropDiatom` transform (`diatoms/padcrop_transform.py`,
from his Slack message), then grayscale to 3 channels and ImageNet mean/std:

```bash
CUDA_VISIBLE_DEVICES= python diatoms/eval_independent_classifier.py --n 1000                     # no physical rescaling (best)
CUDA_VISIBLE_DEVICES= python diatoms/eval_independent_classifier.py --n 1000 --scale_policy csv  # rescale with diatoms.csv scale
CUDA_VISIBLE_DEVICES= python diatoms/eval_independent_classifier.py --n 1000 --split train
CUDA_VISIBLE_DEVICES= python diatoms/eval_independent_classifier.py --n 1000 --preproc ours      # our 256x256 images
```

What we found (2026-10-13):

- ResNet-18, 759 outputs = `species_code` with ≥2 images, connective views
  dropped, alphabetical order. Its own validation accuracy: 0.861.
- `scale` in diatoms.csv = microns per 100 px, so src_scale = scale / 100.
  `tgt_scale` isn't in his message; 0.0826 µm/px (the dataset median) is best.
- Top-1 on 1,000 test images: our preprocessing 0.27 → his transform with csv
  scales 0.69 → his transform without rescaling **0.74** (top-5 0.78).
- The Bourgogne atlas (sources 2–4) does worst with the csv scales (~0.35);
  its best-fitting scale differs from the csv. Still below 0.861; the remaining
  questions for him are in the chat.

---

## 8. Fine-tune the independent classifier ✅ (diagnostic runs, 2026-10-13)

`diatoms/train_classifier.py` starts from the independent ResNet-18 (759 species) and
fine-tunes it. `--eps` = L2 radius of adversarial (PGD) training, 0 = normal;
`--preproc ours|padcrop`; images without `species_code` are skipped unless
`--keep_empty`. Train split for training, val for model selection, test untouched.
Each run takes ~6–10 min (eps 0) on one GPU.

```bash
# A: our stretch-to-fit images
CUDA_VISIBLE_DEVICES=0 nohup python diatoms/train_classifier.py --run resnet18_ft_eps0_noempty --eps 0 > logs/cls_A.out 2>&1 &
# B: his preprocessing
CUDA_VISIBLE_DEVICES=1 nohup python diatoms/train_classifier.py --run resnet18_ft_eps0_noempty_supprep --eps 0 --preproc padcrop > logs/cls_B.out 2>&1 &
tail -f DiatlasModels/resnet18_ft_eps0_noempty/log.txt      # one line per epoch
```

Results (val accuracy, before → after 15 epochs):

| Run | Result |
|---|---|
| `resnet18_ft_eps0` (ours, genus-only images kept) | 28.7% → 32.8% |
| `resnet18_ft_eps1.0` (ours, robust) | stopped at epoch 6: 16.9% clean, 10.3% robust |
| A `resnet18_ft_eps0_noempty` (ours) | 30.3% → 36.6% |
| B `resnet18_ft_eps0_noempty_supprep` (his preprocessing) | **84.8%** → 82.5% |

Conclusion: stretch-to-fit preprocessing loses the information (size) the
classifier needs → new preprocessing in section 9. Fine-tuning his already
good model at lr 0.01 slightly hurt it → use a lower lr next time.

---

## 9. Native-size preprocessing ✅

PadCropDiatom transform without rescaling: crop at 224, 256×256 canvas, same
split. Details and checks: `data/diatlas_nativesize_crop224_canvas256/README.md`.
About 35 s on CPU.

```bash
python diatoms/prepare_diatlas_nativesize.py
```

The classifier must get the **center 224 crop** of these images (identical to
his input: 84.2%), not a resize (70%) or the full 256 image (11%).

---

## 10. Robust fine-tuning of his classifier on native-size data ✅ (2026-10-13)

The independent ResNet-18, `data/diatlas_nativesize_crop224_canvas256`, classifier input =
center 224 crop, genus-only images skipped, 20 epochs, lr 0.01, PGD attacks in
fp32 (5 steps for training, 10 for evaluation on a fixed random set of 1,024
val images). ~40 min per run, one per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python diatoms/train_classifier.py --run resnet18_native_eps0.5 --eps 0.5 --epochs 20 > logs/cls_native_eps0.5.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python diatoms/train_classifier.py --run resnet18_native_eps1.0 --eps 1.0 --epochs 20 > logs/cls_native_eps1.0.out 2>&1 &
```

| Model | Clean val acc | Val acc under attack (eps 0.5 / 1.0) |
|---|---|---|
| his original | 84.8% | 19.8% / 2.3% |
| `resnet18_native_eps0.5/best.pt` | 79.5% | **60.9%** (eps 0.5) |
| `resnet18_native_eps1.0/best.pt` | 73.5% | **39.6%** (eps 1.0) |

Both plateaued from epoch ~15. The independent classifier probably saw many val images in his
own training, so the clean numbers are likely inflated; the robustness gains
are real.

Test split (4,214 images, never used in training here), same attacks for all
models (~5 min on one GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/eval_classifier.py --split test --ckpt independent \
    --ckpt DiatlasModels/resnet18_native_eps0.5/best.pt --ckpt DiatlasModels/resnet18_native_eps1.0/best.pt \
    --eps 0.5 1.0 | tee results/classifier/test_resnet18_native.txt
```

| Model | Top-1 | Top-5 | Under attack eps 0.5 | eps 1.0 |
|---|---|---|---|---|
| his original | 81.0% | 85.2% | 18.1% | 1.8% |
| robust eps 0.5 | 75.7% | 85.7% | 56.5% | 30.3% |
| robust eps 1.0 | 70.4% | 83.6% | 55.0% | 36.1% |

Attack-strength curve for the report (eps 0.25 / 0.5 / 1.0 / 2.0,
~10 min on one GPU) → `results/classifier/test_resnet18_native_curve.txt`,
page: `results/classifier/ROBUSTNESS.md` (the earlier HTML version was deleted in cleanup):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/eval_classifier.py --split test --ckpt independent \
    --ckpt DiatlasModels/resnet18_native_eps0.5/best.pt --ckpt DiatlasModels/resnet18_native_eps1.0/best.pt \
    --eps 0.25 0.5 1.0 2.0 | tee results/classifier/test_resnet18_native_curve.txt
```

---

## 11. Our own robust classifier on 759 species (robust ImageNet ResNet-50) ⛔ stopped (overfitting)

Starts from the robust ImageNet ResNet-50 the DVCE paper uses (Madry L2 + FT,
`ImageNet1000Models/Madry_l2_improved/checkpoint/ep_3.pt`), new 759-species
output layer, trained **only on our train split** (so test numbers are
honest). Same data/input as section 10, 30 epochs, ~5.2 min/epoch → ~2.6 h.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python diatoms/train_classifier.py --run r50madry_native_eps0.5 --init madry_r50 --eps 0.5 --epochs 30 > logs/cls_r50_eps0.5.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python diatoms/train_classifier.py --run r50madry_native_eps1.0 --init madry_r50 --eps 1.0 --epochs 30 > logs/cls_r50_eps1.0.out 2>&1 &
tail -f DiatlasModels/r50madry_native_eps0.5/log.txt
```

Stopped at epoch 5 (2026-10-13 18:50): training accuracy 85% (under attack)
but validation only ~31% on held-out pages, gap growing → section 13.

Afterwards (not run), test accuracy next to the ResNet-18 models:

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/eval_classifier.py --split test --ckpt independent \
    --ckpt DiatlasModels/resnet18_native_eps0.5/best.pt --ckpt DiatlasModels/resnet18_native_eps1.0/best.pt \
    --ckpt DiatlasModels/r50madry_native_eps0.5/best.pt --ckpt DiatlasModels/r50madry_native_eps1.0/best.pt \
    --eps 0.5 1.0 | tee results/classifier/test_all.txt
```

---

## 12. DVCEs on diatoms ✅ pipeline tested (CPU, 1 image); real runs: section 17

`diatlas_VCEs.py` = DVCE on the DIATLAS test set. With `--diatom_classifier`
it uses that classifier's 759 `species_code` classes and the native-size data
(`--diatlas_root`, default `data/diatlas_nativesize_crop224_canvas256`); the
classifier gets the center 224 crop. `--diffusion_ckpt` chooses the diffusion
model. Pairs: JSON `{"entries": [{"path": "256/test/...png", "targets": ["NACD", "NACI"]}]}`
(two targets per image), e.g. `evaluation/vce_pairs_paper50.json` (the 3-image test list used here was deleted in cleanup).

```bash
CUDA_VISIBLE_DEVICES=0 python diatlas_VCEs.py --vce_list evaluation/vce_pairs_paper50.json --num_imgs 6 \
    --diatom_classifier DiatlasModels/resnet18_native_eps0.5/best.pt \
    --diffusion_ckpt checkpoints/diatlas/ft_v1/ema_0.9995_030000.pt --denoise_dist_input
```

Output: `DiatlasVCEs/clf_<classifier>__diff_<diffusion run>/<settings>/`
(figures: original + 2 counterfactuals with target confidences, `single_images/`,
and a `.pth` with everything for `diatoms/eval_dvce.py`).

**CPU test (2026-10-13, ~25 min, no GPU; its pair file and outputs were deleted in cleanup):** 1 diatom × 2 targets, short
schedule (50 steps, skip 25):

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 python diatlas_VCEs.py --gpu cpu --vce_list evaluation/vce_pairs_cpu_test1.json \
    --num_imgs 2 --batch_size 2 --diatom_classifier DiatlasModels/resnet18_native_eps0.5/best.pt \
    --diffusion_ckpt checkpoints/diatlas/ft_v1/ema_0.9995_030000.pt --timestep_respacing 50 --skip_timesteps 25 \
    --denoise_dist_input --verbose > logs/dvce_cpu_test.out 2>&1
python diatoms/eval_dvce.py DiatlasVCEs/clf_resnet18_native_eps0.5__diff_ft_v1/cpu_test_50steps_skip25/2.pth \
    --eval_ckpt independent --out results/dvce/cpu_test1.csv
```

Result (NRCS → NCAT, NVEN): both targets reached (guiding classifier p 0.98,
original independent classifier agrees, p 0.81), LPIPS 0.068, change inside the
diatom 0.043 vs background 0.0005 (80× more on the diatom). Checked on CPU (diffusion step stubbed): data, labels and
classifier input match; classifier accuracy through this path 76.0% on 300
test images. Note: `ft_v1` was trained on stretch-to-fit images, so for real
results use the native-size diffusion model (to be trained).

---

## 13. Which classes can be recognised honestly? (label-set tests) ✅ (2026-10-13)

Normal training (eps 0), standard ImageNet ResNet-50, strong augmentation
(`--augment strong`: rotation ±10°, shift ±8 px, scale ±10%, brightness ±15%,
contrast ±25%, flips), 15 epochs, our page-grouped split, ~5–10 min per run.

```bash
# all 759 species
CUDA_VISIBLE_DEVICES=0 python diatoms/train_classifier.py --run r50imagenet_native_eps0_strongaug --init imagenet_r50 --eps 0 --augment strong --epochs 15
# A: species with >= 3 distinct training pages (92 species, 43 genera)
CUDA_VISIBLE_DEVICES=0 python diatoms/train_classifier.py --run r50imagenet_species92_eps0_strongaug --labels species_pages3 --init imagenet_r50 --eps 0 --augment strong --epochs 15
# B: genera with >= 100 training images (51 genera)
CUDA_VISIBLE_DEVICES=1 python diatoms/train_classifier.py --run r50imagenet_genus51_eps0_strongaug --labels genus_train100 --init imagenet_r50 --eps 0 --augment strong --epochs 15
```

| Label set | Classes | Train images | Best val accuracy (held-out pages) |
|---|---|---|---|
| all species (`independent759`) | 759 | 23,494 | 42.9% |
| **species, ≥3 training pages** (`species_pages3`) | **92** | 8,701 | **80.8%** |
| genus, ≥100 training images (`genus_train100`) | 51 | 23,717 | 92.1% |

Findings: honest species accuracy depends on how many independent atlas
pages a species has (1 page: 28%, 2–3: 47%, 4–7: 66% in the 759 model). The
fine-tuned independent classifier scores ~80% even on species with <10 training
images in our split → it was trained on our val/test pages, so its numbers are
inflated (likely a random image split on his side).

---

## 14. Robust ResNet-50 on a random image split ✅ (2026-10-13 19:53 → 21:52)

`--split_mode random`: every image goes to train/val/test (80/10/10) by a hash
of its path, so photos of one atlas page are spread over all splits, like the
independent classifier's (probable) setup. Test accuracy then measures recognition of
seen pages, not new pages: always report it as "random split", next to the
page-level numbers (section 13). Robust ImageNet ResNet-50 (Madry), 759
species, strong augmentation, 20 epochs, ~5.9 min/epoch → ~2 h.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python diatoms/train_classifier.py --run r50madry_random_eps0.5 --split_mode random --init madry_r50 --eps 0.5 --augment strong --epochs 20 > logs/cls_r50_random_eps0.5.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python diatoms/train_classifier.py --run r50madry_random_eps1.0 --split_mode random --init madry_r50 --eps 1.0 --augment strong --epochs 20 > logs/cls_r50_random_eps1.0.out 2>&1 &
tail -f DiatlasModels/r50madry_random_eps0.5/log.txt
```

Best validation: eps 0.5 (epoch 19) 91.9% clean / 85.9% under attack;
eps 1.0 (epoch 20) 90.0% / 81.2%.

Test evaluation on the same random split (add `--split_mode random`):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/eval_classifier.py --split test --split_mode random \
    --ckpt DiatlasModels/r50madry_random_eps0.5/best.pt --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt \
    --eps 0.25 0.5 1.0 2.0
```

Test results (random split, 3,411 images; `results/classifier/test_r50madry_random_curve.txt`):

| Model | Top-1 | Top-5 | eps 0.25 | eps 0.5 | eps 1.0 | eps 2.0 |
|---|---|---|---|---|---|---|
| independent classifier (original) | 83.3% | 86.5% | 59.7% | 18.6% | 1.7% | 0.1% |
| **r50madry_random_eps0.5** | **91.9%** | **98.7%** | **89.9%** | 86.5% | 76.8% | 45.9% |
| r50madry_random_eps1.0 | 90.9% | 98.7% | 88.9% | **87.0%** | **81.8%** | **62.1%** |

---

## 15. Does `ft_v1` work on native-size diatoms? ✅ No: it changes their size (2026-10-13)

Reconstruction check (noise to the DVCE start, denoise) on 32 native-size
test diatoms, ImageNet model vs `ft_v1`, plus a size check (diatom length
and area after / before, from the outline masks). Run automatically by
`diatoms/after_classifiers.sh`:

```bash
for ck in imagenet:checkpoints/256x256_diffusion_uncond.pt ft_v1:checkpoints/diatlas/ft_v1/ema_0.9995_030000.pt; do
  CUDA_VISIBLE_DEVICES=1 python diatoms/sample_diffusion.py --ckpt ${ck#*:} --mode recon --n 32 --batch_size 8 \
      --data_root data/diatlas_nativesize_crop224_canvas256 --out results/diffusion/native_recon_${ck%%:*}.png
done
python diatoms/recon_metrics.py --grid results/diffusion/native_recon_imagenet.png --name imagenet \
    --grid results/diffusion/native_recon_ft_v1.png --name ft_v1 --masks_out results/diffusion/native_recon_masks.png
```

| 32 native-size test diatoms | ImageNet | ft_v1 |
|---|---|---|
| LPIPS | 0.286 | 0.177 (better 28/32) |
| diatom-only SSIM | 0.475 | 0.550 |
| length after / before | 1.23 | **1.43** |
| length changed by >10% | 17/32 | **29/32** |

`ft_v1` keeps details better but **stretches small diatoms toward the frame
size** (it learned "diatoms fill the frame" from the stretch-to-fit data); see
`results/diffusion/native_recon_strip_orig_imagenet_ftv1.png` (rows:
original / ImageNet / ft_v1). Large diatoms are fine. → Retrain diffusion on
the native-size data.

---

## 16. Diffusion fine-tuning on native-size data ✅ (ft_v2_nativesize, 2026-10-13 22:12 → 2026-10-14 17:00)

Same settings as `ft_v1` (section 5), on `data/diatlas_nativesize_crop224_canvas256`
(train+val of the page split, 32,191 images). ~2.2 s/step → ~18.5 h, finishes
~Wed 16:35.

```bash
GPUS=0,1 RUN=ft_v2_nativesize nohup bash diatoms/train_diffusion.sh --total_steps 30000 \
    --data_root data/diatlas_nativesize_crop224_canvas256 > logs/ft_v2_nativesize.out 2>&1 &
grep "ETA" logs/ft_v2_nativesize.out | tail -3
python diatoms/watch_training.py checkpoints/diatlas/ft_v2_nativesize
```

Check on the same 32 native-size test diatoms as section 15 (~5 min, one GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py --ckpt checkpoints/diatlas/ft_v2_nativesize/ema_0.9995_030000.pt \
    --mode recon --n 32 --batch_size 8 --data_root data/diatlas_nativesize_crop224_canvas256 --out results/diffusion/native_recon_ft_v2.png
python diatoms/recon_metrics.py --grid results/diffusion/native_recon_imagenet.png --name imagenet \
    --grid results/diffusion/native_recon_ft_v1.png --name ft_v1 --grid results/diffusion/native_recon_ft_v2.png --name ft_v2 \
    > results/diffusion/native_recon_metrics_3models.txt
```

| 32 native-size test diatoms | ImageNet | ft_v1 | **ft_v2** |
|---|---|---|---|
| LPIPS (lower = better) | 0.286 | 0.177 | **0.103** (better than ImageNet on 31/32) |
| PSNR | 25.1 | 26.1 | **27.2** (32/32) |
| diatom-only SSIM | 0.475 | 0.550 | **0.599** |
| length after / before | 1.23 | 1.43 | **1.02** |
| length changed by >10% | 17/32 | 29/32 | **2/32** |

`ft_v2` keeps diatom size and shape and is the most faithful model on every
metric → the diffusion model for DVCE. Strip (rows: original / ImageNet /
ft_v1 / ft_v2): `results/diffusion/native_recon_strip_orig_imagenet_ftv1_ftv2.png`.

Same 16-image grid and close-up pairs as section 6 (same 16 test diatoms, seed 0,
here in their native-size version):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py --ckpt checkpoints/diatlas/ft_v2_nativesize/ema_0.9995_030000.pt \
    --mode recon --data_root data/diatlas_nativesize_crop224_canvas256 --out results/diffusion/recon_ft_v2_030000.png
python diatoms/compare_recon.py results/diffusion/recon_ft_v2_030000.png \
    --out results/diffusion/recon_ft_v2_030000_pairs.png --pairs 6 9 13 14 15 3 > results/diffusion/recon_ft_v2_030000_metrics.txt
```

Note: `recon_imagenet.png` / `recon_ft_v1_030000.png` (sections 3, 6) used the
**stretched** versions of these 16 diatoms, so they are not pixel-identical
inputs to `recon_ft_v2_030000.png`. Apples-to-apples ImageNet version on the
**native-size** inputs (originals verified pixel-identical):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py --ckpt checkpoints/256x256_diffusion_uncond.pt --mode recon \
    --data_root data/diatlas_nativesize_crop224_canvas256 --out results/diffusion/recon_imagenet_native.png
python diatoms/compare_recon.py results/diffusion/recon_imagenet_native.png \
    --out results/diffusion/recon_imagenet_native_pairs.png --pairs 6 9 13 14 15 3
python diatoms/recon_metrics.py --grid results/diffusion/recon_imagenet_native.png --name imagenet \
    --grid results/diffusion/recon_ft_v2_030000.png --name ft_v2 > results/diffusion/recon16_native_imagenet_vs_ftv2.txt
```

Same 16 inputs: LPIPS ImageNet 0.273 → ft_v2 **0.108** (16/16 better,
p = 3e-5); length ratio 1.17 → **1.01**.

Close-ups of 8 of the 32 diatoms, all three models side by side (columns:
original / ImageNet / ft_v1 / ft_v2), cut from the three grids above:
`results/diffusion/native_recon_closeups_3models.png`.

32 new diatoms from pure noise (grid + single images, ~7 min):

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/sample_diffusion.py --ckpt checkpoints/diatlas/ft_v2_nativesize/ema_0.9995_030000.pt \
    --mode sample --n 32 --batch_size 16 --save_individual --out results/diffusion/samples_ft_v2_nativesize.png
```

---

## 17. DVCE experiments ✅ (main + full run done)

**Pairs** (paper-style: targets = 2 random species of the same genus, the
taxonomy equivalent of the paper's "same WordNet cluster"; no A/B crossover):
50 test diatoms unseen by both models (test in both splits), classified
correctly by the robust ResNet-50 eps 1.0, 27 genera, all 14 atlases.

```bash
CUDA_VISIBLE_DEVICES=0 python diatoms/select_pairs_paper.py --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt \
    --out evaluation/vce_pairs_paper50.json
```

**One command per experiment:** DVCE on both GPUs (batch size 2: one diatom
with its two targets; 6 runs out of memory), then one figure per diatom and
the metrics:

```bash
NAME=main_r50eps1_ftv2 nohup bash diatoms/run_dvce_experiment.sh > logs/dvce_main.out 2>&1 &
```

Outputs in `results/dvce/<NAME>/`:
- `figures/NNN.png`, one per diatom:
  - row 1: original | counterfactual → target 1 | counterfactual → target 2
    (our classifier's p before → after, the independent classifier's verdict)
  - rows 2–4: 3 real training photos of each species, same scale
- `metrics.csv`, one line per counterfactual
- `summary.txt`: target reached, independent agreement, LPIPS, change on the
  diatom vs background
- `run_info.txt`

Pairs for the full run (all 301 suitable test diatoms, several per species allowed):
`python diatoms/select_pairs_paper.py --ckpt DiatlasModels/r50madry_random_eps1.0/best.pt --n 0 --all_per_species --out evaluation/vce_pairs_paper_all.json`

Figures show only the explained classifier (no other models' verdicts).
The transfer to independent classifiers is in `summary.txt` / `metrics.csv`:
- `independent_robust`: the independent classifier's ResNet-18 robust fine-tuned; different
  architecture, start and split, so the main transfer check
- `independent_nonrobust`: his original non-robust model, as a reference

**Results, 50 diatoms, default settings** (`results/dvce/main_r50eps1_ftv2/`, ~1 h):

| | |
|---|---|
| target reached (explained classifier) | 98% (median p 0.99) |
| LPIPS | 0.170 |
| change on the diatom vs background | 54× |
| transfer: independent_robust sees the target | 32% (p 0.009 → 0.218) |
| transfer: independent_nonrobust sees the target | 28% (p 0.020 → 0.275) |

Our own robust ResNet-50 eps 0.5 sees the target in 98%, but it is too
similar to the guide (same data, start, architecture) to count as independent.

**Settings test** (`diatoms/tune_dvce.sh`, 6 diatoms incl. 4 problem cases,
`results/dvce/tuning_target{1,2}.png`):

| | target reached | LPIPS | independent_robust sees target |
|---|---|---|---|
| default | 100% | 0.156 | — |
| skip 130 | 83% | 0.106 | 17% |
| skip 150 | 42% | 0.071 | 8% |
| guidance 0.05 | 75% | 0.094 | 17% |
| distance penalty 0.3 | 83% | 0.105 | 8% |

Gentler settings give smaller, cleaner edits but miss more targets and
transfer less → **kept the default (paper) settings**.

**Full run, 301 diatoms, default settings** (~6 h, started 2026-10-14 21:27):

```bash
NAME=full_r50eps1_ftv2 PAIRS=evaluation/vce_pairs_paper_all.json nohup bash diatoms/run_dvce_experiment.sh > logs/dvce_full.out 2>&1 &
```

Done 2026-10-15 03:54 (DVCE 6.2 h + figures/metrics 16 min), `results/dvce/full_r50eps1_ftv2/`:

| 602 counterfactuals, 301 diatoms | |
|---|---|
| target reached (explained classifier) | **93.9%** (p ≥ 0.9: 82%; both targets of a diatom: 90%) |
| p(target) before → after | 0.008 → 0.887 (median 0.993) |
| LPIPS | 0.171 (quartiles 0.13 / 0.17 / 0.20) |
| change on the diatom vs background | 52× |
| transfer: independent_robust sees the target | 26.2% (29.4% where it got the original right), p 0.010 → 0.167 |
| transfer: independent_nonrobust sees the target | 25.9% (30.0%), p 0.005 → 0.252 |
| both independent models see it / at least one | 21.6% / 30.6% |

`overview_random8.png`: top rows of 8 random figures.

---

## Where things are

| Path | What |
|---|---|
| `DIATOM_PLAN.md` | overall plan and status |
| `data/diatlas/README.md` | how the preprocessing works |
| `diatoms/prepare_diatlas.py` | preprocessing and split |
| `diatoms/check_setup.py` | pre-training checks |
| `diatoms/train_diffusion.py`, `.sh` | diffusion fine-tuning and its launcher |
| `diatoms/sample_diffusion.py` | reconstruction / sampling from any checkpoint |
| `diatoms/compare_recon.py` | close-up pairs and metrics |
| `diatoms/recon_metrics.py` | LPIPS + diatom-only PSNR/SSIM + Wilcoxon test across models |
| `diatoms/watch_training.py` | progress summary and loss plot of a running job |
| `diatoms/eval_independent_classifier.py` | load the independent classifier, measure accuracy |
| `diatoms/padcrop_transform.py` | the independent classifier's PadCropDiatom preprocessing |
| `diatoms/train_classifier.py` | fine-tune the independent ResNet-18, normal or robust (PGD) |
| `diatoms/eval_classifier.py` | test accuracy + accuracy under attack, several models side by side |
| `diatoms/diatom_classifier.py` | loads a diatom classifier for DVCE (center crop + normalisation inside) |
| `diatlas_VCEs.py` | DVCE on DIATLAS (`--diatom_classifier`, `--diffusion_ckpt`) |
| `diatoms/prepare_diatlas_nativesize.py` | native-size preprocessing (crop 224, canvas 256) |
| `data/diatlas_nativesize_crop224_canvas256/` | the native-size dataset + README |
| `DiatlasModels/independent_resnet18/` | the independent classifier's checkpoint (`last-v4.ckpt`) |
| `checkpoints/diatlas/<run>/` | training checkpoints and logs |
| `results/diffusion/` | images from the checks |
