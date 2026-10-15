# Plan: DVCEs for the DIATLAS diatom dataset

Version 2, 2026-10-13. Replaces the first plan (kept in
`archive/DIATOM_PLAN_v1_2026-10-12.md`). Commands for every step:
`diatoms/COMMANDS.md`.

## Goal

Generate Diffusion Visual Counterfactual Explanations (DVCEs) for diatoms:
start from a real diatom of species A and change it as little as possible,
with a diffusion model, until a robust classifier predicts species B. The
change shows what the classifier thinks separates A from B.

DVCE needs three pieces, all for the same kind of image:

1. a **diffusion model** that knows what diatoms look like
2. an **adversarially robust classifier**, so only visible changes can flip it
3. the **pipeline** connecting both to the diatom data

## Current approach

| Piece | Choice | Why |
|---|---|---|
| **Data** | `data/diatlas_nativesize_crop224_canvas256`: the PadCropDiatom transform, native pixel size, crop 224, 256×256 canvas, seamless padding; connective views dropped | keeps **size**, a species trait; matches the independent classifier |
| **Classifier input** | center 224×224 crop of the 256 image | pixel-identical to the independent classifier's input (84% vs 11% when feeding 256) |
| **Classes** | 759 `species_code` classes (as the independent classifier) | keeps the setup comparable to the independent classifier |
| **Classifier** | robust ResNet-50 started from the robust ImageNet model of the DVCE paper (Madry), adversarial training eps 0.5 / 1.0 | robust start + robust training; a starting model alone loses its robustness when fine-tuned without attacks |
| **Classifier split** | random image split 80/10/10 (`splits_random.csv`) | simple, comparable to the independent classifier; report as "random split" |
| **Diffusion model** | OpenAI 256×256 unconditional model, fine-tuned on the native-size data | fine-tuning is clearly better than the ImageNet model |
| **Pipeline** | `diatlas_VCEs.py` with `--diatom_classifier` and `--diffusion_ckpt` | ImageNet registry bypassed; labels from the classifier |

## Status

| Step | Status |
|---|---|
| First preprocessing (stretch-to-fit, `data/diatlas`) | ✅ done, **superseded** (erases size) |
| Diffusion fine-tuning on stretch-to-fit data (`ft_v1`) | ✅ done: LPIPS 0.223 → 0.112 vs ImageNet model, 16/16 better. Must be redone on native-size data |
| Native-size preprocessing | ✅ done |
| Independent classifier reproduced (transform, classes, scale) | ✅ done |
| The independent classifier robust fine-tuned (`resnet18_native_eps0.5`) | ✅ done: test 75.7% clean, 56.5% under eps-0.5 attack (his original: 81.0% / 18.1%) |
| Honest page-level evaluation | ✅ done: 759 species 43%, 92 species with ≥3 training pages 81%, 51 genera 92% |
| Our robust ResNet-50, random split (`r50madry_random_eps0.5/1.0`) | ✅ done: test 91.9% / 90.9% clean, 86.5% / 87.0% under eps-0.5 attack → **eps 1.0 chosen** for DVCE |
| DVCE pipeline with diatom classifiers | ✅ built and tested on CPU: both targets reached, change 80× more on the diatom than the background |
| `ft_v1` on native-size diatoms | ✅ checked: stretches small diatoms (length ×1.43, 29/32 changed) → retraining needed |
| Pair selection script, DVCE evaluation script | ✅ ready (`diatoms/select_pairs.py`, `diatoms/eval_dvce.py`) |
| Diffusion retraining on native-size data (`ft_v2_nativesize`) | ✅ done 2026-10-14 17:00: LPIPS 0.103 (ImageNet 0.286, ft_v1 0.177), diatom length kept (×1.02, 2/32 changed >10%) |
| DVCE experiments and evaluation | ✅ done: 301 test diatoms, 602 counterfactuals; target reached 93.9%, LPIPS 0.171, 52× more change on the diatom than the background, ~26% transfer to independent classifiers |

## What we learned (and why the plan changed)

1. **Stretch-to-fit was a mistake.** Stretching every diatom to fill the image
   removes size and distorts stria spacing. Classifiers could not learn
   species from it (~35%). We should have tested a classifier on the data
   before the 19 h diffusion run.
2. **The PadCropDiatom preprocessing matters exactly.** Native pixel size, crop
   at 224, ImageNet normalisation. With the DIATLAS `scale` values,
   rescaling makes it worse (Bourgogne atlas).
3. **Atlas pages leak.** Photos on one atlas page share background, contrast
   and specimen batch. A random split puts photos of one page in train and
   test, so accuracy measures recognition of seen pages:
   - page-level split: ~43% on 759 species
   - random split: ~80%
   - the independent classifier likely saw our test pages
   - DVCE can reveal whether a classifier uses page style (changes in the
     background instead of the diatom).
4. **Robustness needs adversarial training.** Fine-tuning from the robust
   ImageNet model without attacks keeps almost none of it (10.5% vs 4% under
   eps 0.5).
5. **Technical fixes found on the way:**
   - two guided-diffusion bugs (training crash, multi-GPU deadlock)
   - no direct GPU-to-GPU access on this machine (`NCCL_P2P_DISABLE`)
   - fp16 attacks too weak (use fp32)
   - biased robust-evaluation sample

## Next steps

Done: classifier choice (eps 1.0), diffusion retraining (`ft_v2_nativesize`)
and its check, pair selection (same genus, `select_pairs_paper.py`), the full
DVCE run and its evaluation. Remaining:

1. **Comparison runs** on the same 50 diatoms:
   - ImageNet diffusion model instead of `ft_v2`
   - the non-robust independent classifier as the explained classifier
     (expected: noise-like changes)
2. **Cropped vs uncropped diatoms:** check with numbers whether failures
   concentrate on diatoms cropped at 224 px.
3. **If counterfactuals turn out to rely on page style:** retrain the classifier
   with the page-level split, or the 92-species / genus label set
   (`--split_mode page --labels species_pages3`).
4. **Expert judgement** of a set of counterfactuals by a diatomist.
5. **Write-up:** technical report with the figures and tables.

## Open questions

- Class set: all 759 species, the 92 well-documented species, or genus level?
- Did he use a random image split? Is the page-level split the fairer evaluation?
- Is cropping long diatoms (46% lose their ends) acceptable?
- Could he, or a diatomist, judge a set of counterfactuals biologically?
- Target venue and deadline (CVMI / BioImage Computing workshop, or ISBI).

## Where things are

| Path | What |
|---|---|
| `diatoms/COMMANDS.md` | every command, step by step, with results |
| `data/diatlas_nativesize_crop224_canvas256/README.md` | the dataset in use, both splits |
| `DiatlasModels/README.md` | the classifiers in use |
| `results/classifier/ROBUSTNESS.md` | independent vs robust classifiers, table + plot |
| `results/diffusion/` | reconstruction comparisons, samples, metrics |
| `checkpoints/diatlas/ft_v1/` | first diffusion model (stretch-to-fit data) |
| `evaluation/vce_pairs_*.json` | DVCE pair lists |
