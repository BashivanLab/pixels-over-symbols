<div align="center">

# Pixels over Symbols

### Sensory Realism Improves Behavioral Alignment in Models of Cognition

**Hao Yuan Bai, Xiaoxuan Lei, Zihan Weng, Motahareh Pourrahimi, Lucas Gomez, Pouya Bashivan**

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/virtual/2026/poster/153571)
[![arXiv](https://img.shields.io/badge/arXiv-2611.10565-b31b1b.svg)](https://arxiv.org/abs/2611.10565)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-pixels--over--symbols-ffd21e.svg)](https://huggingface.co/datasets/rdh1115/pixels-over-symbols)
[![Experiment](https://img.shields.io/badge/PsychoJS-experiment-2ea44f.svg)](https://github.com/rdh1115/abs_nat_psychopy_mturk)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

> **Do models of cognition need to see the world the way we do?**
> We trained the same networks on the same working-memory tasks twice: once on tidy symbolic descriptions of each image, once on the images themselves.
> The models that see images behave far more like people.

## TL;DR

Computational models of cognition usually replace real sensory input with clean symbolic vectors ("a car, on the left").
We test whether that shortcut matters.
**Abstract** models get a one-hot code of each stimulus, **Natural** models get the same tasks through a pretrained vision encoder, and both are compared trial by trial with **humans** and **vision-language models** on the same 960 trials.

## Key findings

- 🧠 **Images make models more human-like.** Natural models reproduce human trial-by-trial responses much more closely than Abstract ones, in all 18 task conditions, even though fully trained Abstract models are the more accurate ones.
- 🔁 **The recurrent weights are not the culprit.** An Abstract-trained core, frozen and driven by image features, recovers the human consistency of a Natural model, with dynamics within the seed-to-seed spread of Natural models.
- 🤖 **Strong VLMs are not the most human-like.** Vision-language models match or beat human accuracy, yet most task-trained Natural models reproduce the human pattern of responses better.
- 📏 **What matters, and what doesn't.** Sensory realism moves human alignment more than the choice of sequence architecture and about as much as the choice of vision encoder, while a bigger encoder barely helps.
- 🚧 **Room to grow.** Even the best model stays well below the consistency between humans.

The takeaway for neuroscience: conclusions drawn from abstract task models deserve a second look.

## What's in this repo

Everything needed to go from the tasks to the paper's figures:

- 🏋️ **Train** every model in the paper (Abstract, Natural, dimensionality-matched and core-substitution models) on 43 n-back and delayed match-to-sample tasks built from the HvM object images.
- 📊 **Evaluate** them on the human trials and measure behavioral consistency, RSA and error consistency.
- 🌀 **Compare their dynamics** with Dynamical Similarity Analysis (DSA).
- 🖼️ **Redraw** every data figure and statistic of the paper from the released data, in a few minutes on a CPU.

The data (de-identified human responses, VLM responses, trial lists, trained checkpoints and cached evaluations) is on Hugging Face at [rdh1115/pixels-over-symbols](https://huggingface.co/datasets/rdh1115/pixels-over-symbols), and the browser experiment we ran on Amazon Mechanical Turk is at [rdh1115/abs_nat_psychopy_mturk](https://github.com/rdh1115/abs_nat_psychopy_mturk).

## Quick start

Redraw the paper's figures:

```bash
conda create -n pixels python=3.12 && conda activate pixels
pip install -r requirements.txt
hf download rdh1115/pixels-over-symbols --repo-type dataset --local-dir data
bash scripts/figures/make_all.sh   # panels go to figures/
```

The sections below explain each step, and how to evaluate and train the models.

## Setup

The analyses were run with the package versions pinned in `requirements.txt` (Python 3.13 with numpy 1.26.4).
numpy 1.26.4 ships wheels up to Python 3.12, so a Python 3.12 environment is the simplest way to install the same versions:

```bash
conda create -n pixels python=3.12
conda activate pixels
pip install -r requirements.txt
```

The DSA fits its operators with the `DSA` package (`dsa-metric` on PyPI) and compares them with a batched version of that package's Procrustes solver in `analysis/dsa/simdist.py`.

## Data

Everything the code reads or writes lives under `data/` (see `utils/paths.py`); make it a symlink if it should live on another disk.
Download the release into it with

```bash
hf download rdh1115/pixels-over-symbols --repo-type dataset --local-dir data
```

which gives

```
data/mturk/trials/       the 12 MTurk trial lists and the 100 stimuli they use (stim_subset.csv)
data/mturk/images/       those 100 stimuli as shown to participants
data/mturk/human/        participant responses, 20 per task, one CSV per participant (s01.csv to s20.csv)
data/mturk/vlm/          VLM responses, 4 models in 5 configurations, 6 runs each
data/checkpoints/        trained models, one folder per run, with the training log
data/evaluations/        cached model, human and VLM responses on the MTurk trials, and the DSA results
data/features/           weights of the frozen random encoder (the encoder features are not included)
```

The dataset card on Hugging Face describes every file and how the human data was de-identified.
The VLM responses were collected through the OpenRouter API with the prompts described in the paper's appendix; the code that queried the models is not in this repository, and `mturk/vlm/` only reads their responses.

Redrawing the figures needs nothing else.
Training, evaluating models and the image-based analyses also need the full HvM image set and metadata (Majaj et al., 2015), which we do not redistribute (the release has only the 100 MTurk stimuli); they are available from the DiCarlo lab.
Put them here:

```
data/original/HvM_with_discfade/Variation00_20110203/*.png
data/original/HvM_with_discfade/Variation03_20110128/*.png
data/original/HvM_with_discfade/Variation06_20110131/*.png
data/original/python/hvmmeta_var0.csv, hvmmeta_var3.csv, hvmmeta_var6.csv
```

The train/validation split (4,271 and 1,389 images, with the 100 MTurk images left out) is derived from the metadata with a fixed seed and written to `data/original/HvM_with_discfade/no_mturk/` the first time it is needed.

## Figures

```bash
bash scripts/figures/make_all.sh
```

redraws every data figure of the paper from `data/evaluations` in a few minutes on a CPU.
The panels go to `figures/`, and the numbers the text quotes are printed in `figures/logs/`.
The RSA results depend on Python's hash seed, so the scripts insist on `PYTHONHASHSEED=0` (`make_all.sh` sets it).

<details>
<summary>Which file is which figure</summary>

| Figure | Files under `figures/` |
|---|---|
| 2 | `seq_type/{correlation,accuracy,legend}.png` |
| 3 | `transplant/final/global/{rho_mh,accuracy}.png`, `transplant/final/legend.png` |
| 4 | `dsa/final/{seq_abs,seq_res,legend}.png` |
| 10 | `metric_alignment/{rho_acc,rho_kappa}.png`, `seq_type/error_consistency.png` |
| 11 | `seq_type/RSA/human_baseline_{ct,cs}.png`, `metric_alignment/{rho_task_rsa,rho_stim_rsa}.png` |
| 12 | `metric_alignment/{cat,id,pos}_{cons,acc,ec}.png` |
| 13 | `ab_match_levels/{correlation,accuracy}.png`, `seq_type/legend.png` |
| 14 | `transplant/acc_matched/global/{rho_mh,accuracy}.png`, `transplant/acc_matched/legend.png` |
| 15 | `noise_dose/{rho_mh,accuracy}.png` |
| 16 | `transplant/dm/global/{rho_mh,accuracy}.png`, `transplant/dm/legend.png` |
| 17 | `vision_type/`, `seq_size/` and `encoder_size/`: `{correlation,accuracy}*.png` and `legend.png` |
| 18 | `seq_type/feature_level/{category,identity,position}/{correlation,accuracy}.png` |
| 19 | `seq_type/task_level/per_task_type_{NBack,InterDMS,CtxDMS,DMS}/{correlation,accuracy}.png` |
| 20 | `transplant/final/global/{cs,ct}_RSA.png` |
| 21 | `seq_type/RSA_ct_panels/*.png` |
| 22 | `dsa/final/{pre_abs,pre_res,legend}.png` |
| 23 | `dsa/dm/{pre_abs,pre_res,seq_abs,seq_res,legend}.png` |
| 24 | `dsa/final/{heatmap,mds}_{pre,seq}_pc50.png` |

Figure 1 and the stimulus and task illustrations (Figures 5 to 9) were made by hand.

</details>

## Statistics

```bash
export PYTHONHASHSEED=0
python scripts/stats/natural_vs_abstract.py   # Natural vs Abstract paired tests (Sec. 4.1)
python scripts/stats/rsa.py                   # RSA tests and the correlations between metrics (Sec. 4.1)
python scripts/stats/text_numbers.py          # validation accuracies and other numbers quoted in the text
python scripts/stats/discriminability.py      # visual discriminability control (appendix; needs HvM and features)
python scripts/stats/separability.py          # linear separability of the encoder features (needs features)
```

The separability values behind Figure 12 (`data/evaluations/vision_sgd.npy`) were computed from an earlier version of the encoder features; `separability.py` recomputes them from the feature caches into `vision_sgd_recomputed.npy`, within 0.03 of the paper's values (its docstring explains the difference).

## Evaluating models

`scripts/evaluate.py` runs the checkpoints in `data/checkpoints` on the MTurk trials and caches their responses in `data/evaluations`, one group of models at a time:

```bash
python scripts/evaluate.py --list               # the groups and their models
python scripts/evaluate.py abstract natural     # evaluate two groups
```

Models whose responses are already cached are skipped, so delete their files in `data/evaluations` to recompute them.
Natural models never run their vision encoder: they read its features from `data/features`, so build those first with `scripts/train/features.sh` (this needs the HvM images).
Evaluated on a CPU, as in the paper and by default here, the released checkpoints reproduce the cached responses exactly.
The one exception is the random-initialization twins of the dimensionality-matched models, whose confidences can change in the fifth decimal with the number of CPU threads (the cache was computed with 64).

## DSA

The DSA compares the hidden states of the native and substituted models on the MTurk trials (Sec. 4.3):

```bash
python scripts/evaluate.py substitution --features   # cache the hidden states
python scripts/dsa/run_dsa.py                        # distances behind Figures 4, 22, 23 and 24
python scripts/dsa/sensitivity.py                    # the 19 settings of the sensitivity analysis
python scripts/dsa/sensitivity_report.py             # its table and the counts in the appendix
python scripts/dsa/whitening.py                      # the check behind the sentence on PCA whitening
```

The hidden states of all 26 DSA systems are in the release, so the first command only fills in what is missing.
PCA and the operator fits run on multithreaded BLAS, and the alignment of every pair starts from a random initialization drawn in pair order.
The distances are therefore bitwise reproducible only with the same thread counts (the scripts set 16 and 12) and the same list of systems (`analysis/dsa/systems.py`), which is why that list keeps four ViT systems no figure shows.

## Training

Each family of models has a launcher in `scripts/train/` that takes the seed (0 to 5) and the family's own choices, passes the paper's training settings explicitly, and writes the run to the folder of `data/checkpoints` the analysis code reads.
The header of each script lists its arguments.

| Script | Models |
|---|---|
| `features.sh` | frozen-encoder features of every HvM image, for each of the 15 encoders and the random encoder |
| `abstract.sh` | one-hot Abstract models: 5 sequence cores, hidden-size sweep, and the reruns that stop at 0.80 validation accuracy |
| `natural.sh` | Natural models: 5 cores, 15 encoders, hidden-size sweep, frozen random encoder |
| `dim_match.sh` | dimensionality-matched Abstract LSTM at noise level s |
| `abs_to_nat.sh` | Abs->Nat substitutions: a Natural model around a frozen Abstract core |
| `nat_to_abs.sh` | Nat->Abs substitutions: an Abstract model around a frozen Natural core |

Run them in that order: Natural models need the encoder features, and substitutions need their donor models.
For example, the 0.80-matched one-hot LSTM of seed 0 and its Abs->Nat substitution with a frozen core and output head:

```bash
scripts/train/abstract.sh lstm 128 0 until080
CUDA_VISIBLE_DEVICES=0 scripts/train/abs_to_nat.sh m80 wout 0
```

<details>
<summary>Provenance of the 0.80-matched checkpoints</summary>

Not every 0.80-matched checkpoint comes from an `until080` rerun.
A seed was rerun only when its original run had no epoch-0 checkpoint labelled 0.80; otherwise the paper used that checkpoint of the original run.
These 23 checkpoints (19 in `abstract_until080`, 4 in `nat_to_abs_until080`) keep the two-decimal name of the original run (`RNN_model_acc0.80_epoch_0_batch_<N>.pth`) and come with its training log, while the rerun checkpoints have four decimals (`acc0.8012`).
To retrain one of them, use the launcher's regular mode (`abstract.sh` without `until080`, `nat_to_abs.sh` with recipient `onehot`) and take the epoch-0 checkpoint labelled 0.80: `until080` checks the validation accuracy more often near 0.80 and can select a different checkpoint, which for LSTM seeds 3 and 4 would also change the `m80` donor of `abs_to_nat.sh`.

</details>

Training is only bitwise reproducible on the same hardware and software.
Most Abstract, dimensionality-matched and Nat->Abs models of the paper were trained on CPUs and the rest on NVIDIA A5000, A100 and H100 GPUs; the training log in each run folder records its settings.
The ResNet-CLIP Natural LSTM and GRU (all seeds) and Transformer (seeds 0 to 4) were trained from raw images with an earlier version of this code, while `natural.sh` trains from the feature cache.
The ImageNet-21k, DINO and DINOv3 ViTs receive the CLIP image normalization although their timm configurations list other statistics, as in the paper's runs.
Weights & Biases logging is off unless a trainer gets `--wandb`.

## Model names

The analysis code refers to every model by a key (`analysis/factory.py`, `analysis/transplant_configs.py`).
`AB_LSTM_NOMTURK_U80` is the one-hot Abstract LSTM at its checkpoint closest to 0.80 validation accuracy; `_FINAL` marks the final checkpoint, and no suffix the epoch-0 checkpoint closest to 0.75.
`NOMTURK` means trained on the split without the MTurk images, as every model in the paper is.
Natural keys name the encoder and the core (`ResNet_CLIP_LSTM_NOMTURK`), `DIM_MATCH_LSTM_*` are the dimensionality-matched models, and `REG_XFER_*` and `DM_XFER_*` are the substitutions with one-hot and dimensionality-matched Abstract models (`ABS` or `RES`: the recipient is Abstract or Natural; `THETA`, `THETA_WOUT`, `THETA_WIN`, `UNTRAINED`: the frozen modules).

## Code layout

| Path | Contents |
|---|---|
| `task/` | the 43 tasks and on-the-fly trial sampling from the HvM metadata |
| `models/` | Abstract (`CustomRNN`), Natural (`CNNRNN`) and dimensionality-matched models, sequence cores, vision encoders |
| `train_*.py`, `training_loop.py` | training |
| `cache_image_embed.py` | frozen-encoder features |
| `mturk/` | MTurk task definitions, participant and VLM responses |
| `analysis/` | model registry, evaluation on the MTurk trials, behavioral metrics, DSA |
| `scripts/` | launchers, evaluation, figures, statistics, MTurk stimuli and trial lists |
| `test/` | unit tests (`python -m unittest discover -s test`) |

## Citation

```bibtex
@inproceedings{bai2026pixels,
  title     = {Pixels over Symbols: Sensory Realism Improves Behavioral Alignment in Models of Cognition},
  author    = {Bai, Hao Yuan and Lei, Xiaoxuan and Weng, Zihan and Pourrahimi, Motahareh and Gomez, Lucas and Bashivan, Pouya},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the Apache 2.0 license (`LICENSE`).
The data on Hugging Face is released under CC BY 4.0, except the HvM images, which keep their original terms.
