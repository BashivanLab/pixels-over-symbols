# Training

Commands run from the repository root; set up the environment and download the data first (see the [main README](../../README.md)).

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

## The 0.80-matched checkpoints

Not every 0.80-matched checkpoint comes from an `until080` rerun.
A seed was rerun only when its original run had no epoch-0 checkpoint labelled 0.80; otherwise the paper used that checkpoint of the original run.
These 23 checkpoints (19 in `abstract_until080`, 4 in `nat_to_abs_until080`) keep the two-decimal name of the original run (`RNN_model_acc0.80_epoch_0_batch_<N>.pth`) and come with its training log, while the rerun checkpoints have four decimals (`acc0.8012`).
To retrain one of them, use the launcher's regular mode (`abstract.sh` without `until080`, `nat_to_abs.sh` with recipient `onehot`) and take the epoch-0 checkpoint labelled 0.80: `until080` checks the validation accuracy more often near 0.80 and can select a different checkpoint, which for LSTM seeds 3 and 4 would also change the `m80` donor of `abs_to_nat.sh`.

## Reproducibility

Training is only bitwise reproducible on the same hardware and software.
Most Abstract, dimensionality-matched and Nat->Abs models of the paper were trained on CPUs and the rest on NVIDIA A5000, A100 and H100 GPUs; the training log in each run folder records its settings.
The ResNet-CLIP Natural LSTM and GRU (all seeds) and Transformer (seeds 0 to 4) were trained from raw images with an earlier version of this code, while `natural.sh` trains from the feature cache.
The ImageNet-21k, DINO and DINOv3 ViTs receive the CLIP image normalization although their timm configurations list other statistics, as in the paper's runs.
Weights & Biases logging is off unless a trainer gets `--wandb`.
