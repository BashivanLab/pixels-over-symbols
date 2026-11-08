"""Cache the frozen-backbone features of every HvM image.

The Natural trainers read these caches instead of running the backbone
(:func:`task.dataloader.prepare_natural_dataloaders`). A cache holds the
``forward_features`` output of all images in metadata order (the row order of
``python/hvmmeta_var*.csv``), stored as float16, plus the output for one blank
(all-zero) frame. It is written under the train and val file names the loaders
expect, ``<cnn_type>_{train,val}_features.pt`` and ``<cnn_type>_{train,val}_blank.pt``;
both pairs hold the same tensors.

Usage:
    python cache_image_embed.py --cnn_type resnet50_clip.openai
    python cache_image_embed.py --cnn_type resnet50_clip.openai --no_pretrained --seed 2025
"""

import argparse
from pathlib import Path

import torch

from models.vision_model import load_vision_backbone, get_pretrained_stats
from utils.helper import seed_everything
from utils.paths import FEATURE_DIR, HVM_DIR
from utils.stim_io import HvMMetaData, pil_read_fn


@torch.no_grad()
def extract_features(model, images, batch_size=32):
    """``forward_features`` of each image and of one blank frame, as float16 on the CPU."""
    device = next(model.parameters()).device
    # fp16 autocast on the GPU (a no-op on the CPU). cache_enabled=False keeps no fp16
    # copy of the weights for the whole block, so EVA02-enormous fits on a 24 GB GPU.
    with torch.autocast('cuda', enabled=device.type == 'cuda', cache_enabled=False):
        features = [model.forward_features(images[i: i + batch_size].to(device)).cpu().half()
                    for i in range(0, len(images), batch_size)]
        blank = model.forward_features(torch.zeros(1, *images.shape[1:], device=device)).cpu().half()
    return torch.cat(features, dim=0), blank


def main():
    parser = argparse.ArgumentParser(description="Cache vision-backbone features of the HvM images.")
    parser.add_argument('--cnn_type', type=str, required=True,
                        help="timm name of the backbone, e.g. resnet50_clip.openai")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="HvM folder holding python/ and HvM_with_discfade/ (default: data/original)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Inference batch size for feature extraction")
    parser.add_argument('--no_pretrained', action='store_true', default=False,
                        help=("Cache features of a randomly initialized backbone (the random-encoder "
                              "model). The draw is seeded with --seed and its state_dict is written "
                              "next to the cache, so training and evaluation load the same encoder."))
    parser.add_argument('--seed', type=int, default=2025,
                        help="Seed for the random backbone draw when --no_pretrained is set")
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="Where to write the caches (default: data/features, or "
                             "data/features/randinit_seed<seed> with --no_pretrained)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running on device: {device} ---")
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
    elif args.no_pretrained:
        cache_dir = FEATURE_DIR / f'randinit_seed{args.seed}'
    else:
        cache_dir = FEATURE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading vision backbone: {args.cnn_type} "
          f"({'random init, seed ' + str(args.seed) if args.no_pretrained else 'pretrained'}) ---")
    if args.no_pretrained:
        seed_everything(args.seed)
    model = load_vision_backbone(args.cnn_type, pretrained=not args.no_pretrained)
    if args.no_pretrained:
        # The training and evaluation code load this file rather than re-drawing the
        # encoder from the seed, so the encoder in a checkpoint is the cached one.
        backbone_path = cache_dir / f"{args.cnn_type}_randinit_seed{args.seed}.pth"
        torch.save(model.state_dict(), backbone_path)
        print(f"--- Saved random backbone state_dict to {backbone_path} ---")
    model = model.to(device)

    # The caches are indexed by row of the full metadata (HvMMetaData.filename_to_idx),
    # so every image listed there has to be present.
    meta = HvMMetaData(args.data_dir or HVM_DIR)
    if len(meta.df) != len(meta.filename_to_idx):
        raise FileNotFoundError(
            f'{len(meta.filename_to_idx) - len(meta.df)} HvM images listed in the metadata are missing')
    images = meta.build_image_cache(pil_read_fn(dataset_stats=get_pretrained_stats(args.cnn_type)))
    print(f"--- Extracting features of {len(images)} images ---")
    features, blank = extract_features(model, images, args.batch_size)

    for split in ('train', 'val'):
        torch.save(features, cache_dir / f"{args.cnn_type}_{split}_features.pt")
        torch.save(blank, cache_dir / f"{args.cnn_type}_{split}_blank.pt")
    print(f"--- Saved {tuple(features.shape)} features to {cache_dir} ---")


if __name__ == "__main__":
    main()
