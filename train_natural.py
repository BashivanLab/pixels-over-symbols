"""Training entry point for the Natural model (frozen vision backbone + sequence core).

Trains a :class:`CNNRNN` on the HvM images via :func:`training_loop.train_model`,
reading the backbone features from the cache written by ``cache_image_embed.py``
when it exists (and running the frozen backbone on the images otherwise).
"""
import os
import time
from pprint import pprint

import torch
import torchvision

import wandb
from models.model import CNNRNN
from models.vision_model import get_pretrained_stats
from task.dataloader import prepare_natural_dataloaders
from training_loop import train_model
from utils.argparse import TrainArgParser
from utils.const import TASK_NAME_TASK_INDEX, OUTPUT_SIZE
from utils.helper import is_compile_supported, seed_everything


def train_natural(
        # dataset args
        batch_size: int = 128,
        train_dataset_size: int = 4096,
        val_dataset_size: int = 1024,
        num_workers: int = 6,
        pin_mem: bool = True,
        data_dir: str = None,
        train_vars=(0, 3, 6),
        val_vars=(0, 3, 6),
        # training args
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_norm: float = None,
        # model args
        cnn_type: str = 'resnet50',
        seq_model_type: str = 'GRU',
        num_seq_layers: int = 1,
        hidden_size: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
        vision_state_dict_path: str = None,
        emb_cache_dir: str = None,
        # Transformer-specific args
        num_heads: int = 8,
        # configs
        rep_index: int = 0,
        results_dir: str = './results',
        print_freq: int = 5,
        use_wandb: bool = False,
        **kwargs,
):
    """Train a :class:`CNNRNN` (frozen vision backbone + sequence core) on the naturalistic task.

    Seeds from ``2025 + rep_index``, builds naturalistic dataloaders with the
    backbone's normalization stats, constructs the model, ``torch.compile``s it
    when supported, and runs :func:`training_loop.train_model`, logging to wandb
    when ``use_wandb``. Extra keyword args are ignored so it can be called
    directly from parsed CLI args.
    """
    tasks = 'all'
    print(f"[Start] Seq Model: {seq_model_type}, Hidden: {hidden_size}, Rep: {rep_index}, Tasks: {tasks}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    seed_everything(2025 + rep_index)

    run = wandb.init(
        project="RNN-NatAbs",
        config={
            "tasks": tasks,
            "batch_size": batch_size,
            "train_dataset_size": train_dataset_size,
            "val_dataset_size": val_dataset_size,
            "num_epochs": num_epochs,
            "warmup_epochs": warmup_epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "hidden_size": hidden_size,
            "seq_model_type": seq_model_type,
            "num_seq_layers": num_seq_layers,
            "rep_index": rep_index,
            "natural_stim": True,
            "model": f"CNN+{seq_model_type.upper()}",
        },
        name=f"natural_{seq_model_type}_hidden{hidden_size}_tasks_{tasks}_nomturk",
        mode=None if use_wandb else 'disabled',
    )

    # Determine normalization stats based on CNN type
    mean, std = get_pretrained_stats(cnn_type)
    # Prepare datasets and loaders
    train_loader, val_loader, _, _ = prepare_natural_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size,
        val_dataset_size=val_dataset_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        train_vars=train_vars,
        val_vars=val_vars,
        pretrained_stats=(mean, std),
        cnn_type=cnn_type,
        emb_cache_dir=emb_cache_dir,
    )

    model = CNNRNN(
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        label_size=len(TASK_NAME_TASK_INDEX),
        cnn_type=cnn_type,
        seq_model_type=seq_model_type,
        num_seq_layers=num_seq_layers,
        dropout=dropout,
        pretrained=pretrained,
        vision_state_dict_path=vision_state_dict_path,
        # Transformer args
        num_heads=num_heads,
    ).to(device)

    if is_compile_supported():
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[Warning] torch.compile() failed: {e}")
            print("[Info] Falling back to eager mode.")

    # EVA02-enormous checkpoints hold the 17 GB backbone, so only the final one is kept
    save_final_only = cnn_type == 'eva02_enormous_patch14_clip_224.laion2b_plus'
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rep_index=rep_index,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        grad_norm=grad_norm,
        save_folder=results_dir,
        device=device,
        print_freq=print_freq,
        natural_stim=True,
        save_final_only=save_final_only,
        run=run
    )
    run.finish()
    print(f"[Complete] Seq: {seq_model_type}, Hidden: {hidden_size}, Rep: {rep_index}, Tasks: {tasks}")


if __name__ == "__main__":
    start_time = time.time()
    torchvision.disable_beta_transforms_warning()
    torch.set_float32_matmul_precision('high')

    parser = TrainArgParser()

    parser.add_argument(
        '--cnn_type',
        type=str,
        default='resnet50',
        help="timm name of the vision backbone, e.g. resnet50_clip.openai"
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        default=False,
        help=("Random-encoder control: build the vision backbone with random weights "
              "instead of pretrained ones, still frozen. Pair with --vision_state_dict_path "
              "and --emb_cache_dir pointing at the draw that cache_image_embed.py "
              "--no_pretrained produced. Every rep should point at the same draw, so the "
              "model shares one frozen encoder exactly as the pretrained models do and only "
              "the sequence model varies across seeds.")
    )
    parser.add_argument(
        '--vision_state_dict_path',
        type=str,
        default=None,
        help=("Explicit backbone state_dict to load (the "
              "<cnn_type>_randinit_seed<S>.pth written by cache_image_embed.py). "
              "Required with --no_pretrained so the encoder saved in this checkpoint is "
              "the same one the feature cache was generated from.")
    )
    parser.add_argument(
        '--emb_cache_dir',
        type=str,
        default=None,
        help="Feature-cache directory (default: data/features). Point at the random-encoder "
             "cache (data/features/randinit_seed2025) for the random-encoder model."
    )

    args = parser.parse_args()
    args.pretrained = not args.no_pretrained
    if args.no_pretrained and args.vision_state_dict_path is None:
        parser.error(
            "--no_pretrained requires --vision_state_dict_path. Without it the backbone is "
            "re-drawn from whatever RNG state training happens to be in, which will not match "
            "the encoder your feature cache was built from, and nothing will report an error."
        )

    os.makedirs(args.results_dir, exist_ok=True)
    pprint(vars(args))

    train_natural(**vars(args))

    total_secs = time.time() - start_time
    hours = int(total_secs // 3600)
    minutes = int((total_secs % 3600) // 60)
    seconds = int(total_secs % 60)
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
