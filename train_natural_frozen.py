"""Abs->Nat substitution: a Natural model around a frozen sequence core.

Loads the sequence core of a trained model (``--seq_model_path``, a one-hot or
dimensionality-matched Abstract model in the paper) into a :class:`CNNRNN`,
optionally freezes it, the output head and/or the input pathway, and trains the
rest on the naturalistic task. Without ``--seq_model_path``,
``--freeze_seq_model`` gives the untrained-core baseline. Backbone features are
read from the cache in data/features when it exists.
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
        seq_model_path=None,
        freeze_seq_model=False,
        freeze_fc_out=False,
        freeze_input_pathway=False,
        label_smoothing=None,
        # Transformer-specific args
        num_heads: int = 8,
        # configs
        rep_index: int = 0,
        results_dir: str = './results',
        print_freq: int = 5,
        use_wandb: bool = False,
        **kwargs,
):
    """Train a :class:`CNNRNN` whose sequence core is loaded from a checkpoint and optionally frozen.

    Like :func:`train_natural.train_natural`, but initializes the sequence core
    from ``seq_model_path`` and can freeze the core, output head and/or input
    pathway via ``freeze_seq_model`` / ``freeze_fc_out`` /
    ``freeze_input_pathway``. The last covers the whole input pathway
    (``proj``/``attn_pool``/``fc_in``/``norm_in``); with ``freeze_seq_model`` it
    leaves the output head as the only trainable module.
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
        name=f"natural_frozen_{seq_model_type}_hidden{hidden_size}_tasks_{tasks}_nomturk",
        mode=None if use_wandb else 'disabled',
        reinit=True,
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
    )

    print(f'Found frozen seq model checkpoint at {seq_model_path}')

    model = CNNRNN(
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        label_size=len(TASK_NAME_TASK_INDEX),
        cnn_type=cnn_type,
        seq_model_type=seq_model_type,
        num_seq_layers=num_seq_layers,
        dropout=dropout,
        seq_model_path=seq_model_path,
        freeze_seq_model=freeze_seq_model,
        freeze_fc_out=freeze_fc_out,
        freeze_input_pathway=freeze_input_pathway,
        # Transformer args
        num_heads=num_heads,
    ).to(device)

    if is_compile_supported():
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[Warning] torch.compile() failed: {e}")
            print("[Info] Falling back to eager mode.")

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
        label_smoothing=label_smoothing,
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
        '--seq_model_path',
        type=str,
        default=None,
        help="Donor checkpoint (CustomRNN, CNNRNN or DimMatchCNNRNN) whose sequence core is "
             "loaded; the input pathway (see --freeze_input_pathway) and, unless "
             "--freeze_fc_out, the output head are re-initialized"
    )
    parser.add_argument(
        '--freeze_seq_model',
        action='store_true',
        default=False,
        help='Freeze sequential model backbone'
    )
    parser.add_argument(
        '--freeze_fc_out',
        action='store_true',
        default=False,
        help='Keep the loaded output head (norm_out/fc_out) and freeze it (theta+W_out)'
    )
    parser.add_argument(
        '--freeze_input_pathway',
        action='store_true',
        default=False,
        help="Load the donor's whole input pathway (proj/attn_pool/fc_in/norm_in) "
             "and freeze it, along with input_dropout. With --freeze_seq_model this "
             "leaves norm_out/fc_out as the only trainable module: the theta+W_in "
             "(output-head-only) condition. Without this flag the donor pathway is "
             "dropped and re-initialized."
    )
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=None,
        help='Override the training-loss label smoothing. Default (None) keeps the '
             'Natural recipe, i.e. 0.2.'
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    pprint(vars(args))

    train_natural(**vars(args))

    total_secs = time.time() - start_time
    hours = int(total_secs // 3600)
    minutes = int((total_secs % 3600) // 60)
    seconds = int(total_secs % 60)
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
