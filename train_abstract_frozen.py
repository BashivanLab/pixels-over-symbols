"""Nat->Abs substitution: a one-hot Abstract model around a frozen sequence core.

Loads the sequence core of a trained model (``--seq_model_path``, a Natural
model in the paper), optionally freezes it and the output head, and trains the
rest on the abstract task. Without ``--seq_model_path``, ``--freeze_seq_model``
gives the untrained-core baseline.
"""

import os
import time
from pprint import pprint

import torch
import torchvision

import wandb
from models.model import CustomRNN
from task.dataloader import prepare_abstract_dataloaders
from training_loop import train_model
from utils.argparse import TrainArgParser
from utils.const import INPUT_SIZE, OUTPUT_SIZE
from utils.helper import is_compile_supported, seed_everything


def train_abstract(
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
        save_mid_epoch: bool = True,
        save_until_acc: float = None,
        # model args
        seq_model_type: str = 'GRU',
        num_seq_layers: int = 1,
        hidden_size: int = 512,
        dropout: float = 0.3,
        seq_model_path=None,
        freeze_seq_model=False,
        freeze_fc_out=False,
        # Transformer-specific args
        num_heads: int = 8,
        # configs
        rep_index: int = 0,
        results_dir: str = './results',
        print_freq: int = 5,
        use_wandb: bool = False,
        **kwargs,
):
    """Train a :class:`CustomRNN` whose sequence core is loaded (and optionally frozen).

    ``seq_model_path`` gives the donor checkpoint; ``freeze_seq_model`` /
    ``freeze_fc_out`` control which components stay fixed during training.
    Otherwise mirrors :func:`train_abstract.train_abstract`.
    """
    tasks = 'all'
    print(f"[Start] Seq: {seq_model_type}, Hidden: {hidden_size}, Rep: {rep_index}, Tasks: {tasks}")
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
            "rep_index": rep_index,
            "natural_stim": False,
            "model": "RNN",
        },
        name=f"abs_frozen_{seq_model_type}_hidden{hidden_size}_tasks_{tasks}_nomturk",
        mode=None if use_wandb else 'disabled',
    )

    # Prepare datasets and loaders
    train_loader, val_loader, _, _ = prepare_abstract_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size,
        val_dataset_size=val_dataset_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        train_vars=train_vars,
        val_vars=val_vars,
    )

    print(f'Found frozen seq model checkpoint at {seq_model_path}')

    model = CustomRNN(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        seq_model_type=seq_model_type,
        num_seq_layers=num_seq_layers,
        dropout=dropout,
        seq_model_path=seq_model_path,
        freeze_seq_model=freeze_seq_model,
        freeze_fc_out=freeze_fc_out,
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
        natural_stim=False,
        save_mid_epoch=save_mid_epoch,
        save_until_acc=save_until_acc,
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
        '--save_mid_epoch',
        action='store_true',
        default=False,
        help="Save checkpoint during first epoch if validation accuracy is between 60-80%%"
    )
    parser.add_argument(
        '--save_until_acc',
        type=float,
        default=None,
        help="Save mid-epoch checkpoints in any epoch (denser near the target) and stop "
             "training at the first one with validation accuracy >= this value"
    )
    parser.add_argument(
        '--seq_model_path',
        type=str,
        default=None,
        help="Donor checkpoint (CustomRNN, CNNRNN or DimMatchCNNRNN) whose sequence core is "
             "loaded; fc_in, and fc_out unless --freeze_fc_out, are re-initialized"
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
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    pprint(vars(args))

    train_abstract(
        **vars(args)
    )

    total_secs = time.time() - start_time
    hours = int(total_secs // 3600)
    minutes = int((total_secs % 3600) // 60)
    seconds = int(total_secs % 60)
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
