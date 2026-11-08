"""Command-line arguments shared by the train_*.py scripts.

:class:`TrainArgParser` is an ``argparse.ArgumentParser`` preconfigured with the
dataset/training/model hyperparameters every trainer takes; each script adds its
own flags on top. Every model is trained on all 43 tasks of the battery.
"""

import argparse


class TrainArgParser(argparse.ArgumentParser):
    """
    Argument parser for multitask training
    """

    def __init__(self, description="Train a model on the 43-task battery"):
        super().__init__(description=description)
        self._add_common_args()
        self._add_training_args()
        self._add_dataset_args()
        self._add_model_args()

    def _add_dataset_args(self):
        self.add_argument(
            '--batch_size',
            type=int,
            default=128,
            help="Batch size per training run"
        )
        self.add_argument(
            '--train_dataset_size',
            type=int,
            default=16384,
            help="Number of samples in training dataset"
        )
        self.add_argument(
            '--val_dataset_size',
            type=int,
            default=2048,
            help="Number of samples in validation dataset"
        )
        self.add_argument(
            '--num_workers',
            type=int,
            default=6,
            help="Number of parallel processes for dataloader (at least 1)"
        )
        self.add_argument(
            '--pin_mem',
            action='store_true',
            default=False
        )
        self.add_argument(
            '--data_dir',
            type=str,
            help="HvM folder holding python/ and HvM_with_discfade/ (default: data/original)"
        )
        self.add_argument(
            '--train_vars',
            type=int,
            nargs='+',
            default=[0, 3, 6],
            help='stimuli var subset to be used for training '
        )
        self.add_argument(
            '--val_vars',
            type=int,
            nargs='+',
            default=[0, 3, 6],
            help='stimuli var subset to be used for validation'
        )

    def _add_model_args(self):
        self.add_argument(
            '--seq_model_type',
            type=str,
            default='GRU',
            help="Sequence model type: 'rnn', 'gru', 'lstm', 'decoder' (Transformer) or 'bert'"
        )
        self.add_argument(
            '--num_seq_layers',
            type=int,
            default=1,
            help="Number of sequence model layers"
        )
        # Transformer-specific arguments
        self.add_argument(
            '--num_heads',
            type=int,
            default=8,
            help="Number of attention heads for Transformer"
        )
        self.add_argument(
            '--hidden_size',
            type=int,
            default=256,
        )
        self.add_argument(
            '--dropout',
            type=float,
            default=0.3,
        )

    def _add_common_args(self):
        self.add_argument(
            '--rep_index',
            type=int,
            default=0,
            help="Repetition index per configuration; the run is seeded with 2025 + rep_index"
        )
        self.add_argument(
            '--results_dir',
            type=str,
            default='./results',
            help="Run folder that receives the checkpoints, plots and validation curves"
        )
        self.add_argument(
            '--print_freq',
            type=int,
            default=5
        )
        self.add_argument(
            '--wandb',
            dest='use_wandb',
            action='store_true',
            default=False,
            help="Log metrics to Weights & Biases (project RNN-NatAbs). The mode follows "
                 "WANDB_MODE, e.g. WANDB_MODE=offline"
        )

    def _add_training_args(self):
        self.add_argument(
            '--lr',
            type=float,
            default=3e-4,
            help="Learning rate"
        )
        self.add_argument(
            '--weight_decay',
            type=float,
            default=1e-5,
            help="Weight Decay"
        )
        self.add_argument(
            '--grad_norm',
            type=float,
            default=None,
            help="Gradient clipping"
        )
        self.add_argument(
            '--num_epochs',
            type=int,
            default=100,
            help="Number of training epochs"
        )
        self.add_argument(
            '--warmup_epochs',
            type=int,
            default=5,
            help="Number of linear warmup epochs before the cosine decay"
        )
