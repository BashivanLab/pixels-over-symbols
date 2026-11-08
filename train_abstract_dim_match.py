"""Training entry point for the dimensionality-matched Abstract model.

Routes *abstract* one-hot embeddings through the *naturalistic* CNNRNN pipeline
(projection -> attention pool -> task concat -> sequence core), so the only thing
that differs from a Natural model is the *content* of the token features:

  1. a fixed, non-trainable random projection lifts each 85-d stimulus embedding
     to ``num_tokens`` tokens of dimensionality ``upsample_dim`` (matching the
     token tensor a frozen ResNet-50 emits), then
  2. Gaussian noise with std ``noise_scale`` (the noise level s) is added to every
     channel, scaled up with the pose extremity (view/size) of the stimulus when
     ``noise_pose_k > 0``.

The clean token channels have std ``||emb|| / sqrt(embedding_size)`` ~= 0.2 (see
:func:`models.dim_match.build_projection`), so ``--noise_scale 0.2`` is roughly
unit SNR.

With ``--seq_model_path`` this is the Nat->Abs substitution for the
dimensionality-matched model.
"""

import os
import time
from pprint import pprint

import torch
import torchvision

import wandb
from models.dim_match import DimMatchCNNRNN, build_projection
from models.model import CNNRNN
from task.dataloader import prepare_abstract_dataloaders
from training_loop import train_model
from utils.argparse import TrainArgParser
from utils.const import TASK_NAME_TASK_INDEX, OUTPUT_SIZE
from utils.helper import is_compile_supported, seed_everything

def train_abstract_dim_match(
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
        # model args
        seq_model_type: str = 'GRU',
        num_seq_layers: int = 1,
        hidden_size: int = 256,
        dropout: float = 0.3,
        num_heads: int = 8,
        # dimensionality-match args
        upsample_dim: int = 2048,
        num_tokens: int = 49,
        noise_scale: float = 1.0,
        noise_pose_k: float = 0.0,
        # substitution args (as in train_natural_frozen.py)
        seq_model_path: str = None,
        freeze_seq_model: bool = False,
        freeze_fc_out: bool = False,
        freeze_input_pathway: bool = False,
        label_smoothing: float = None,
        # configs
        rep_index: int = 0,
        results_dir: str = './results',
        print_freq: int = 5,
        use_wandb: bool = False,
        **kwargs,
):
    """Train a :class:`DimMatchCNNRNN` on upsampled + noised abstract embeddings.

    Mirrors :func:`train_abstract.train_abstract` (abstract dataloaders, symbolic
    input, ``natural_stim=False``) but swaps the model for the natural
    CNNRNN pipeline fed dimensionality-matched, noised tokens.
    ``(num_tokens, upsample_dim)`` default to a ResNet-50 backbone (49 tokens x
    2048-d).

    Passing ``seq_model_path`` (with ``freeze_seq_model``) makes this the Nat->Abs
    substitution: the sequence core is loaded from a Natural (or Abstract)
    checkpoint while ``proj``/``attn_pool``/``fc_in``/``fc_out`` retrain on
    dimensionality-matched tokens. With ``natural_stim=False`` the label smoothing
    defaults to 0.0 while Natural runs use 0.2; the paper models set 0.2 on both.
    """
    tasks = 'all'
    print(f"[Start] DimMatch Seq: {seq_model_type}, Hidden: {hidden_size}, "
          f"upsample_dim: {upsample_dim}, tokens: {num_tokens}, "
          f"noise_scale: {noise_scale} (absolute std, ~0.2 = unit SNR), "
          f"noise_pose_k: {noise_pose_k}, Rep: {rep_index}")
    if seq_model_path is not None:
        print(f"[Start] Substituting sequence core from {seq_model_path} "
              f"(freeze_seq_model={freeze_seq_model}, freeze_fc_out={freeze_fc_out}, "
              f"freeze_input_pathway={freeze_input_pathway})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    seed = 2025 + rep_index
    seed_everything(seed)

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
            "model": "DimMatchCNNRNN",
            "upsample_dim": upsample_dim,
            "num_tokens": num_tokens,
            "noise_scale": noise_scale,
            "noise_pose_k": noise_pose_k,
            "seq_model_path": str(seq_model_path) if seq_model_path else None,
            "freeze_seq_model": freeze_seq_model,
            "freeze_fc_out": freeze_fc_out,
            "freeze_input_pathway": freeze_input_pathway,
            "label_smoothing": label_smoothing,
        },
        name=(f"abs_dimmatch_{seq_model_type}_hidden{hidden_size}_dim{upsample_dim}"
              f"_ns{noise_scale}_k{noise_pose_k}"
              f"{'_frozen' if freeze_seq_model else ''}"
              f"_nomturk_tasks_{tasks}"),
        mode=None if use_wandb else 'disabled',
    )

    # Abstract dataloaders: yield the 85-d symbolic embeddings (no vision backbone).
    train_loader, val_loader, train_dataset, _ = prepare_abstract_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size,
        val_dataset_size=val_dataset_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        train_vars=train_vars,
        val_vars=val_vars,
    )

    embedding_size = train_dataset.embedding_size
    n_cont = train_dataset.view_size + train_dataset.size_dim

    # No-backbone CNNRNN sized for the synthetic (B, T, num_tokens, upsample_dim) tokens.
    core = CNNRNN(
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        label_size=len(TASK_NAME_TASK_INDEX),
        seq_model_type=seq_model_type,
        num_seq_layers=num_seq_layers,
        dropout=dropout,
        num_heads=num_heads,
        no_backbone=True,
        feat_dim=upsample_dim,
        num_input_tokens=num_tokens,
        # Substitution: CNNRNN loads the donor core itself, then re-initializes the
        # input pathway unless freeze_input_pathway keeps (and freezes) the donor's.
        seq_model_path=seq_model_path,
        freeze_seq_model=freeze_seq_model,
        freeze_fc_out=freeze_fc_out,
        freeze_input_pathway=freeze_input_pathway,
    )

    projection = build_projection(embedding_size, num_tokens, upsample_dim, seed)
    model = DimMatchCNNRNN(
        core=core,
        projection=projection,
        n_cont=n_cont,
        noise_scale=noise_scale,
        noise_pose_k=noise_pose_k,
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
        label_smoothing=label_smoothing,
        save_mid_epoch=save_mid_epoch,
        run=run,
    )
    run.finish()
    print(f"[Complete] DimMatch Seq: {seq_model_type}, Hidden: {hidden_size}, Rep: {rep_index}")


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
        '--upsample_dim',
        type=int,
        default=2048,
        help="Token feature dimensionality to upsample the one-hot to "
             "(match the Natural backbone's feature dim; ResNet50=2048)."
    )
    parser.add_argument(
        '--num_tokens',
        type=int,
        default=49,
        help="Number of synthetic tokens per frame for attention pooling "
             "(match the Natural backbone; ResNet50 7x7=49)."
    )
    parser.add_argument(
        '--noise_scale',
        type=float,
        default=1.0,
        help="Noise level s: the std of the Gaussian noise added to the tokens. Clean "
             "channels have std ~0.2, so 0.2 is roughly unit SNR and 1.0 is a 5:1 "
             "noise:signal ratio that does not train."
    )
    parser.add_argument(
        '--noise_pose_k',
        type=float,
        default=0.0,
        help="Pose sensitivity of the noise: "
             "sigma = noise_scale*(1 + k*(pose_rms - 0.715)). 0 = constant noise, "
             "i.e. no scaling with viewing difficulty. The paper's noisy models use k = 0.5."
    )

    parser.add_argument(
        '--seq_model_path',
        type=str,
        default=None,
        help="Checkpoint to load the sequence core from (Natural CNNRNN, "
             "Abstract CustomRNN, or another DimMatchCNNRNN). The input pathway "
             "(proj/attn_pool/fc_in/norm_in) is re-initialized unless "
             "--freeze_input_pathway."
    )
    parser.add_argument(
        '--freeze_seq_model',
        action='store_true',
        default=False,
        help="Freeze the sequence core, so only the input pathway and output head "
             "adapt (theta). Without --seq_model_path this freezes a random core."
    )
    parser.add_argument(
        '--freeze_fc_out',
        action='store_true',
        default=False,
        help="Also freeze norm_out/fc_out, leaving only the input pathway trainable."
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
        help="Override the training-loss label smoothing. Default (None) uses the "
             "abstract recipe, i.e. 0.0, while the Natural trainers use 0.2. The "
             "paper's dimensionality-matched models use 0.2."
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    pprint(vars(args))

    train_abstract_dim_match(**vars(args))

    total_secs = time.time() - start_time
    hours = int(total_secs // 3600)
    minutes = int((total_secs % 3600) // 60)
    seconds = int(total_secs % 60)
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
