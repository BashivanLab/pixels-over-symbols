"""Training loop shared by every model family.

:func:`train_model` runs the AdamW training/validation loop over a DataLoader,
computing loss/accuracy only on action frames (label != 2), logging
per-task/per-feature statistics through the summary loggers, checkpointing at a
few epochs, early-stopping at >0.99 validation accuracy, and finally writing
training-curve and data-distribution plots.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.helper import print_model_info, print_cuda_info
from utils.logger import MetricLogger, SummaryLogger
from utils.const import OUTPUT_SIZE
from utils.plot import plot_training_stats
from utils.stim_io import HvMImageMapper


def warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for ``warmup_epochs`` then cosine decay to zero over the remaining epochs."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1. + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def run_quick_val(model, val_loader, device):
    """Return mean action-frame validation accuracy in eval mode, then restore training mode."""
    criterion_val = nn.CrossEntropyLoss(reduction='mean')
    val_loss = 0.0
    val_acc = 0.0
    num_val_batches = len(val_loader)

    model.eval()
    with torch.no_grad():
        for inputs, labels, task_index, task_info in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            task_index = task_index.to(device, non_blocking=True)

            outputs, _, _ = model(inputs, task_index)
            labels_flat = labels.view(-1).long()
            mask = (labels_flat != 2)
            outputs_flat = outputs.contiguous().view(-1, OUTPUT_SIZE)[mask]
            labels_flat = labels_flat[mask]
            loss = criterion_val(outputs_flat, labels_flat)

            val_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            preds_flat = preds.contiguous().view(-1)[mask]
            val_acc += (preds_flat == labels_flat).float().mean().item()

    model.train()  # make sure to restore training mode
    return val_acc / num_val_batches


def train_model(
        model,
        train_loader,
        val_loader,
        rep_index,
        num_epochs,
        warmup_epochs,
        learning_rate,
        weight_decay,
        grad_norm,
        save_folder,
        device,
        print_freq=10,
        natural_stim=False,
        label_smoothing=None,
        save_mid_epoch=False,
        save_final_only=False,
        save_until_acc=None,
        run=None,
):
    """Train ``model`` for ``num_epochs`` and write checkpoints, stats, and plots.

    Loss and accuracy are computed only over action frames. The learning rate
    follows a linear warmup and a cosine decay, stepped once per epoch. Checkpoints
    are saved at quarter-epoch milestones (or only the final epoch when
    ``save_final_only``), and with ``save_mid_epoch`` also at the end of epoch 0
    plus mid-epoch whenever epoch-0 validation accuracy lands in [0.6, 0.8], and a
    final ``final_{model_name}_rep{rep_index}.pth``. ``run`` is an optional wandb
    run for metric logging. Training stops early once the 10-epoch mean val
    accuracy > 0.99.

    ``save_until_acc`` replaces the epoch-0 window: mid-epoch checks run in every
    epoch on the ``save_mid_epoch`` schedule, five times as often once a check is
    within 0.01 of the target (0.05 if the last regular interval gained >= 0.01,
    where a regular step could overshoot), every check >= 0.6 is saved (accuracy to 4
    decimals), and training returns, without a final checkpoint, at the first
    check >= ``save_until_acc``. The extra checks leave the weights unchanged:
    the persistent val workers draw nothing from the training RNG after their
    first pass.
    """
    if save_final_only:
        save_point_epochs = [num_epochs]
    else:
        # save 4 checkpoints including final
        save_point_epochs = [num_epochs // 4, num_epochs // 2, (3 * num_epochs) // 4, num_epochs]
    if save_mid_epoch:
        # end of the first epoch, so there is always an early-training checkpoint
        # even when the mid-epoch accuracy window below never triggers
        save_point_epochs.append(1)
    print_cuda_info()
    print_model_info(model)
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
    train_mapper, val_mapper = HvMImageMapper(train_dataset), HvMImageMapper(val_dataset)
    train_metric_logger, val_metric_logger = MetricLogger(), MetricLogger()
    train_summary_logger = SummaryLogger(
        train_dataset.hvm_loader.df,
        train_mapper,
    )
    val_summary_logger = SummaryLogger(
        val_dataset.hvm_loader.df,
        val_mapper
    )

    # Natural models train with label smoothing 0.2 and abstract ones without; the
    # substitution and dimensionality-matched trainers pass it explicitly.
    if label_smoothing is None:
        label_smoothing = 0.2 if natural_stim else 0.0
    print(f'[Training] label_smoothing={label_smoothing} (natural_stim={natural_stim})')
    criterion_train = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    criterion_val = nn.CrossEntropyLoss(
        label_smoothing=0.0,
        reduction='mean'
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs
    )
    model_name = 'CNNRNN_model' if natural_stim else 'RNN_model'
    stats = {
        "losses": [],
        "accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
    }
    last_check_acc = last_regular_acc = last_gain = 0.0

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = len(train_loader)
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = len(val_loader)

        model.train()
        for batch_i, (inputs, labels, task_index, task_info) in enumerate(train_metric_logger.log_every(train_loader)):
            optimizer.zero_grad()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            task_index = task_index.to(device, non_blocking=True)
            outputs, _, _ = model(inputs, task_index)

            labels_flat = labels.view(-1).long()
            mask = (labels_flat != 2)  # Create mask for labels that are NOT 2
            if not mask.any():
                raise RuntimeError('Should have at least 1 action frame')
            # gradient only on action frames
            outputs_flat = outputs.contiguous().view(-1, OUTPUT_SIZE)[mask]
            labels_flat = labels_flat[mask]
            loss = criterion_train(outputs_flat, labels_flat)

            # backward + step
            train_loss += loss.item()
            loss.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
            optimizer.step()

            # predictions & accuracy
            with torch.no_grad():
                preds = outputs.argmax(dim=-1)
                preds_flat = preds.contiguous().view(-1)[mask]
                train_batch_acc = (preds_flat == labels_flat).float().mean().item()
            train_acc += train_batch_acc

            train_summary_logger.batch_update(
                labels=labels,
                outputs=outputs,
                task_info=task_info,
                task_index=task_index
            )

            checkpoints_per_epoch = 20
            # at least 1, so that runs with fewer than 20 batches per epoch also work
            save_interval = max(1, num_train_batches // checkpoints_per_epoch)

            if save_until_acc is not None:
                # dense checks only where a regular interval could step past the target: within
                # 0.05 after a fast interval (gain >= 0.01), else within 0.01
                near = (last_check_acc >= save_until_acc - 0.01
                        or (last_check_acc >= save_until_acc - 0.05 and last_gain >= 0.01))
                check_interval = max(1, save_interval // 5) if near else save_interval
                if batch_i > 0 and batch_i % check_interval == 0:
                    last_check_acc = run_quick_val(model, val_loader, device)
                    if batch_i % save_interval == 0:
                        last_gain, last_regular_acc = last_check_acc - last_regular_acc, last_check_acc
                    if last_check_acc >= 0.6:
                        model_fp = os.path.join(
                            save_folder,
                            f'{model_name}_acc{last_check_acc:.4f}_epoch_{epoch}_batch_{batch_i}.pth'
                        )
                        print(f'[Training] Saving mid-epoch checkpoint: Acc {last_check_acc:.4f} '
                              f'(epoch {epoch}, batch {batch_i})')
                        torch.save(model.state_dict(), model_fp)
                    if last_check_acc >= save_until_acc:
                        print(f'[Training] Reached {save_until_acc}, stopping without a final checkpoint')
                        return
            elif save_mid_epoch:
                if epoch == 0 and batch_i > 0 and batch_i % save_interval == 0:
                    mid_epoch_val_acc = run_quick_val(model, val_loader, device)

                    if 0.6 <= mid_epoch_val_acc <= 0.8:
                        model_fp = os.path.join(
                            save_folder,
                            f'{model_name}_acc{mid_epoch_val_acc:.2f}_epoch_{epoch}_batch_{batch_i}.pth'
                        )
                        print(f'[Training] Saving mid-epoch checkpoint: Acc {mid_epoch_val_acc:.4f}')
                        torch.save(model.state_dict(), model_fp)

        avg_train_loss = train_loss / num_train_batches
        avg_train_acc = train_acc / num_train_batches
        train_metric_logger.update(loss=avg_train_loss, acc=avg_train_acc)
        train_summary_logger.epoch_update()

        model.eval()
        with torch.no_grad():
            for inputs, labels, task_index, task_info in val_metric_logger.log_every(val_loader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                task_index = task_index.to(device, non_blocking=True)
                outputs, _, _ = model(inputs, task_index)

                labels_flat = labels.view(-1).long()
                mask = (labels_flat != 2)
                if not mask.any():
                    raise RuntimeError('Should have at least 1 action frame')
                outputs_flat = outputs.contiguous().view(-1, OUTPUT_SIZE)[mask]
                labels_flat = labels_flat[mask]
                loss = criterion_val(outputs_flat, labels_flat)

                val_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                preds_flat = preds.contiguous().view(-1)[mask]
                val_batch_acc = (preds_flat == labels_flat).float().mean().item()
                val_acc += val_batch_acc

                val_summary_logger.batch_update(
                    labels=labels,
                    outputs=outputs,
                    task_info=task_info,
                    task_index=task_index
                )

        avg_val_loss = val_loss / num_val_batches
        avg_val_acc = val_acc / num_val_batches
        val_metric_logger.update(loss=avg_val_loss, acc=avg_val_acc)
        val_summary_logger.epoch_update()

        train_metric_logger.print_epoch_summary(
            header='TRAIN: ',
            print_freq=print_freq,
            summary_logger=train_summary_logger,
            scheduler=scheduler
        )
        val_metric_logger.print_epoch_summary(
            header='VAL: ',
            print_freq=print_freq,
            summary_logger=val_summary_logger,
        )

        if run is not None:
            run.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": avg_val_acc,
                "lr": scheduler.get_last_lr()[0],
            })
        stats["losses"].append(avg_train_loss)
        stats["accuracies"].append(avg_train_acc)
        stats["val_losses"].append(avg_val_loss)
        stats["val_accuracies"].append(avg_val_acc)

        scheduler.step()

        if (epoch + 1) in save_point_epochs:
            model_fp = os.path.join(save_folder, f'{model_name}_epoch_{epoch}.pth')

            print(f'[Training] Epoch {epoch + 1}, saving model at {model_fp} ...')
            torch.save(model.state_dict(), model_fp)

        if epoch >= 9 and np.mean(stats["val_accuracies"][-10:]) > 0.99:
            print(f'[Training] Validation accuracy reached over 0.99, early breaking at epoch {epoch + 1}')
            break

    final_model_path = os.path.join(save_folder, f'final_{model_name}_rep{rep_index}.pth')
    torch.save(model.state_dict(), final_model_path)

    stats['train_task_acc'] = train_summary_logger.task_acc
    stats['val_task_acc'] = val_summary_logger.task_acc
    stats['train_var_acc'] = train_summary_logger.var_acc
    stats['val_var_acc'] = val_summary_logger.var_acc
    stats['train_feature_acc'] = train_summary_logger.feature_acc
    stats['val_feature_acc'] = val_summary_logger.feature_acc
    stats['train_action_acc'] = train_summary_logger.action_acc
    stats['val_action_acc'] = val_summary_logger.action_acc
    # plot per epoch training stats
    plot_training_stats(stats, save_folder, rep_index)

    # plot data distribution
    dist_folder = os.path.join(save_folder, 'data_distribution')
    os.makedirs(dist_folder, exist_ok=True)

    epochs_ran = len(stats["losses"])
    for epoch in range(epochs_ran):
        if (epoch + 1) % 20 == 0 or epoch == epochs_ran - 1:
            for feature in ['category', 'identity', 'position', 'row', 'var']:
                train_summary_logger.plot_feature_sampling(
                    feature=feature,
                    epoch=epoch,
                    show=False,
                    save_path=os.path.join(dist_folder, f'train_{feature}_epoch{epoch}_distribution.png')
                )
                val_summary_logger.plot_feature_sampling(
                    feature=feature,
                    epoch=epoch,
                    show=False,
                    save_path=os.path.join(dist_folder, f'val_{feature}_epoch{epoch}_distribution.png')
                )
    train_summary_logger.save_pair_dfs(save_folder, prefix='train')
    val_summary_logger.save_pair_dfs(save_folder, prefix='val')
    torch.cuda.empty_cache()
    return
