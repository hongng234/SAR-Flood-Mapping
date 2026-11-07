"""Train the U-Net on the train split and keep the version that scores best on the validation split.

Where this file sits in the pipeline (download -> baselines -> train -> evaluate -> figures):
    - `python main.py train` calls `train()`; it writes outputs/checkpoint.pt (best weights) and
      outputs/history.json (loss and validation IoU per epoch, used for the training figure).
    - `src/evaluate.py` and `src/visualize.py` call `load_model()` to get the trained model back.

Only the train and val splits are used here; test and Bolivia are never touched (a test in
tests/test_model_train.py checks this), so the final scores are honest.

Words used below:
    - epoch: one full pass over all training chips.
    - batch: a small group of chips (config train.batch_size) processed together on the GPU.
    - IoU (Intersection over Union): overlap of predicted and true water / their union; 1.0 = perfect.
    - early stopping: stop training when validation IoU has not improved for `patience` epochs.
"""

# Allows modern type hints on older Python versions.
from __future__ import annotations

import json  # writes history.json
import time  # measures how long training takes
from pathlib import Path  # file paths

import numpy as np  # mean of the batch losses
import torch  # PyTorch
from torch.utils.data import DataLoader  # groups dataset items into shuffled batches

# Our own code: dataset + train statistics (data.py), IoU bookkeeping (evaluate.py), model + loss (model.py).
from .data import FloodDataset, compute_norm_stats, get_split
from .evaluate import Metrics
from .model import build_unet, flood_loss


def get_device() -> torch.device:
    """Return the GPU ("cuda") if one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate(batch):
    """Combine a list of dataset items into one batch (used by the DataLoader).

    Args:
        batch: list of (image (2,H,W), mask (H,W), chip_id) tuples.
    Returns:
        (images (B,2,H,W), masks (B,H,W), list of B chip ids).
    """
    # zip(*batch) "unzips" the list of triples into three tuples: all images, all masks, all ids.
    imgs, masks, ids = zip(*batch)
    # torch.stack puts the tensors on top of each other along a new first (batch) axis.
    return torch.stack(imgs), torch.stack(masks), list(ids)


# @torch.no_grad() turns off gradient tracking inside this function: faster and uses less memory.
# Gradients are only needed while training, not while measuring.
@torch.no_grad()
def validate(model, loader, device) -> float:
    """Water IoU of the model on all chips of a data loader.

    A pixel is predicted as water when its sigmoid probability is above 0.5.

    Args:
        model: the U-Net. loader: DataLoader over the validation chips. device: cpu or cuda.
    Returns:
        IoU as a float between 0 and 1 (micro-averaged over all pixels of all chips).
    """
    # eval mode: BatchNorm uses its stored statistics instead of the batch's, so results are deterministic.
    model.eval()
    # Accumulates TP/FP/FN/TN counts over all chips (see src/evaluate.py).
    metrics = Metrics()
    # Loop over the validation batches.
    for img, mask, ids in loader:
        # Move images to the GPU, run the model, take channel 0 -> (B, H, W) logits,
        # turn them into probabilities with sigmoid, then into True/False with "> 0.5".
        pred = torch.sigmoid(model(img.to(device))[:, 0]) > 0.5
        # Loop over the chips in this batch; .cpu().numpy() moves the predictions back to numpy on the CPU.
        for p, m, chip_id in zip(pred.cpu().numpy(), mask.numpy(), ids):
            metrics.update(p, m, chip_id)
    # Score of all chips together; we only need the IoU.
    return metrics.scores()["iou"]


def train(cfg: dict) -> dict:
    """Train the U-Net with early stopping on validation IoU.

    Saves outputs/checkpoint.pt (weights of the best epoch + normalisation stats) and outputs/history.json.

    Args:
        cfg: config dict (train.*, model.*, paths, seed).
    Returns:
        Summary dict: {"best_epoch", "best_val_iou", "minutes"}.
    """
    # Shortcut to the training settings (epochs, batch_size, lr, ...).
    t = cfg["train"]
    # Output folder (created if missing).
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # GPU if available, else CPU.
    device = get_device()

    # ---- Data. Normalisation statistics come from the TRAIN split only (no information from val/test).
    # Lists of chips for the two splits we are allowed to use.
    train_samples, val_samples = get_split(cfg, "train"), get_split(cfg, "val")
    # Per-band mean/std of the training chips, used to normalise every chip.
    stats = compute_norm_stats(train_samples)
    # Training loader: augmentation on (train=True), shuffled order every epoch.
    train_dl = DataLoader(FloodDataset(train_samples, stats, train=True, seed=cfg["seed"]),
                          batch_size=t["batch_size"], shuffle=True, collate_fn=collate)
    # Validation loader: no augmentation, fixed order.
    val_dl = DataLoader(FloodDataset(val_samples, stats), batch_size=t["batch_size"], collate_fn=collate)

    # ---- Model and optimiser. Cosine schedule: the learning rate decays smoothly towards 0.
    # Build the U-Net and move its weights to the device.
    model = build_unet(cfg["model"]["encoder"], cfg["model"]["encoder_weights"]).to(device)
    # AdamW updates the weights from the gradients; lr = step size, weight_decay = small pull of weights towards 0.
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # Learning-rate schedule: follows half a cosine wave from lr down to ~0 over `epochs` epochs.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    # AMP (automatic mixed precision) = compute in 16-bit floats where safe. Only useful on a GPU.
    use_amp = t["amp"] and device.type == "cuda"  # float16 mixed precision: faster on GPU
    # GradScaler multiplies the loss before backward() so tiny float16 gradients do not round to zero.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # prevents float16 gradients underflowing

    # history = one dict per epoch; best_iou starts below any real IoU; best_epoch = epoch of the saved model.
    history, best_iou, best_epoch = [], -1.0, 0
    # Start time, to report training minutes.
    start = time.time()
    # Main loop: epochs 1, 2, ..., epochs.
    for epoch in range(1, t["epochs"] + 1):
        # train mode: BatchNorm/Dropout behave as during training.
        model.train()
        # Loss of each batch in this epoch.
        losses = []
        # One optimisation step per batch (the chip ids are not needed here, hence "_").
        for img, mask, _ in train_dl:
            # Move the batch to the GPU/CPU.
            img, mask = img.to(device), mask.to(device)
            # Reset the gradients left over from the previous step.
            opt.zero_grad()
            # Run the forward pass in mixed precision when AMP is on.
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(img)
            # How wrong the predictions are (ignores -1 pixels).
            loss = flood_loss(logits, mask)
            # Compute the gradient of the loss w.r.t. every weight (on the scaled loss).
            scaler.scale(loss).backward()  # backpropagation
            # Un-scale the gradients and let AdamW change the weights a little.
            scaler.step(opt)  # update the weights
            # Adjust the scale factor for the next step.
            scaler.update()
            # .item() turns the 1-element tensor into a Python float.
            losses.append(loss.item())
        # Lower the learning rate for the next epoch.
        sched.step()

        # Measure IoU on the validation chips after this epoch.
        val_iou = validate(model, val_dl, device)
        # Record this epoch's numbers (used for the training-curve figure).
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": val_iou})
        print(f"epoch {epoch:3d}  train_loss {np.mean(losses):.4f}  val_iou {val_iou:.4f}")
        # Case 1: this is the best validation IoU so far -> save the model.
        if val_iou > best_iou:
            # New best on VALIDATION: save weights + normalisation stats needed for prediction.
            best_iou, best_epoch = val_iou, epoch
            # state_dict() = all the weights; stats and encoder name are needed to rebuild the model later.
            torch.save({"model": model.state_dict(), "norm_stats": stats, "encoder": cfg["model"]["encoder"]},
                       out / "checkpoint.pt")
        # Case 2: no improvement for `patience` epochs in a row -> stop early (more epochs would just overfit).
        elif epoch - best_epoch >= t["patience"]:
            print(f"early stopping: no improvement for {t['patience']} epochs")
            break

    # Short summary of the run; training time in minutes rounded to 1 decimal.
    summary = {"best_epoch": best_epoch, "best_val_iou": best_iou, "minutes": round((time.time() - start) / 60, 1)}
    # Save summary + per-epoch history as readable (indented) JSON.
    (out / "history.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2))
    return summary


def load_model(cfg: dict, device: torch.device):
    """Load the best checkpoint saved by train().

    Args:
        cfg: config dict (paths.output_dir). device: where to put the model.
    Returns:
        (model in eval mode, norm_stats dict) - the stats are needed to normalise new chips the same way.
    """
    # Read the checkpoint file. map_location puts tensors on the requested device.
    # weights_only=False because the file also holds plain dicts/strings (our own file, so it is safe).
    ckpt = torch.load(Path(cfg["paths"]["output_dir"]) / "checkpoint.pt", map_location=device, weights_only=False)
    # Rebuild an empty U-Net with the same encoder (no ImageNet download needed).
    model = build_unet(ckpt["encoder"], encoder_weights=None).to(device)  # weights come from the checkpoint
    # Copy the saved weights into it.
    model.load_state_dict(ckpt["model"])
    # eval() switches to prediction mode and returns the model itself.
    return model.eval(), ckpt["norm_stats"]
