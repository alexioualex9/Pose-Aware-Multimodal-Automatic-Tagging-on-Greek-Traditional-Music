import torch
import math
import os

from video_scores import video_level_val_loss

def train_model(train_ds, train_loader, val_loader, mlp, labels, config, subset):

    criterion = config['loss_function']
    opt = config['optimizer']
    scheduler = config.get("scheduler", None)
    mlp_hidden = config['mlp_hidden']
    epochs = config['epochs']
    patience = config['early_stopping_patience']
    mean, std = config['mean'], config['std']
    standardize = config['standardize']
    device = config['device']
    save_path = config['save_path']
    D, C = train_ds.emb_dim, train_ds.C

    best_val = math.inf
    best_sd = None
    noimp = 0
    for ep in range(1, epochs+1):
        # re-sample ONE random clip per video for this epoch
        train_ds.set_epoch(ep)
        mlp.train()
        running = 0.0
        steps = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yb = yb.float()

            logits = mlp(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1
        tr_loss = running / max(1, steps)

        # Video-level VAL loss (ALL clips → avg logits)
        val_loss = video_level_val_loss(mlp, val_loader, criterion, device, subset)
        print(f"[epoch {ep:03d}] train_loss={tr_loss:.6f} | val_loss={val_loss:.6f}")

        if scheduler is not None:
            scheduler.step()

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            noimp = 0
            best_sd = {k: v.detach().cpu() for k, v in mlp.state_dict().items()}
            ckpt = {
                "state_dict": best_sd,
                "emb_dim": D,
                "num_classes": C,
                "labels": labels,
                "mlp_hidden": mlp_hidden,
                "standardize": bool(standardize),
                "mean": (None if mean is None else mean.detach().to("cpu", dtype=torch.float32)),
                "std":  (None if std  is None else std.detach().to("cpu", dtype=torch.float32)),
                "val_loss": float(best_val),
            }
            torch.save(ckpt, save_path)
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[early stop] no improvement for {patience} epochs — stopping.")
                break

