import numpy as np
import torch
import os
import math
from config import MODELS_DIR

#def swap_np(x, i, j):
#    x = x.copy()
#    x[i], x[j] = x[j], x[i]
#    return x

#def swap_torch(x, i , j):
#    x = x.clone()
#    x[i], x[j] = x[j].clone(), x[i].clone()
#    return x

def swap_np(x, i, j):
    x = x.copy()
    if x.ndim == 1:          # [C]
        x[i], x[j] = x[j], x[i]
    elif x.ndim == 2:        # [B, C]
        x[:, [i, j]] = x[:, [j, i]]
    else:
        raise ValueError(f"Unsupported shape {x.shape}")
    return x

def swap_torch(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    # x: [B,C] ή [C]
    x = x.clone()
    if x.dim() == 1:
        x[i], x[j] = x[j].clone(), x[i].clone()
    elif x.dim() == 2:
        x[:, [i, j]] = x[:, [j, i]]
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {x.shape}")
    return x


# ------------------------------------------------------------
# Training mode
# ------------------------------------------------------------
def train_transformer(train_loader, val_loader, model, mean_t, std_t, config, subset, time_window, seed, transformer):

    # Optimizer
    if config['optimizer'] == "AdamW":
       optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=config['weight_decay'])
    else:
       raise NotImplementedError(
            'No optimizer implementation found for the given config.')

    criterion = config['loss_function']

    # Fix paths
    if subset:
       dataset_folder = "subset"
    else:
       dataset_folder = "whole_dataset"


    if config['model_name'] == "seq_transformer_av":
         model_folder =  f"{config['audio_model_name']}_{config['video_model_name']}"
    elif config['model_name'] == "seq_transformer_vs_masked":
         model_folder = f"{config['video_model_name']}_{config['skeleton_model_name']}"
    elif config['model_name'] == "seq_transformer_as_masked":
         model_folder = f"{config['audio_model_name']}_{config['skeleton_model_name']}"
    elif config['model_name'] == "seq_transformer_avs_masked":
         model_folder = f"{config['audio_model_name']}_{config['video_model_name']}_{config['skeleton_model_name']}"
    else:
         raise ValueError ("No such combination of modalities or model used")

    temp_path = os.path.join(MODELS_DIR, dataset_folder, time_window, config['transformer'], str(seed), model_folder)
    os.makedirs(temp_path, exist_ok=True)
    best_path = os.path.join(temp_path, f"best_{config['model_name']}.pt")
    best_val = math.inf
    for ep in range(1, config['epochs'] + 1):
        model.train()
        running = 0.0
        steps = 0

        for batch in train_loader:
            if config['model_name'] == "seq_transformer_avs_masked":
                emb_a, emb_v, emb_s, mask_s, y, valid_mask, _vid = batch
                emb_a = emb_a.to(config['device'])
                emb_v = emb_v.to(config['device'])
                emb_s = emb_s.to(config['device'])
                mask_s = mask_s.to(config['device'])
                y = y.to(config['device'])
                valid_mask = valid_mask.to(config['device'])
                logits = model(emb_a, emb_v, emb_s, mask_s, valid_mask=valid_mask)

            elif config['model_name'] == "seq_transformer_as_masked":
                emb_a, emb_s, mask_s, y, valid_mask, _vid = batch
                emb_a = emb_a.to(config['device'])
                emb_s = emb_s.to(config['device'])
                mask_s = mask_s.to(config['device'])
                y = y.to(config['device'])
                valid_mask = valid_mask.to(config['device'])
                logits = model(emb_a, emb_s, mask_s, valid_mask=valid_mask)

            elif config['model_name'] == "seq_transformer_vs_masked":
                emb_v, emb_s, mask_s, y, valid_mask, _vid = batch
                emb_v = emb_v.squeeze(0).to(config['device'])
                emb_s = emb_s.squeeze(0).to(config['device'])
                mask_s = mask_s.squeeze(0).to(config['device'])
                valid_mask = valid_mask.to(config['device'])
                y = y.squeeze(0).to(config['device'])
                logits = model(emb_v, emb_s, mask_s, valid_mask=valid_mask)

            else:
                emb_a, emb_v, y, valid_mask, _vid = batch
                emb_a = emb_a.to(config['device'])
                emb_v = emb_v.to(config['device'])
                y = y.to(config['device'])
                valid_mask = valid_mask.to(config['device'])
                logits = model(emb_a, emb_v, valid_mask=valid_mask)


#            if not torch.isfinite(logits).all():
#               print("NON-FINITE LOGITS!", torch.isnan(logits).any().item(), torch.isinf(logits).any().item())
#               print("logits min/max:", logits.min().item(), logits.max().item())
#               raise SystemExit


            loss = criterion(logits, y)
#            if getattr(model, "last_gate_reg", None) is not None:
#                loss = loss + model.lambda_gate * model.last_gate_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            steps += 1

        tr_loss = running / max(1, steps)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if config['model_name'] == "seq_transformer_avs_masked":
                    emb_a, emb_v, emb_s, mask_s, y, valid_mask, _vid = batch
                    emb_a = emb_a.to(config['device'])
                    emb_v = emb_v.to(config['device'])
                    emb_s = emb_s.to(config['device'])
                    mask_s = mask_s.to(config['device'])
                    y = y.to(config['device'])
                    valid_mask = valid_mask.to(config['device'])
                    logits = model(emb_a, emb_v, emb_s, mask_s, valid_mask=valid_mask)

                    if subset:
                       y = swap_torch(y, 20, 21)
                       logits = swap_torch(logits, 20, 21)
                    else:
                       y = swap_torch(y, 22, 23)
                       logits = swap_torch(logits, 22, 23)

                elif config['model_name'] == "seq_transformer_as_masked":
                    emb_a, emb_s, mask_s, y, valid_mask, _vid = batch
                    emb_a = emb_a.to(config['device'])
                    emb_s = emb_s.to(config['device'])
                    mask_s = mask_s.to(config['device'])
                    y = y.to(config['device'])
                    valid_mask = valid_mask.to(config['device'])
                    logits = model(emb_a, emb_s, mask_s, valid_mask=valid_mask)

                    if subset:
                       y = swap_torch(y, 20, 21)
                       logits = swap_torch(logits, 20, 21)
                    else:
                       y = swap_torch(y, 22, 23)
                       logits = swap_torch(logits, 22, 23)

                elif config['model_name'] == "seq_transformer_vs_masked":
                    emb_v, emb_s, mask_s, y, valid_mask, _vid = batch
                    emb_v = emb_v.squeeze(0).to(config['device'])
                    emb_s = emb_s.squeeze(0).to(config['device'])
                    mask_s = mask_s.squeeze(0).to(config['device'])
                    y = y.squeeze(0).to(config['device'])
                    logits = model(emb_v, emb_s, mask_s, valid_mask=valid_mask)

                    if subset:
                       y = swap_torch(y, 20, 21)
                       logits = swap_torch(logits, 20, 21)
                    else:
                       y = swap_torch(y, 22, 23)
                       logits = swap_torch(logits, 22, 23)

                else:
                    emb_a, emb_v, y, valid_mask, _vid = batch
                    emb_a = emb_a.to(config['device'])
                    emb_v = emb_v.to(config['device'])
                    y = y.to(config['device'])
                    y = y.float()
                    valid_mask = valid_mask.to(config['device'])
                    logits = model(emb_a, emb_v, valid_mask=valid_mask)

                    if subset:
                       y = swap_torch(y, 20, 21)
                       logits = swap_torch(logits, 20, 21)
                    else:
                       y = swap_torch(y, 22, 23)
                       logits = swap_torch(logits, 22, 23)

                val_losses.append(float(criterion(logits, y).cpu()))

        validation_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        print(f"[ep {ep:03d}] train_loss={tr_loss:.6f} | val_loss={validation_loss:.6f}")

        if validation_loss < best_val - 1e-6:
            best_val = validation_loss
            ckpt = {
                "fusion_state_dict": model.state_dict(),
                "epoch": ep,
                "val_loss": validation_loss,
                "mean_v": mean_t.cpu().numpy() if mean_t is not None else None,
                "std_v":  std_t.cpu().numpy()  if std_t  is not None else None,
            }

            torch.save(ckpt, best_path)
            print(f"[info] Saved best checkpoint to {best_path}")

    print(f"[done] Best val loss: {best_val:.6f}")
    if best_path:
        print(f"[done] Best ckpt: {best_path}")
