import os
import torch
from tqdm import tqdm

def swap_np(x, i, j):
    x = x.copy()
    x[i], x[j] = x[j], x[i]
    return x

def swap_torch(x, i , j):
    x = x.clone()
    x[i], x[j] = x[j].clone(), x[i].clone()
    return x


def train_model(model, train_loader, val_loader, config):

    # Define loss and optimizer
    model = model.to(config['device'])
    criterion = config['loss']
    if config['optimizer'] == "AdamW":
       optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
       raise ValueError ("No implementation for this optimizer")

    best_val_loss = float("inf")
    best_path = os.path.join(config['save_dir'], "best.pt")


    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running = 0.0
        n_samples = 0

        for xb, yb, _vids in tqdm(train_loader, desc=f"Train {epoch}"):

            if config['finetuning'] == True:
               if epoch <= warmup_epochs:
                   for p in model.encoder.parameters():
                       p.requires_grad = False
               else:
                   for p in model.encoder.parameters():
                       p.requires_grad = True


            xb = xb.to(config['device'])
            yb = yb.to(config['device'])
            yb = yb.float()

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config['finetuning'] == True:
               torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
            optimizer.step()

            bs = yb.size(0)
            running += float(loss.detach().cpu()) * bs
            n_samples += bs

        train_loss = running / max(1, n_samples)
        print(f"[train] epoch={epoch} loss={train_loss:.6f}")

        # Validation (video-level)
        model.eval()
        val_running = 0.0
        n_videos = 0
        with torch.no_grad():
             for Xs, y, _vid in tqdm(val_loader, desc=f"Val {epoch}"):
                 Xs = Xs.squeeze(0).to(config['device'])  # [Nv,C,T,V]
                 y = y.squeeze(0).to(config['device'])    # [Cout]
                 y = y.float()

                 logits = model(Xs)        # [Nv,Cout]
                 video_logits = logits.mean(dim=0)

                 y = swap_torch(y, 20, 21)
                 video_logits = swap_torch(video_logits, 20, 21)

                 loss_v = criterion(video_logits, y)
                 val_running += float(loss_v.detach().cpu())
                 n_videos += 1

        val_loss = val_running / max(1, n_videos)
        print(f"[val]   epoch={epoch} video-level loss={val_loss:.6f}")

        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[saved best] epoch={epoch} → {best_path}")
        else:
            print("[no improvement]")

    print(f"[done] best_val_loss={best_val_loss:.6f}")
