import torch
import numpy as np

# Save metrics in a .txt file
def save_metrics(metrics, out_dir, model_name, dataset, split):
    #out = Path(out_dir)
    #out.mkdir(parents=True, exist_ok=True)

    # TXT classification report
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(f'Evaluation of model "{model_name}" on "{dataset}" {split} set:\n')
        f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
        f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
        f.write(f'F1 score: {metrics["f1_macro"]}\n')
       # f.write(metrics["report"])


def load_embed_stats(stats_path: str):
    z = np.load(stats_path)
    mu = z["mu"].astype(np.float32)
    std = z["std"].astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mu, std



def collate_videos(batch):
    # batch: list of (Xs [Nv,D], y [Cout], vid)
    Xs_list, y_list, vids = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in Xs_list], dtype=torch.long)

    D = Xs_list[0].size(1)
    Tmax = int(lengths.max().item())

    Xpad = torch.zeros((len(batch), Tmax, D), dtype=Xs_list[0].dtype)
    for i, x in enumerate(Xs_list):
        Xpad[i, :x.size(0)] = x

    Y = torch.stack(y_list, dim=0)  # [B, Cout]
    return Xpad, lengths, Y, list(vids)
