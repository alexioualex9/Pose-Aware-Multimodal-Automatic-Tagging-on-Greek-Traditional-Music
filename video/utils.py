import json
from pathlib import Path

# Load Videos that we should exclude because all skeletons were filtered out 
def load_exclude_ids(path: str):
    if path is None:
        return None
    ids = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    print(f"[info] Loaded {len(ids)} video_ids to exclude from {path}")
    return ids


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
        #f.write(metrics["report"])
