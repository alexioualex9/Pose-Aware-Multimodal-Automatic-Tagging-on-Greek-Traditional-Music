#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build subset indices (e.g. index_danced_28.json) for a given model_name.

Filtering rules:
1) keep only videos with is-danced == 1 from raw.tsv (covers train/val/test)
2) remove videos listed in exclude_ids_txt (optional)
3) remap labels from labels.json (30) -> labels_subset.json (28)
4) write new index file per split: train/val/test
"""

import argparse
import json
import os
from pathlib import Path
from typing import Set, List, Dict, Any, Optional

import pandas as pd

from config import EMBEDDINGS_DIR, FINETUNED_EMBEDDINGS_DIR, EXCLUDED_IDS, LABELS_DIR, LABELS_SUBSET_DIR
from utils import load_exclude_ids

# -----------------------------
# Helpers
# -----------------------------
def _load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["labels"]


def build_allowed_ids(raw_tsv: str, exclude_ids_txt: Optional[str] = None) -> Set[str]:
    df = pd.read_csv(raw_tsv, sep="\t", keep_default_na=False)

    # normalize danced values to int(0/1)
    danced_mask = df["is-danced"] == 1
    danced_ids = set(df.loc[danced_mask, "id"].astype(str))

    if exclude_ids_txt:
        exclude_ids = load_exclude_ids(exclude_ids_txt)

    return danced_ids - exclude_ids


def build_subset_index_for_split(
    index_in: str,
    index_out: str,
    full_labels: List[str],
    subset_labels: List[str],
    allowed_ids: Set[str],
) -> Dict[str, Any]:
    with open(index_in, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)

    full_set = set(full_labels)
    subset_set = set(subset_labels)

    dropped = [lbl for lbl in full_labels if lbl not in subset_set]
    if len(subset_labels) == 0:
        raise RuntimeError("labels_subset.json is empty.")

    # build mapping subset_labels -> indices in full_labels
    try:
        keep_idx = [full_labels.index(lbl) for lbl in subset_labels]
    except ValueError as e:
        raise RuntimeError(
            "labels_subset contains a label not present in full labels.json."
        ) from e

    new_entries: List[Dict[str, Any]] = []
    kept_clips = 0
    removed_clips = 0

    for e in entries:
        vid = str(e.get("video_id", ""))

        if vid not in allowed_ids:
            removed_clips += 1
            continue

        old_y = e.get("labels", None)
        if old_y is None:
            raise RuntimeError(f"Entry missing 'labels': {e}")

        if len(old_y) != len(full_labels):
            raise RuntimeError(
                f"Label length mismatch for video_id={vid}. "
                f"Entry has {len(old_y)} labels but full_labels has {len(full_labels)}."
            )

        new_y = [float(old_y[i]) for i in keep_idx]

        e2 = dict(e)
        e2["labels"] = new_y
        e2["C"] = len(subset_labels)
        new_entries.append(e2)
        kept_clips += 1

    Path(index_out).parent.mkdir(parents=True, exist_ok=True)
    with open(index_out, "w", encoding="utf-8") as f:
        json.dump(new_entries, f, indent=2)

    return {
        "index_in": index_in,
        "index_out": index_out,
        "clips_in": len(entries),
        "clips_out": len(new_entries),
        "removed_clips": removed_clips,
        "kept_clips": kept_clips,
        "dropped_labels": dropped,
        "C_full": len(full_labels),
        "C_subset": len(subset_labels),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build danced subset indices (28 labels) per split for a given model."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="lyra",
        help="Dataset name (folder under EMBEDDINGS_DIR). Default: lyra",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["slowfast50", "timesformer", "r21d", "resnet50", "vitb16", "videomae"],
        help="Model folder under embeddings/<dataset>/",
    )
    ap.add_argument(
        "--raw_tsv",
        type=str,
        required=True,
        help="Path to raw.tsv containing all ids and is-danced column.",
    )
    ap.add_argument(
        "--embs",
        type=str,
        required=True,
        default="frozen",
        choices=["frozen", "finetuned"],
        help="Path to raw.tsv containing all ids and is-danced column.",
    )

    ap.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to process. Default: train,val,test",
    )
    ap.add_argument(
        "--in_name",
        type=str,
        default="index.json",
        help="Input index filename inside each split folder. Default: index.json",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="index_danced_28.json",
        help="Output subset index filename inside each split folder. Default: index_danced_28.json",
    )
    args = ap.parse_args()


    exclude_ids_txt = EXCLUDED_IDS

    # Load labels
    full_labels = _load_labels(LABELS_DIR)
    subset_labels = _load_labels(LABELS_SUBSET_DIR)

    # Compute allowed ids (danced & not excluded)
    allowed_ids = build_allowed_ids(args.raw_tsv, exclude_ids_txt=exclude_ids_txt)

    # Resolve index paths from model_name and dataset
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise RuntimeError("No splits provided.")

    print(f"[info] dataset={args.dataset} model_name={args.model_name}")
    print(f"[info] raw_tsv={args.raw_tsv}")
    print(f"[info] exclude_ids_txt={exclude_ids_txt}")
    print(f"[info] allowed_ids={len(allowed_ids)}")
    print(f"[info] labels: full={len(full_labels)} subset={len(subset_labels)}")

    for split in splits:
        if args.embs == "frozen":
           split_dir = os.path.join(EMBEDDINGS_DIR, args.dataset, args.model_name, split)
        else:
           split_dir = os.path.join(FINETUNED_EMBEDDINGS_DIR, args.dataset, args.model_name, split)
        index_in = os.path.join(split_dir, args.in_name)
        index_out = os.path.join(split_dir, args.out_name)

        if not os.path.exists(index_in):
            print(f"[warn] Missing {index_in}, skipping split={split}")
            continue

        stats = build_subset_index_for_split(
            index_in=index_in,
            index_out=index_out,
            full_labels=full_labels,
            subset_labels=subset_labels,
            allowed_ids=allowed_ids,
        )

        print(
            f"[done] split={split} | {stats['clips_in']} -> {stats['clips_out']} clips "
            f"| out={stats['index_out']}"
        )

    # Summary of dropped labels
    dropped = [lbl for lbl in full_labels if lbl not in set(subset_labels)]
    if dropped:
        print(f"[info] dropped labels ({len(dropped)}): {dropped}")


if __name__ == "__main__":
    main()
