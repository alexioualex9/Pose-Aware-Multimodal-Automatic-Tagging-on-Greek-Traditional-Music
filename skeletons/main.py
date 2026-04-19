#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified skeleton pipeline (COCO17, [N,C,T,V]):

1) BUILD:  reads AlphaPose JSONs, robust-filters, selects T frames per clip,
           produces .npz per clip with:
             - data: float32 [C=6, T, V=17]  (pos xyz + vel xyz; z is conf)
             - valid_len, pad
           and writes skeleton_index.json (+ labels_28.json)

2) TRAIN/EVAL: trains an ST-GCN-style model closer to canonical ST-GCN:
           - K=3 adjacency subsets (self/same, centripetal, centrifugal)
           - edge importance weighting (learnable)
           - optional multi-scale temporal conv
           - multilabel BCEWithLogitsLoss
           - video-level evaluation by averaging logits across clips

Run:
  Build:
    python main.py cr_embeddings --model_name {STGCN} --set {train, val, test} --device {cpu, cuda, mps}

  Train:
    python main.py train --model_name {STGCN} --device {cpu, cuda, mps}

  Eval only:
    python main.py eval --model_name {STGCN} --device {cpu, cuda, mps}
"""




import argparse
from cr_embeddings import build_npz
from training import train
from evaluation import evaluate



# ============================================================
# ----------------------------- CLI --------------------------
# ============================================================

def create_parser():
    p = argparse.ArgumentParser("Unified skeleton ST-GCN pipeline (build + train + eval)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- build ----
    c_e = sub.add_parser("cr_embeddings", help="Build .npz clips with data [6,T,17] and write skeleton_index.json")
    c_e.add_argument("--model_name", default='STGCN', choices=["STGCN"],help="give model")
    c_e.add_argument("--time_window", default='3.69', choices=["3.69", "8.00"],help="defines window time")
    c_e.add_argument("--dataset", default='lyra', help="give dataset")
    c_e.add_argument("--set", default='train', choices=["train", "val", "test"])
    c_e.add_argument("--device", type=str, default="cpu")

    # ---- train ----
    t = sub.add_parser("train", help="Train ST-GCN model")

    t.add_argument("--model_name", type=str, default="STGCN", choices=["STGCN", "CTRGCN", "CTRGCN_FT"])
    t.add_argument("--time_window", default='3.69', choices=["3.69", "8.00"],help="defines window time")
    t.add_argument("--dataset", type=str, default="lyra")
    t.add_argument("--finetuning", type=bool, default=False)
    t.add_argument("--seed", type=int, default=42, choices=[42, 123, 1337, 2024, 9999])
    t.add_argument("--device", type=str, default="cpu")


    # ---- eval ----
    e = sub.add_parser("eval", help="Evaluate a saved model on an index.json (video-level)")

    e.add_argument("--model_name", type=str, default="STGCN", choices=["STGCN", "CTRGCN", "CTRGCN_FT"])
    e.add_argument("--time_window", default='3.69', choices=["3.69", "8.00"],help="defines window time")
    e.add_argument("--dataset", type=str, default="lyra")
    e.add_argument("--pool", type=str, default="topk", choices=["mean", "topk"])
    e.add_argument("--k", type=int, default=10)
    e.add_argument("--finetuning", type=bool, default=False)
    e.add_argument("--seed", type=int, default=42, choices=[42, 123, 1337, 2024, 9999])
    e.add_argument("--device", type=str, default="cpu")


    return p


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.cmd == "cr_embeddings":
        build_npz(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
