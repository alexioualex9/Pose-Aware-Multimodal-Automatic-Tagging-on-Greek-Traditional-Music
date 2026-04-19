import os, sys

from STGCN_model import STGCNModel
from head_classifiers.mlp_head import MLPHead


def return_model(model_name: str, C_out: int, C_in: int, config: dict):
    """
    model_name:
      - "stgcn"  -> STGCNModel
      - "ctrgcn_emb" -> head πάνω σε embeddings (π.χ. 256-dim)
    """
    return STGCNModel(
         num_class=C_out,
         in_channels=C_in,
         hidden_channels=config['hidden_channels'],
         num_layers=config['num_layers'],
         use_edge_importance=config['no_edge_importance'],
         multi_scale_tcn=config['multi_scale_tcn'],
         root=config['root'],
    ).to(config['device'])