import os
import torch.nn as nn

# dir paths used in several places
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = "/../DATA/SKELETONS"
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
EVALUATIONS_DIR = os.path.join(ROOT_DIR, 'evaluation')
LABELS_DIR = os.path.join(ROOT_DIR, 'labels.json')
LABELS_SUBSET_DIR = os.path.join(ROOT_DIR, 'labels_subset.json')
OUT_DIR = os.path.join(ROOT_DIR, "skeleton_embeddings")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
VIDEO_DIR =  "/../Unimodals/Video/"


SKELETONS_CONFIG = {
    'lyra': {
           'T': 32,
           'max_skeletons' : 32,
           'conf_thresh': 0.35,
           'min_valid_joints': 12,
           'min_scale': 0.15,
           'min_shoulder_conf': 0.35,
           'min_shoulder_dist_abs': 12,
           'max_abs_xy': 200.0,
           'min_clip_frames': 30,
           'sim_tukey_k': 0.2,
           'y_weight': 1.2,
           'sim_min_keep': 8,
           'interp_max_gap': 3,
           'w_bone': 2.0,
           'w_jitter': 2,
           'w_lr': 1.0,
           'already_centered': True,
           'do_scale': False,
     },
}


MODELS_CONFIG = {
    'lyra': {
        'head_lr': 5e-4,
        'encoder_lr': 1e-5,
        'warmup_epochs': 3,
        'grad_clip': 1.0,
        'dropout': 0.2,


        'epochs': 30,
        'batch_size' : 32,
        'lr': 3e-4,
        'weight_decay': 1e-2,
        'C_in': 6,
        'hidden_channels': 64,
        'num_layers': 6,
        'min_valid': 5,
        'no_edge_importance': False,
        'multi_scale_tcn': True,
        'root': 11,
        'loss': nn.BCEWithLogitsLoss(),
        'optimizer': "AdamW",


        'ctrgcn_cin': 512,
     },
}
