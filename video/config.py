import os
import torch.nn as nn

# dir paths used in several places

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
EMBEDDINGS_DIR = os.path.join(ROOT_DIR,'extract_video_embeddings/embeddings')
DATA_DIR = "/data/datasets/mir_datasets/lyra"
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
EVALUATIONS_DIR = os.path.join(ROOT_DIR, 'evaluation')
LABELS_DIR = os.path.join(ROOT_DIR, "files_utils"  , 'labels.json')
LABELS_SUBSET_DIR = os.path.join(ROOT_DIR, "files_utils", 'labels_subset.json')
EXCLUDED_IDS = os.path.join(ROOT_DIR, "files_utils", 'missing_skeletons.txt')


MODELS_CONFIG = {
    'lyra': {
            'top_N_tags': 30,
            'mlp_hidden': 1024,
            'dropout': 0.3,
            'LR': 1e-3,
            'epochs': 200,
            'batch_size': 512,
            'validation_size': 0.1,
            'early_stopping_patience': 8,
            'weight_decay': 1e-4,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer': "AdamW",
            'scheduler': "cosine_decay",
    },
}
