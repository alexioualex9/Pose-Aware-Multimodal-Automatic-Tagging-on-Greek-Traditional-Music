import os
import torch.nn as nn

# Late Fusion
AUDIO_DIR = "/../audio/save_audio_probs"
VIDEO_DIR = "/../video/save_video_probs"
SKELETON_DIR = "/../skeletons/STGCN/skeleton_probs"
VIDEO_INDEX = "/../video"

# Transformer
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
VIDEO_EMBEDDINGS_DIR = os.path.join("/../video/extract_video_embeddings/embeddings/lyra")
AUDIO_EMBEDDINGS_DIR = os.path.join("/../audio",'save_audio_embs')
SKELETON_EMBEDDINGS_DIR = os.path.join("/../skeletons/STGCN/embeddings/lyra")


MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
EVALUATIONS_DIR = os.path.join(ROOT_DIR, 'evaluation')
LABELS_DIR = os.path.join(ROOT_DIR, "files_utils"  , 'labels.json')
LABELS_SUBSET_DIR = os.path.join(ROOT_DIR, "files_utils", 'labels_subset.json')
EXCLUDED_IDS = os.path.join(ROOT_DIR, "files_utils", 'missing_skeletons.txt')


LATE_FUSION_CONFIG = {
    'lyra': {
       'whole_dataset': {
          'late_fusion': {
               'top_N_tags': 30,
               'dropout': 0.1,
               'LR': 5e-4,
               'batch_size': 128,
               'epochs': 40,
               'num_workers': 0,
               'd_model': 256,
               'hidden_dim': 128,
               'use_missing_skeleton_token': True,
               'loss_function': nn.BCEWithLogitsLoss(),
               'optimizer': "AdamW",
               'weight_decay': 1e-4,
          },
       },
       'subset': {
          'late_fusion': {
               'top_N_tags': 28   ,
               'dropout': 0.2,
               'LR': 2e-2,
               'batch_size': 64,
               'epochs': 40,
               'num_workers': 0,
               'd_model': 256,
               'hidden_dim': 128,
               'use_missing_skeleton_token': True,
               'loss_function': nn.BCEWithLogitsLoss(),
               'optimizer': "AdamW",
               'weight_decay': 1e-4,
          }
       }
    }
}



MODELS_CONFIG = {
    'lyra': {
       'whole_dataset': {
          'self_attention': {
              'AV': {
                   'top_N_tags': 30,
                   'dropout': 0.1,
                   'LR': 5e-4,
                   'batch_size': 128,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 6,
                   'd_model': 384,
                   'max_clips': 200,
                   'p_drop_v_seq': 0.0,
                   'use_missing_v_token': True,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
                   #'scheduler': "cosine_decay",
              },
           },

          'gated': {
              'AV': {
                   'top_N_tags': 30,
                   'dropout': 0.1,
                   'LR': 5e-4,
                   'batch_size': 128,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 6,
                   'd_model': 384,
                   'max_clips': 200,
                   'p_drop_v_seq': 0.0,
                   'use_missing_v_token': True,
                   'lambda_gate': 0,
                   'last_gate_reg': None,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },
           },

          'cross_attention': {
              'AV': {
                   'top_N_tags': 30,
                   'dropout': 0.1,
                   'LR': 5e-4,
                   'batch_size': 128,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 6,
                   'd_model': 384,
                   'max_clips': 200,
                   'p_drop_v_seq': 0.0,
                   'alpha': 1,
                   'use_v2a': True,
                   'use_missing_v_token': True,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },
           },
       },


       'subset': {
          'self_attention': {
              'AV': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'p_drop_v_seq': 0.05,
                   'use_missing_v_token': True,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'AS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'VS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",

              },

              'AVS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'use_missing_v_token': True,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },
           },

          'gated': {
              'AV': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'p_drop_v_seq': 0.0,
                   'use_missing_v_token': True,
                   'lambda_gate': 0,
                   'last_gate_reg': None,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'AS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'VS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'AVS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 1,
                   'attn_heads': 4,
                   'd_model': 256,
                   'max_clips': 200,
                   'p_drop_v_seq': 0.0,
                   'lambda_gate': 0,
                   'use_missing_v_token': True,
                   'last_gate_reg': None,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },
           },

          'cross_attention': {
              'AV': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 4,
                   'd_model': 64,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'AS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 4,
                   'd_model': 64,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'VS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 4,
                   'd_model': 64,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

              'AVS': {
                   'top_N_tags': 28,
                   'dropout': 0.2,
                   'LR': 2e-4,
                   'batch_size': 64,
                   'epochs': 60,
                   'attn_layers': 2,
                   'attn_heads': 4,
                   'd_model': 64,
                   'early_stopping_patience': None,
                   'num_workers': 0,
                   'weight_decay': 1e-4,
                   'loss_function': nn.BCEWithLogitsLoss(),
                   'optimizer': "AdamW",
              },

           },
       },



    },
}

#              'AVS': {
#                   'top_N_tags': 28,
#                   'dropout': 0.2,
#                   'LR': 3e-5,
#                   'batch_size': 64,
#                   'epochs': 100,
#                   'attn_layers': 1,
#                   'attn_heads': 4,
#                   'd_model': 256,
#                   'max_clips': 200,
#                   'alpha': 1,
#                   'p_drop_v_seq': 0.05,
#                   'use_missing_v_token': True,
#                   'early_stopping_patience': None,
#                   'num_workers': 0,
#                   'weight_decay': 1e-4,
#                   'loss_function': nn.BCEWithLogitsLoss(),
#                   'optimizer': "AdamW",
                   #'scheduler': "cosine_decay",
#              },

