import torch
from video_scores import collect_video_scores_avs, collect_video_scores_av, collect_video_scores_as, collect_video_scores_vs
from metrics.global_metrics import compute_global_metrics
from utils import plot_macro_roc_av

# ------------------------------------------------------------
# Eval-only
# ------------------------------------------------------------
def evaluate_transformer(test_loader, labels, model, model_name, device, subset):

     if model_name == "seq_transformer_avs_masked":
         _, Y, S = collect_video_scores_avs(model, test_loader, device, subset)
     elif model_name == "seq_transformer_as_masked":
         _, Y, S = collect_video_scores_as(model, test_loader, device, subset)
     elif model_name == "seq_transformer_vs_masked":
         _, Y, S = collect_video_scores_vs(model, test_loader, device, subset)
     else:
         _, Y, S = collect_video_scores_av(model, test_loader, device, subset)

     m = compute_global_metrics(Y, S, threshold=0.5, label_names=labels)

     auc_macro, auc_per_class = plot_macro_roc_av(
         Y, S,
         title="AV (macro-average ROC)",
         save_path="roc_av_macro.png"  # προαιρετικό
     )

     return m


def evaluate_transf(test_loader, labels, model, model_name, device, subset):

     if model_name == "seq_transformer_avs_masked":
         _, Y, S = collect_video_scores_avs(model, test_loader, device, subset)
     elif model_name == "seq_transformer_as_masked":
         _, Y, S = collect_video_scores_as(model, test_loader, device, subset)
     elif model_name == "seq_transformer_vs_masked":
         _, Y, S = collect_video_scores_vs(model, test_loader, device, subset)
     else:
         _, Y, S = collect_video_scores_av(model, test_loader, device, subset)

     m = compute_global_metrics(Y, S, threshold=0.5, label_names=labels)


     return Y,S,m
