import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)



def safe_auc(fn, Y, S, average: str):
    try:
        return fn(Y, S, average=average)
    except Exception:
        return float("nan")


def compute_global_metrics(Y, S, threshold, label_names):
    P = (S >= threshold).astype(np.int32)

    roc_micro = safe_auc(roc_auc_score, Y, S, average="micro")
    roc_macro = safe_auc(roc_auc_score, Y, S, average="macro")
    pr_micro  = safe_auc(average_precision_score, Y, S, average="micro")
    pr_macro  = safe_auc(average_precision_score, Y, S, average="macro")

    f1_micro = f1_score(Y, P, average="micro", zero_division=0)
    f1_macro = f1_score(Y, P, average="macro", zero_division=0)

    report = classification_report(Y, P, target_names=label_names, zero_division=0, digits=2)

    return dict(
        report=report,
        roc_micro=roc_micro, roc_macro=roc_macro,
        pr_micro=pr_micro, pr_macro=pr_macro,
        f1_micro=f1_micro, f1_macro=f1_macro,
    )


# Print test results
def test_results(m, model_name=None, dataset=None, split="test",
                 auc_kind="macro", decimals=2):
    """
    m: dict με keys όπως: report, roc_micro, roc_macro, pr_micro, pr_macro
    auc_kind: "macro" ή "micro" -> ποιο AUC να τυπώσει (στο παράδειγμα σου είναι macro)
    """

    if model_name is not None and dataset is not None:
        print(f'\nEvaluation of model "{model_name}" on "{dataset}" {split} set:')
    else:
        # fallback αν δεν δώσεις ονόματα
        print(f"\nEvaluation results ({split} set):")

    roc_key = f"roc_{auc_kind}"
    pr_key  = f"pr_{auc_kind}"

    if roc_key in m:
        print(f"ROC-AUC score: {m[roc_key]}")
    if pr_key in m:
        print(f"PR-AUC score: {m[pr_key]}")
    print()

    # Μορφοποίηση classification report σε 2 δεκαδικά όπως στο παράδειγμα
    report = m.get("report", "")
    if isinstance(report, str) and report:
        # sklearn classification_report έχει "0.7500" κτλ.
        # το κάνουμε "0.75" και κρατάμε στοίχιση όσο γίνεται.
        import re
        def _fmt(match):
            return f"{float(match.group(0)):.{decimals}f}"
        report = re.sub(r"\d+\.\d{4,}", _fmt, report)

    print(report)