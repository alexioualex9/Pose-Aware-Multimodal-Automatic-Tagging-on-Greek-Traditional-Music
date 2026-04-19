# Print test results
def test_results(m, audio_model_name=None, video_model_name=None, skeleton_model_name=None, dataset=None, modalities=None, split="test",
                 auc_kind="macro", decimals=2):
    """
    m: dict με keys όπως: report, roc_micro, roc_macro, pr_micro, pr_macro
    auc_kind: "macro" ή "micro" -> ποιο AUC να τυπώσει (στο παράδειγμα σου είναι macro)
    """

    if dataset is not None:
       if 'a' in modalities and 'v' not in modalities and 's' not in modalities:
           print(f'\nEvaluation of model "{audio_model_name}" on "{dataset}" {split} set:')
       elif 'a' not in modalities and 'v' in modalities and 's' not in modalities:
           print(f'\nEvaluation of model "{video_model_name}" on "{dataset}" {split} set:')
       elif 'a' not in modalities and 'v' not in modalities and 's' in modalities:
           print(f'\nEvaluation of model "{skeleton_model_name}" on "{dataset}" {split} set:')
       elif 'a' in modalities and 'v' in modalities and 's' not in modalities:
           print(f'\nEvaluation of model "{audio_model_name}" + "{video_model_name}" on "{dataset}" {split} set:')
       elif 'a' not in modalities and 'v' in modalities and 's' in modalities:
           print(f'\nEvaluation of model "{video_model_name}" + "{skeleton_model_name}" on "{dataset}" {split} set:')
       elif 'a' in modalities and 'v' not in modalities and 's' in modalities:
           print(f'\nEvaluation of model "{audio_model_name}" + "{skeleton_model_name}" on "{dataset}" {split} set:')
       else:
           print(f'\nEvaluation of model "{audio_model_name}" + "{video_model_name}" + "{skeleton_model_name}" on "{dataset}" {split} set:')

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
