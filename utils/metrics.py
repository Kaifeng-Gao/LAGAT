from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch

def calculate_metrics(labels, pred, scores):
    f1 = f1_score(labels, pred, average='macro')
    try:
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    except Exception as e:
        print(f"Warning in metrics calculation: {e}")
        auc, ap = float('nan'), float('nan')
    return f1, auc, ap