import torch
import torch.nn.functional as F
from utils.metrics import calculate_metrics

def evaluate(model, data, node_type):
    model.eval()
    with torch.no_grad():
        out = model(data.feature_dict, data.edge_index_dict, data.label_mask_dict)
        scores = torch.softmax(out[node_type], dim=1)
    
    labels = data[node_type].label.cpu()
    pred = scores.argmax(dim=1).cpu()
    
    def evaluate_split(mask):
        mask_indices = mask.cpu().bool()
        split_labels = labels[mask_indices]
        split_pred = pred[mask_indices]
        split_scores = scores[mask_indices][:, 1].cpu()
        return calculate_metrics(split_labels, split_pred, split_scores)
    
    train_metrics = evaluate_split(data[node_type].train_mask)
    val_metrics = evaluate_split(data[node_type].val_mask)
    test_metrics = evaluate_split(data[node_type].test_mask)
    
    return {
        'train': dict(zip(['f1', 'auc', 'ap'], train_metrics)),
        'val': dict(zip(['f1', 'auc', 'ap'], val_metrics)),
        'test': dict(zip(['f1', 'auc', 'ap'], test_metrics))
    }

if __name__ == "__main__":
    # Code to load model and run evaluation
    pass