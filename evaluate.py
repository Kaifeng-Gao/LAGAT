import torch
import torch.nn.functional as F
from utils.metrics import calculate_metrics

def evaluate(model, data, node_type):
    model.eval()
    losses = {}
    with torch.no_grad():
        out = model(data.feature_dict, data.edge_index_dict, data.label_mask_dict)
        scores = torch.softmax(out[node_type], dim=1)
        def calculate_split_loss(mask):
            mask_indices = mask.cpu().bool()
            split_out = out[node_type][mask_indices]
            split_labels = data[node_type].label[mask_indices]
            criterion = torch.nn.CrossEntropyLoss()
            return criterion(split_out, split_labels).item()
        
        # Calculate losses for each split
        losses['train'] = calculate_split_loss(data[node_type].train_mask)
        losses['val'] = calculate_split_loss(data[node_type].val_mask)
        losses['test'] = calculate_split_loss(data[node_type].test_mask)
    
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
        'train': {
            'loss': losses['train'],
            'f1': train_metrics[0],
            'auc': train_metrics[1],
            'ap': train_metrics[2]
        },
        'val': {
            'loss': losses['val'],
            'f1': val_metrics[0],
            'auc': val_metrics[1],
            'ap': val_metrics[2]
        },
        'test': {
            'loss': losses['test'],
            'f1': test_metrics[0],
            'auc': test_metrics[1],
            'ap': test_metrics[2]
        }
    }

if __name__ == "__main__":
    pass