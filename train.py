import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from tqdm import tqdm
import time

from config.config import Config
from model.lagat import LAGAT
from data.load_fraud_dataset import load_fraud_dataset
from utils.earlystopping import EarlyStopping

from evaluate import evaluate

def train_model(model, data, optimizer, device, node_type):
    model.train()
    optimizer.zero_grad()
    out = model(data.feature_dict, data.edge_index_dict, data.label_mask_dict)

    train_mask = data[node_type].train_mask.to(device)
    label = data[node_type].label.to(device)
    
    logits = out[node_type][train_mask.bool()]
    targets = label[train_mask.bool()].long()
    
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set_random_seed(Config.RANDOM_SEED)
    
    # Load dataset
    data = load_fraud_dataset(
        dataset_name=Config.DATASET_NAME,
        train_size=Config.TRAIN_SIZE,
        val_size=Config.VAL_SIZE,
        random_seed=Config.RANDOM_SEED,
        force_reload=Config.FORCE_RELOAD
    )
    
    # Initialize model
    model = LAGAT(
        in_channels=Config.IN_CHANNELS,
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_layers=Config.NUM_LAYERS,
        out_channels=Config.OUT_CHANNELS,
        num_labels=Config.NUM_LABELS,
        label_embedding_dim=Config.LABEL_EMBEDDING_DIM,
        heads=Config.HEADS,
        dropout=Config.DROPOUT
    )
    
    model = to_hetero(model, data.metadata(), aggr='sum')
    model = model.to(device)
    data = data.to(device)
    
    # Training loop
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    early_stopper = EarlyStopping(
        dataset_name=Config.DATASET_NAME,
        timestamp=time.strftime("%Y%m%d-%H%M%S"),
        patience=Config.TOLERATION
    )
    
    progress_bar = tqdm(range(Config.EPOCHS), desc='Training')
    for epoch in progress_bar:
        train_loss = train_model(model, data, optimizer, device, Config.TARGET_NODE_TYPE)
        # Evaluate model and get metrics for all splits
        eval_results = evaluate(model, data, Config.TARGET_NODE_TYPE)
        val_metrics = eval_results['val']
        val_loss = val_metrics.get('loss', 0)
        val_f1 = val_metrics.get('f1', 0)
        val_auc = val_metrics.get('auc', 0)
        val_ap = val_metrics.get('ap', 0)

        # Using AP (Average Precision) for early stopping
        if early_stopper.step(epoch, val_loss, val_metrics['ap'], model):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        progress_bar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val F1': f'{val_f1:.4f}',
            'Val AUC': f'{val_auc:.4f}',
            'Val AP': f'{val_ap:.4f}',
            'Best Val AP': f'{early_stopper.best_result:.4f}'
        })
    
    early_stopper.load_checkpoint(model)
    eval_results = evaluate(model, data, Config.TARGET_NODE_TYPE)
    print(eval_results)

if __name__ == "__main__":
    main()