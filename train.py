import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from tqdm import tqdm
import wandb
import time

from config.config import PARAM_GRID, create_grid_search_configs
from model.lagat import LAGAT
from model.gat import GAT
from data.load_fraud_dataset import load_fraud_dataset
from utils.earlystopping import EarlyStopping
from utils.logging import init_wandb, log_metrics

from evaluate import evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_wandb', action='store_true', 
                    help='Use Weights & Biases for logging')
args = parser.parse_args()

def train_model(model, data, optimizer, device, node_type, is_lagat):
    model.train()
    optimizer.zero_grad()
    if is_lagat:
        out = model(data.feature_dict, data.edge_index_dict, data.label_mask_dict)
    else:
        out = model(data.feature_dict, data.edge_index_dict)

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
    configs = create_grid_search_configs(PARAM_GRID)
    for config in configs:
        if args.use_wandb:
            init_wandb(config)
        # Load dataset
        data = load_fraud_dataset(
            dataset_name=config.dataset_name,
            train_size=config.train_size,
            val_size=config.val_size,
            random_seed=config.random_seed,
            force_reload=config.force_reload
        )
        
        # Initialize model
        if config.model_name == "LAGAT":
            is_lagat = True
            model = LAGAT(
                in_channels=config.in_channels,
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
                out_channels=config.out_channels,
                num_labels=config.num_labels,
                label_embedding_dim=config.label_embedding_dim,
                heads=config.heads,
                dropout=config.dropout
            )
        elif config.model_name == "GAT":
            is_lagat = False
            model = GAT(
                in_channels=config.in_channels,
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
                out_channels=config.out_channels,
                heads=config.heads,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown model name: {config.model_name}")
        
        model = to_hetero(model, data.metadata(), aggr='sum')
        model = model.to(device)
        data = data.to(device)
        
        # Training loop
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        early_stopper = EarlyStopping(
            dataset_name=config.dataset_name,
            timestamp=time.strftime("%Y%m%d-%H%M%S"),
            patience=config.toleration
        )
        
        progress_bar = tqdm(range(config.epochs), desc='Training')
        for epoch in progress_bar:
            train_loss = train_model(model, data, optimizer, device, config.target_node_type, is_lagat)
            # Evaluate model and get metrics for all splits
            eval_results = evaluate(model, data, config.target_node_type, is_lagat)

            # Log training metrics
            if args.use_wandb:
                log_metrics(eval_results['train'], step=epoch, prefix='train/')
            
            # Log validation metrics
            if args.use_wandb:
                log_metrics(eval_results['val'], step=epoch, prefix='val/')

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
        eval_results = evaluate(model, data, config.target_node_type, is_lagat)
        
        if args.use_wandb:
            # update summary metrics with the best model
            wandb.run.summary['test_ap'] = eval_results['test']['ap']
            wandb.run.summary['test_auc'] = eval_results['test']['auc']
            wandb.run.summary['test_f1'] = eval_results['test']['f1']
            wandb.finish()

if __name__ == "__main__":
    main()