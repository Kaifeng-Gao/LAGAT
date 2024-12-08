import wandb
import time

def init_wandb(config):
    """Initialize wandb run with given config"""
    wandb.init(
        project="LAGAT",
        config=config.__dict__,
        name=f"lagat_{config.dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

def log_metrics(metrics, step, prefix=""):
    """Log metrics to wandb with optional prefix"""
    logged_metrics = {}
    for k, v in metrics.items():
        logged_metrics[f"{prefix}{k}"] = v
    wandb.log(logged_metrics, step=step)