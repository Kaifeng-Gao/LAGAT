from dataclasses import dataclass
from typing import List
import itertools

PARAM_GRID = {
    "dataset_config": [
        # dataset_name, target_node_type, in_channels
        # ("yelp", "review", 32),
        ("amazon", "user", 25)
    ],
    "model_name": [
        "LAGAT", 
        "GAT"
    ],
    "train_size": [0.6],
    "val_size": [0.2],
    "random_seed": [42, 43, 44],
    "force_reload": [False],
    "epochs": [10000],
    "toleration": [300],
    "learning_rate": [0.005],
    "weight_decay": [5e-4],
    "hidden_channels": [32],
    "label_embedding_dim": [4],
    "num_layers": [2],
    "num_labels": [3],
    "out_channels": [2],
    "heads": [2],
    "dropout": [0.6]
}

@dataclass
class Config:
    # Dataset parameters
    dataset_name: str
    target_node_type: str
    train_size: float
    val_size: float
    random_seed: int
    force_reload: bool
    
    # Training parameters
    epochs: int
    toleration: int
    learning_rate: float
    weight_decay: float
    
    # Model parameters
    model_name: str
    in_channels: int
    hidden_channels: int
    label_embedding_dim: int
    num_layers: int
    num_labels: int
    out_channels: int
    heads: int
    dropout: float


def create_grid_search_configs(param_grid: dict) -> List[Config]:
    """Create configs for all combinations of parameters in param_grid"""
    # Get the dataset configurations
    dataset_configs = param_grid["dataset_config"]
    
    # Remove dataset_config from param_grid
    other_params = {k: v for k, v in param_grid.items() if k != "dataset_config"}
    
    # Generate all combinations of other parameters
    other_keys = other_params.keys()
    other_values = other_params.values()
    other_combinations = list(itertools.product(*other_values))
    
    configs = []
    
    # For each dataset configuration
    for dataset_name, target_node_type, in_channels in dataset_configs:
        # For each combination of other parameters
        for other_combo in other_combinations:
            # Create a config dictionary
            config_dict = dict(zip(other_keys, other_combo))
            # Add the dataset-specific parameters
            config_dict.update({
                "dataset_name": dataset_name,
                "target_node_type": target_node_type,
                "in_channels": in_channels
            })
            # Create and append the Config object
            configs.append(Config(**config_dict))
    
    return configs