# class Config:
#     DATASET_NAME = "yelp"
#     TARGET_NODE_TYPE = "review"
#     TRAIN_SIZE = 0.4
#     VAL_SIZE = 0.1
#     RANDOM_SEED = 42
#     FORCE_RELOAD = False
    
#     EPOCHS = 10000
#     TOLERATION = 10
    
#     # Model parameters
#     IN_CHANNELS = 32
#     HIDDEN_CHANNELS = 32
#     LABEL_EMBEDDING_DIM = 32
#     NUM_LAYERS = 2
#     NUM_LABELS = 3
#     OUT_CHANNELS = 2
#     HEADS = 2
#     DROPOUT = 0.6
    
#     # Training parameters
#     LEARNING_RATE = 0.005
#     WEIGHT_DECAY = 5e-4


from dataclasses import dataclass
from typing import List
import itertools

PARAM_GRID = {
    "dataset_config": [
        # dataset_name, target_node_type, in_channels
        ("yelp", "review", 32),
        # ("amazon", "user", 25)
    ],
    "train_size": [0.6],
    "val_size": [0.1],
    "random_seed": [42, 43, 44],
    "force_reload": [False],
    "epochs": [10000],
    "toleration": [10],
    "learning_rate": [0.005],
    "weight_decay": [5e-4],
    "hidden_channels": [32],
    "label_embedding_dim": [32],
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