from data.fraud_dataset import FraudDataset
from torch_geometric.utils import from_dgl
import torch


def load_fraud_dataset(dataset_name, train_size, val_size, random_seed, force_reload, observed_ratio):
    fraud_data = FraudDataset(
        dataset_name,
        train_size=train_size,
        val_size=val_size,
        random_seed=random_seed,
        force_reload=force_reload
    )
    graph = fraud_data[0]
    data = from_dgl(graph)
    mask_label(data, observed_pct=observed_ratio)
    return data


def mask_label(data, observed_pct=1):
    # Ensure observed_pct is a value between 0 and 1
    assert 0 <= observed_pct <= 1, "observed_pct must be between 0 and 1"
    
    # Initialize the dictionary to store label masks for each node type
    unknown_encoding = -1
    
    # Create label masks for each node type
    for node_type in data.node_types:
        if hasattr(data[node_type], 'label'):
            # Create a copy of the labels to modify
            label_mask = data[node_type].label.clone()
            
            # Mask all validation and test labels if they exist
            if hasattr(data[node_type], 'val_mask'):
                label_mask[data[node_type].val_mask.bool()] = unknown_encoding
            if hasattr(data[node_type], 'test_mask'):
                label_mask[data[node_type].test_mask.bool()] = unknown_encoding
            
            # Handle training mask
            if hasattr(data[node_type], 'train_mask'):
                # Identify the indices of the training data
                train_indices = data[node_type].train_mask.nonzero(as_tuple=False).squeeze()
                
                # Calculate the number of training labels to mask
                num_train_labels = train_indices.size(0)
                num_to_mask = int((1 - observed_pct) * num_train_labels)
                
                # Randomly select indices to mask
                mask_indices = train_indices[torch.randperm(num_train_labels)[:num_to_mask]]
                label_mask[mask_indices] = unknown_encoding
            
            # Add 1 to shift from -1,0,1 to 0,1,2 encoding
            label_mask = label_mask + 1
            data[node_type].label_mask = label_mask