import torch.nn.functional as F
from model.lagatconv import LAGATConv
import torch


class LAGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_labels, 
                 label_embedding_dim, num_layers=2, heads=1, dropout=0.6):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            LAGATConv(in_channels, hidden_channels, num_labels, 
                     label_embedding_dim, heads=heads, concat=True, 
                     dropout=dropout, bias=False)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                LAGATConv(hidden_channels * heads, hidden_channels, num_labels,
                         label_embedding_dim, heads=heads, concat=True,
                         dropout=dropout, bias=False)
            )
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                LAGATConv(hidden_channels * heads, out_channels, num_labels,
                         label_embedding_dim, heads=1, concat=True,
                         bias=False)
            )

    def forward(self, x, edge_index, label_index):
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.convs[i](x, edge_index, label_index))
            
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, label_index)
        
        return x