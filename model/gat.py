import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, heads=1, dropout=0.6):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, 
                   dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, 
                       heads=heads, dropout=dropout, concat=True)
            )
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, 
                       heads=1, concat=False)
            )

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.convs[i](x, edge_index))
            
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        return x