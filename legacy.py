# First Version LAGAT Layer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset, zeros
from torch_geometric.utils import (add_remaining_self_loops, add_self_loops,
                                   remove_self_loops, softmax)
from torch_scatter import scatter_add


class LAGATConvLayer(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_labels,
                 label_embedding_dim,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=False,
                 **kwargs):
        super(LAGATConvLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        # add label related params
        self.num_labels = num_labels
        self.label_embedding_dim = label_embedding_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        # Add learnable label embedding 
        self.label_embs = Parameter(torch.Tensor(self.num_labels, self.label_embedding_dim))

        # add Label Embedding into attention calculation
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + self.label_embedding_dim))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, label_mask, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                None if x[1] is None else torch.matmul(x[1], self.weight))

        # print(f"x: {x.shape}, edge_index: {edge_index.shape}, label_mask: {label_mask.shape}")
        return self.propagate(edge_index, size=size, x=x, label_mask=label_mask)

    def message(self, edge_index_i, edge_index_j, x_i, x_j, size_i, label_mask):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        # index and expanded label_emb to be concatenated
        loop_edge_mask = edge_index_i == edge_index_j
        label_j = label_mask[edge_index_j]
        # Replace labels for loop edges with 0 (index for self.label_embs[0])
        label_j = torch.where(loop_edge_mask, torch.zeros_like(label_j), label_j)
        label_emb = self.label_embs[label_j]
        label_emb = label_emb.unsqueeze(1).repeat(1, self.heads, 1)
        if x_i is None:
            alpha = (torch.cat([x_j, label_emb], dim=-1) * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j, label_emb], dim=-1) * self.att).sum(dim=-1)


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads*self.out_channels)
        output = (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads*self.out_channels)
        # print(f"x_i: {x_i.shape}, x_j: {x_j.shape}, label_mask: {label_mask.shape}, alpha: {alpha.shape}, output: {output.shape}")
        return output

    def update(self, aggr_out):
        # if self.concat is True:
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:
        #     aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    

# Second Version LAGAT Layer
class LAGATConvLayer(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_labels,
                 label_embedding_dim,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=False,
                 **kwargs):
        super(LAGATConvLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        # add label related params
        self.num_labels = num_labels
        self.label_embedding_dim = label_embedding_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        # Add learnable label embedding 
        self.label_embs = Parameter(torch.Tensor(num_labels, 1, label_embedding_dim))

        # add Label Embedding into attention calculation
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_lab = Parameter(torch.Tensor(1, heads, label_embedding_dim))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_dst)
        glorot(self.att_src)
        glorot(self.att_lab)
        glorot(self.label_embs)
        zeros(self.bias)

    def forward(self, x, edge_index, label_mask, size=None):
        H, C = self.heads, self.out_channels
        
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                        num_nodes=x.size(self.node_dim))
        
        if torch.is_tensor(x):
            x_src = x_dst = torch.matmul(x, self.weight).view(-1, H, C)
        else:
            x_src, x_dst = x
            x_src = torch.matmul(x_src, self.weight).view(-1, H, C)
            if x_dst is not None:
                x_dst = torch.matmul(x_dst, self.weight).view(-1, H, C)

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha_lab = (self.label_embs * self.att_lab).sum(-1)
        alpha = (alpha_src, alpha_dst)

        x = (x_src.view(-1, H*C), x_dst.view(-1, H*C))
        
        return self.propagate(edge_index, size=size, x=x, 
                            alpha=alpha, alpha_lab=alpha_lab, label_mask=label_mask)

    def message(self, edge_index_i, edge_index_j, x_j, alpha_i, 
                alpha_j, size_i, label_mask, alpha_lab):
        H, C = self.heads, self.out_channels
        
        # Initialize alpha
        if alpha_i is not None:
            alpha = alpha_j + alpha_i
        else:
            alpha = alpha_j
        
        # Handle label attention
        loop_edge_mask = edge_index_i == edge_index_j
        label_j = torch.index_select(label_mask, 0, edge_index_j)
        label_j = torch.where(loop_edge_mask, torch.zeros_like(label_j), label_j)
        
        alpha_label = torch.index_select(alpha_lab, 0, label_j)
        alpha = alpha + alpha_label
        
        # Apply attention mechanisms
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Ensure correct shapes for final multiplication
        x_j = x_j.view(-1, H, C)
        alpha = alpha.view(-1, H, 1)
        
        return (x_j * alpha).view(-1, H * C)

    def update(self, aggr_out):
        # if self.concat is True:
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:
        #     aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)