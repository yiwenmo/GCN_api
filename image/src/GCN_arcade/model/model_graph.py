"""
This script defines GNN models along with training and testing functions for graph-based classification tasks.

GCN is a type of neural network designed to operate on graph-structured data, while GAT is an extension of GCN that incorporates attention mechanisms.

Author: MO, YI WEN
Date: 2024/4/8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv, global_mean_pool
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch_geometric.explain import Explainer, GraphMaskExplainer


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, normalize=True)
        self.conv2 = SAGEConv(h_feats, h_feats, normalize=True)
        self.lin = Linear(h_feats, out_feats)
        

    def forward(self, x, edge_index, batch):
            # Apply dropout to the input features
            x = F.dropout(x, p=0.6, training=self.training)
            # Apply the first GraphSAGE layer with ReLU activation
            x = self.conv1(x, edge_index).relu()
            # Apply the second GraphSAGE layer
            x = self.conv2(x, edge_index).relu()
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
            return x

class GCN(torch.nn.Module):
    """Graph Convolutional Network (GCN) model."""
    def __init__(self, in_feats, h_feats, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, h_feats)
        self.conv3 = GCNConv(h_feats, h_feats)
        self.lin = Linear(h_feats, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GAT(torch.nn.Module):
    """Graph Attention Network (GAT) model."""
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False)
        self.conv2 = GATConv(h_feats, h_feats, heads=8, concat=False)
        self.lin = Linear(h_feats, out_feats)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.lin(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out= model(data.x, data.edge_index, data.batch) # 2D tensor 
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out, 1)
        print(predicted)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
        total_loss += loss.item()
        
    train_acc = correct / total
    train_epoch_loss = total_loss / len(train_loader)
    return train_acc, train_epoch_loss


def test(model, test_loader, criterion, device):
    """Evaluate the performance of the given model on test data."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            _, predicted = torch.max(out, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
            total_loss += loss.item()
    test_acc = correct / total
    test_epoch_loss = total_loss / len(test_loader)
    return test_acc, test_epoch_loss




