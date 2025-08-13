"""
This script defines GNN models along with training and testing functions for node-based classification tasks.

GCN is a type of neural network designed to operate on graph-structured data, while GAT is an extension of GCN that incorporates attention mechanisms.

Author: MO, YI WEN
Date: 2024/4/29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv, global_mean_pool, BatchNorm
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch_geometric.explain import Explainer, GraphMaskExplainer


class GCN(torch.nn.Module):
    """Graph Convolutional Network (GCN) model."""
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv3 = GCNConv(hidden_feats, out_feats)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        # return x
        return F.log_softmax(x, dim=1)


class GCN_batchnorm(torch.nn.Module):
    """Graph Convolutional Network (GCN) model."""
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN_batchnorm, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.bn1 = BatchNorm(hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.bn2 = BatchNorm(hidden_feats)
        self.conv3 = GCNConv(hidden_feats, out_feats)
        self.bn3 = BatchNorm(out_feats)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        # Apply log softmax activation
        return F.log_softmax(x, dim=1)


# 2024/9/16 add
class GCN_NodeClassification(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=3):
        super(GCN_NodeClassification, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 首層
        self.convs.append(GCNConv(in_feats, hidden_feats, normalize=True))
        self.bns.append(BatchNorm(hidden_feats))

        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_feats, hidden_feats, normalize=True))
            self.bns.append(BatchNorm(hidden_feats))

        # 輸出層
        self.convs.append(GCNConv(hidden_feats, out_feats, normalize=True))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x_res = x  # 保存殘差連接
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.leaky_relu(x)  # 使用 LeakyReLU
            x = F.dropout(x, p=0.5, training=self.training)
            if x_res.size(-1) == x.size(-1):  # 確保維度匹配
                x = x + x_res  # 添加殘差連接

        # 最後一層，不使用 BatchNorm 和激活函數
        x = self.convs[-1](x, edge_index)
        return x  # 直接返回線性輸出，不使用 softmax



class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, normalize=True)
        self.conv2 = SAGEConv(h_feats, out_feats, normalize=True)
        
    def forward(self, x, edge_index):
            # Apply dropout to the input features
            x = F.dropout(x, p=0.6, training=self.training)
            # Apply the first GraphSAGE layer with ReLU activation
            x = self.conv1(x, edge_index).relu()
            # Apply the second GraphSAGE layer
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)



class GraphSAGE_batchnorm(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE_batchnorm, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, normalize=True)
        self.bn1 = BatchNorm(h_feats)
        self.conv2 = SAGEConv(h_feats, h_feats, normalize=True)
        self.bn2 = BatchNorm(h_feats)
        self.conv3 = SAGEConv(h_feats, out_feats, normalize=True)

    def forward(self, x, edge_index):
        # Apply dropout to the input features
        x = F.dropout(x, p=0.6, training=self.training)
        # Apply the first GraphSAGE layer with BatchNorm and ReLU activation
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # Apply the second GraphSAGE layer with BatchNorm and ReLU activation
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # Apply the third GraphSAGE layer
        x = self.conv3(x, edge_index)
        # Apply log softmax activation
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """Graph Attention Network (GAT) model."""
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GAT_batchnorm(torch.nn.Module):
    """Graph Attention Network (GAT) model."""
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT_batchnorm, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=True, dropout=0.6)
        self.bn1 = BatchNorm(h_feats * 8)  # If concat=True, output size is h_feats * heads
        self.conv2 = GATConv(h_feats * 8, h_feats, heads=8, concat=True, dropout=0.6)
        self.bn2 = BatchNorm(h_feats * 8)
        self.conv3 = GATConv(h_feats * 8, out_feats, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # out= model(data.x, data.edge_index, data.batch)
        out= model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out, 1)

        # 2024/7/4 add
        # Filter for labels 0 and 2
        # mask = (data.y == 0) | (data.y == 2)
        # filtered_predicted = predicted[mask]
        # filtered_labels = data.y[mask]

        # total += filtered_labels.size(0)
        # correct += (filtered_predicted == filtered_labels).sum().item()
        # total_loss += loss.item() * mask.sum().item()
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
        total_loss += loss.item()


    if total > 0:
        train_acc = correct / total
        train_epoch_loss = total_loss / total
    else:
        train_acc = 0.0
        train_epoch_loss = 0.0
    # train_acc = correct / total
    # train_epoch_loss = total_loss / len(train_loader)
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
            # out = model(data.x, data.edge_index, data.batch)
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            _, predicted = torch.max(out, 1)

            # # Filter for labels 0 and 2
            # mask = (data.y == 0) | (data.y == 2)
            # filtered_predicted = predicted[mask]
            # filtered_labels = data.y[mask]

            # total += filtered_labels.size(0)
            # correct += (filtered_predicted == filtered_labels).sum().item()
            # total_loss += loss.item() * mask.sum().item()

            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
            total_loss += loss.item()

    if total > 0:
        test_acc = correct / total
        test_epoch_loss = total_loss / total
    else:
        test_acc = 0.0
        test_epoch_loss = 0.0

    return test_acc, test_epoch_loss







