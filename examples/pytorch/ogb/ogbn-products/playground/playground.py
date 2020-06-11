import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import SAGEConv

import subprocess


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h

name = "playground"
monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])

dataset = DglNodePropPredDataset(name='ogbn-products') # list with one graph as element

splitted_idx = dataset.get_idx_split() # dict
train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test'] # torch.Tensor
graph = dataset.graph[0] # dgl.graph.DGLGraph
labels = dataset.labels # torch.Tensor
labels = labels[:, 0] # unpack labels
num_classes = dataset.num_classes # int
features = graph.ndata["feat"] # torch.Tensor

feature_dim = features.size()[1]

# graph.ndata dgl.view.NodeDataView
# graph.ndata.keys() collections.abc.KeysView

num_hidden_channels = 128
num_hidden_layers = 1
activation = F.relu
p_dropout = 0.2
aggregator_type = "mean"
model = GraphSAGE(graph, feature_dim, num_hidden_channels, num_classes, num_hidden_layers, activation, p_dropout, aggregator_type)
learning_reate = 3e-3
optimizer = optim.Adam(model.parameters(), lr=learning_reate)
loss_fn = nn.CrossEntropyLoss()

mb = 1e6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Memory reserved beginning: {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
model.to(device)
print("Memory reserved after model.to(): {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
features = features.to(device)
labels = labels.to(device)
print("Memory reserved after data.to(): {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))

# training
num_epochs = 3
try:
    for _ in range(num_epochs):
        embeddings = model(features)
        print("Memory reserved after forward: {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
        loss = loss_fn(embeddings[train_idx], labels[train_idx])
        print("Memory reserved after loss: {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
        optimizer.zero_grad()
        print("Memory reserved after zero_grad(): {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
        loss.backward()
        print("Memory reserved after backward: {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
        optimizer.step()
        print("Memory reserved after optimizer.step(): {:.2f} MB".format(torch.cuda.memory_reserved(device=device) / mb))
except:
    print("Training failed")
finally:
    monitoring_gpu.terminate()
