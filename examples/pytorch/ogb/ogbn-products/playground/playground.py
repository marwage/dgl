import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import SAGEConv


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


dataset = DglNodePropPredDataset(name='ogbn-products') # list with one graph as element

splitted_idx = dataset.get_idx_split() # dict
train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test'] # torch.Tensor
graph = dataset.graph[0] # dgl.graph.DGLGraph
labels = dataset.labels # torch.Tensor
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
features = features.to(device)

# training
model(features)
