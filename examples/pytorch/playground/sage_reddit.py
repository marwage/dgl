import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import SAGEConv
from dgl.data.reddit import RedditDataset

import subprocess
import time
import logging
import sys


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
logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)
def log(start, when):
    mb = 1e6
    logging.debug("{:.1f}s:{}:active {:.2f}MB, allocated {:.2f}MB, reserved {:.2f}MB".format(time.time() - start, when, torch.cuda.memory_stats()["active_bytes.all.allocated"] / mb, torch.cuda.memory_allocated() / mb, torch.cuda.memory_reserved() / mb))
start = time.time()

data = RedditDataset()
graph = data.graph
num_classes = data.num_labels
features = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)
train_mask = torch.BoolTensor(data.train_mask)
val_mask = torch.BoolTensor(data.val_mask)
test_mask = torch.BoolTensor(data.test_mask)
feature_dim = features.shape[1]

logging.info("Dataset:Reddit")
logging.info("Number of nodes:{}".format(graph.number_of_nodes()))
logging.info("Number of edges:{}".format(graph.number_of_edges()))
logging.info("Dimensionality of features:{}".format(feature_dim))
logging.info("Number of classes:{}".format(num_classes))
logging.info("Number of training samples:{}".format(len(np.nonzero(train_mask)[0])))
logging.info("Number of validation samples:{}".format(len(np.nonzero(val_mask)[0])))
logging.info("Number of test samples:{}".format(len(np.nonzero(test_mask)[0])))

num_hidden_channels = 256
num_hidden_layers = 2
activation = F.relu
p_dropout = 0.2
aggregator_type = "mean"
model = GraphSAGE(graph, feature_dim, num_hidden_channels, num_classes, num_hidden_layers, activation, p_dropout, aggregator_type)
learning_rate = 3e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

logging.info("Number of hidden channels:{}".format(num_hidden_channels))
logging.info("Number of hidden layers:{}".format(num_hidden_layers))
logging.info("Activation function:{}".format("F.relu"))
logging.info("Dropout:{}".format(p_dropout))
logging.info("Aggregation:{}".format(aggregator_type))
logging.info("Model:{}".format("GraphSAGE"))
logging.info("Optimizer:{}".format("optim.Adam"))
logging.info("Loss function:{}".format("nn.CrossEntropyLoss"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(start, "after declaration")
model.to(device)
log(start, "after model.to()")
features = features.to(device)
labels = labels.to(device)
log(start, "after data.to()")

# training
try:
    embeddings = model(features)
    log(start, "after forward")
    loss = loss_fn(embeddings[train_mask], labels[train_mask])
    log(start, "after loss")
    optimizer.zero_grad()
    log(start, "after optimizer.zero_grad()")
    loss.backward()
    log(start, "after backward")
    optimizer.step()
    log(start, "after optimizer.zero_grad()")
except:
    logging.error(sys.exc_info()[0])

time.sleep(5)
monitoring_gpu.terminate()
