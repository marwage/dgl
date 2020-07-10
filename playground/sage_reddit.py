"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data
# from dgl.nn.pytorch.conv import SAGEConv
from sageconv import SAGEConv

import logging
import mw_logging
import subprocess
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
        for _ in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features):
        logging.debug("---------- Model forward ----------")

        h = features

        for layer in self.layers:
            logging.debug("---------- layer ----------")

            h = layer(self.g, h)

            mw_logging.log_tensor(h, "h_layer")

        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        mw_logging.log_gpu_memory("After eval forward")
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    start = time.time()
    mw_logging.log_gpu_memory("beginning", start)

    # load and preprocess dataset
    data = load_data(args)
    logging.debug("Type of data: {}".format(type(data)))

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    logging.debug("Number of nodes: {}".format(data.graph.number_of_nodes()))
    logging.debug("Number of features: {}".format(in_feats))

    logging.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        logging.info("use cuda: {}".format(args.gpu))

    mw_logging.log_peak_increase("After copying data")
    mw_logging.log_tensor(features, "features")
    mw_logging.log_tensor(labels, "labels")
    mw_logging.log_tensor(train_mask, "train_mask")
    mw_logging.log_tensor(val_mask, "val_mask")
    mw_logging.log_tensor(test_mask, "test_mask")

    # graph preprocess and calculate normalization factor
    g = data.graph
    # g.remove_edges_from(nx.selfloop_edges(g)) # 'DGLGraph' object has no attribute 'remove_edges_from'
    g = DGLGraph(g)
    n_edges = g.number_of_edges()

    logging.debug("Number of edges: {}".format(n_edges))

    logging.debug("---------- graph ----------")
    logging.debug("Type of graph: {}".format(type(g)))
    logging.debug("Size of graph: {} B".format(sys.getsizeof(g)))
    
    # create GraphSAGE model
    model = GraphSAGE(g,
                      in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type
                      )

    if cuda:
        model.cuda()

    mw_logging.log_peak_increase("After copying model")
    logging.debug("-------- model ---------")
    logging.debug("Type of model: {}".format(type(model)))
    for i, param in enumerate(model.parameters()):
        mw_logging.log_tensor(param, "param {}".format(i))

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        logits = model(features)
        mw_logging.log_gpu_memory("After forward", start)
        mw_logging.log_peak_increase("After forward")
        mw_logging.log_tensor(logits, "logits")
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        mw_logging.log_peak_increase("After loss")
        mw_logging.log_tensor(loss, "loss")

        optimizer.zero_grad()
        mw_logging.log_peak_increase("After zero_grad")
        loss.backward()
        mw_logging.log_peak_increase("After backward")
        optimizer.step()
        mw_logging.log_peak_increase("After step")

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    acc = evaluate(model, features, labels, test_mask)
    logging.info("Test Accuracy {:.4f}".format(acc))

    mw_logging.log_gpu_memory("End of training", start)


if __name__ == '__main__':
    name = "sage_reddit"
    monitoring_gpu = subprocess.Popen(["nvidia-smi", "dmon", "-s", "umt", "-o", "T", "-f", f"{name}.smi"])
    logging.basicConfig(filename=f"{name}.log",level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=3,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    args = parser.parse_args()
    
    logging.info(str(args))

    main(args)

    monitoring_gpu.terminate()
