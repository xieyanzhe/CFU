import os
import pandas as pd
import torch
import numpy as np
import dgl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(
    node_path=None,
    edge_path=None,
    label_path=None,
    feat_save_path=None,
    train_ratio=None,
    val_ratio=None,
    test_ratio=None,
    random_state=None,
    threshold=None
):
    node_df = pd.read_csv(node_path, sep="\t", header=None, names=["idx", "name", "ntype", "feat"])
    feat_arr = node_df["feat"].str.split(",", expand=True).astype(np.float32).values
    num_nodes = feat_arr.shape[0]
    label_df = pd.read_csv(label_path, sep="\t", header=None, names=["idx", "name", "ntype", "label"])
    label_idx = label_df["idx"].values
    label_vals = label_df["label"].values
    train_idx, test_idx = train_test_split(label_idx, test_size=(1 - train_ratio), stratify=label_vals, random_state=random_state)
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(test_idx, test_size=(1 - val_ratio_adj), stratify=label_vals[test_idx], random_state=random_state)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    features = torch.tensor(feat_arr, dtype=torch.float32)
    edge_df = pd.read_csv(edge_path, sep="\t", header=None, names=["src", "dst", "etype", "weight"])
    edge_df = edge_df[edge_df["weight"] > threshold]
    src = edge_df["src"].values
    dst = edge_df["dst"].values
    weights = edge_df["weight"].values.tolist()
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    added_loops = g.num_edges() - len(weights)
    weights += [1.0] * added_loops
    g.edata["w"] = torch.tensor(weights, dtype=torch.float32)
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    labels[label_idx] = torch.tensor(label_vals, dtype=torch.long)
    return g, features, labels, train_mask, val_mask, test_mask
