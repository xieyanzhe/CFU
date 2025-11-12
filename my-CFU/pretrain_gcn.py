import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import random
from dataset_loader import load_dataset
import dgl
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.update_all(message_func=dgl.function.copy_u('h', 'm'),
                         reduce_func=dgl.function.mean('m', 'h_neigh'))
            h = g.ndata['h']
            return self.linear(h)
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim)
    def forward(self, g, x):
        h = F.relu(self.layer1(g, x))
        h = self.layer2(g, h)
        h = F.softmax(h,dim=1)
        return h
def contrastive_loss_batched(h_src, h_dst, temperature=0.5, batch_size=1024):
    device = h_src.device
    N = h_src.shape[0]
    total_loss = 0.0
    num_batches = (N + batch_size - 1) // batch_size
    all_loss = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, N)
        h_s = h_src[start:end]
        logits = torch.matmul(h_s, h_dst.T) / temperature
        labels = torch.arange(start, end, device=device)
        loss = F.cross_entropy(logits, labels)
        all_loss.append(loss)
    return torch.stack(all_loss).mean()
def train_contrastive_encoder(
    hidden_dim=64,
    lr=0.001,
    epochs=20,
    temperature=0.5,
    max_edges=10000,
    batch_size=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="pre_edg_parameter/encoder.pth",
    region_insert=False,
    dataset_path="../data/mydata_region/",
    dataset_path2="../data/mydata/",
    year="2023",
    region="SZ"
):
    set_seed(42)
    if region_insert:
        g, x, _, _, _, _ = load_dataset(
            node_path=dataset_path + year + "_" + region + "/feature_deal/node_processed.dat",
            edge_path=dataset_path + year + "_" + region + "/link.dat",
            label_path=dataset_path + year + "_" + region + "/label.dat",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            threshold=-1)
    else:
        g, x, _, _, _, _ = load_dataset(
            node_path=dataset_path2 + "/feature_deal/node_processed.dat",
            edge_path=dataset_path2 + "/link.dat",
            label_path=dataset_path2 + "/label.dat",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            threshold=-1)
    g = g.to(device)
    x = x.to(device)
    encoder = GCNEncoder(in_dim=x.size(1), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    src_all, dst_all = g.edges()
    all_edges = list(zip(src_all.tolist(), dst_all.tolist()))
    for epoch in range(1, epochs + 1):
        encoder.train()
        random.shuffle(all_edges)
        if len(all_edges) > max_edges:
            all_edges = all_edges[:max_edges]
        src = torch.tensor([s for s, _ in all_edges], device=device)
        dst = torch.tensor([d for _, d in all_edges], device=device)
        h = encoder(g, x)
        h_src = F.normalize(h[src], dim=1)
        h_dst = F.normalize(h[dst], dim=1)
        loss = contrastive_loss_batched(h_src, h_dst, temperature, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:03d} | Contrastive Loss: {loss:.6f}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(encoder.state_dict(), save_path)
    print(f"[✓] Encoder saved to {save_path}")
if __name__ == "__main__":
    train_contrastive_encoder(
        hidden_dim=64,
        lr=0.01,
        epochs=1000,
        batch_size=1024,
        max_edges=10000,
        save_path="pre_edge_parameter/all_2023/encoder.pth",
        region_insert=False,
        dataset_path="../data/mydata_region/",
        dataset_path2="../data/mydata/",
        year="2023",
        region="SH"
    )
