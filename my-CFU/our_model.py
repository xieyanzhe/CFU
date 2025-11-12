import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl
import torch.nn.functional as F
from pretrain_gcn import GCNEncoder
class StructureLearner(nn.Module):
    def __init__(self, in_dim, hidden_dim, k=5, use_cosine=True):
        super().__init__()
        self.k = k
        self.use_cosine = use_cosine
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x):
        device = x.device
        N = x.size(0)
        x = self.mlp(x)
        if self.use_cosine:
            x_norm = F.normalize(x, p=2, dim=1)
            sim = torch.matmul(x_norm, x_norm.T)
            sim.fill_diagonal_(-1.0)
            topk = torch.topk(sim, self.k, dim=1).indices
        else:
            dist = torch.cdist(x, x)
            dist.fill_diagonal_(float('inf'))
            topk = torch.topk(-dist, self.k, dim=1).indices

        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, self.k).flatten()
        dst = topk.flatten()
        g = dgl.graph((src, dst), num_nodes=N, device=device)
        return g
class EdgePruner(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size=50000,
                 initial_threshold=0.1, final_threshold=0.5
                 , max_epoch=2000,
                 pretrained_path="pre_edge_parameter/all_2022/encoder.pth"):
        super().__init__()
        self.encoder = GCNEncoder(in_dim=in_dim, hidden_dim=out_dim)
        if os.path.exists(pretrained_path):
            self.encoder.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
            print(f"[✓] Loaded pretrained GCN encoder from {pretrained_path}")
        else:
            raise FileNotFoundError(f"Pretrained encoder not found at: {pretrained_path}")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.batch_size = batch_size
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.max_epoch = max_epoch

    def get_dynamic_threshold(self, epoch):
        alpha = min(epoch / self.max_epoch, 1.0)
        return self.initial_threshold + alpha * (self.final_threshold - self.initial_threshold)

    def forward(self, g, x, epoch):
        device = x.device
        threshold = self.get_dynamic_threshold(epoch)

        with torch.no_grad():
            h = self.encoder(g.to(device), x.to(device))

        src_all, dst_all = g.edges()
        num_edges = src_all.shape[0]
        keep_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        all_scores = []

        with torch.no_grad():
            for i in range(0, num_edges, self.batch_size):
                j = min(i + self.batch_size, num_edges)
                src = src_all[i:j]
                dst = dst_all[i:j]
                h_src = h[src]
                h_dst = h[dst]
                raw_score = torch.sum(h_src * h_dst, dim=1)
                min_val = raw_score.min()
                max_val = raw_score.max()
                edge_score = (raw_score - min_val) / (max_val - min_val + 1e-8)
                keep_mask[i:j] = edge_score > threshold
                all_scores.append(edge_score.cpu())

        all_scores_tensor = torch.cat(all_scores)
        print("Edge Score Stats:")
        print("Min:", all_scores_tensor.min().item())
        print("Max:", all_scores_tensor.max().item())
        print("Mean:", all_scores_tensor.mean().item())
        print("Threshold:", threshold)

        src = src_all[keep_mask]
        dst = dst_all[keep_mask]
        pruned_g = dgl.graph((src, dst), num_nodes=g.num_nodes(), device=device)
        return pruned_g


def manual_merge_graphs(g1, g2):
    src1, dst1 = g1.edges()
    src2, dst2 = g2.edges()
    src = torch.cat([src1, src2])
    dst = torch.cat([dst1, dst2])
    edge_pairs = torch.stack([src, dst], dim=1)
    unique_edges = torch.unique(edge_pairs, dim=0)
    src_unique = unique_edges[:, 0]
    dst_unique = unique_edges[:, 1]
    new_g = dgl.graph((src_unique, dst_unique), num_nodes=g1.num_nodes(), device=src.device)
    return new_g
class DynamicGraphRiskModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k=5, refresh_every=5,
                 use_pruner=True, use_structure_learner=True, use_cosine=True,
                 warmup_prune_epochs=1):
        super().__init__()

        self.use_pruner = use_pruner
        self.use_structure_learner = use_structure_learner
        self.refresh_every = refresh_every
        self.warmup_prune_epochs = warmup_prune_epochs
        self.epoch_counter = 0
        if use_pruner:
            self.edge_pruner = EdgePruner(in_dim, hidden_dim)
        if use_structure_learner:
            self.structure_learner = StructureLearner(in_dim, hidden_dim, k, use_cosine)
        self.gnn1 = dglnn.SAGEConv(in_dim, hidden_dim, 'mean')
        self.gnn2 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.cached_graph = None
    def forward(self, g, x, return_embed=False):
        update_graph = (
                self.training and
                self.epoch_counter >= self.warmup_prune_epochs and
                (self.epoch_counter - self.warmup_prune_epochs) % self.refresh_every == 0
        )
        if update_graph:
            graphs_to_merge = []
            if self.cached_graph is None:
                self.cached_graph = g
            if self.use_pruner and self.epoch_counter >= self.warmup_prune_epochs:
                pruned_g = self.edge_pruner(self.cached_graph, x,self.epoch_counter)
                graphs_to_merge.append(pruned_g)
            if self.use_structure_learner:
                dynamic_g = self.structure_learner(x)
                graphs_to_merge.append(dynamic_g)
            if len(graphs_to_merge) > 1:
                fused_g = manual_merge_graphs(graphs_to_merge[0], graphs_to_merge[1])
            elif len(graphs_to_merge) == 1:
                fused_g = graphs_to_merge[0]
            else:
                fused_g = g  # fallback to original
            self.cached_graph = fused_g
        if self.training:
            if self.epoch_counter < self.warmup_prune_epochs:
                self.epoch_counter += 1
            else:
                g = self.cached_graph
                self.epoch_counter += 1
        x = self.gnn1(g, x)
        x = F.relu(x)
        x = self.gnn2(g, x)
        if return_embed:
            return x
        out = self.classifier(x)
        return out,g
