import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import load_dataset
from our_model import DynamicGraphRiskModel
import random
import numpy as np
import torch
import dgl
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def evaluate(logits, labels, mask):
    logits = logits[mask]
    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels[mask].cpu().numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = 0.0
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    else:
        ks_stat = 0.0
    return acc, precision, recall, f1, auc, pr_auc, ks_stat
set_seed(1234)
def train(
    epochs=2000,
    lr=0.001,
    hidden_dim=None,
    weight_decay=5e-4,
    dropout=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="best_model.pt",
    modelname="our",  # or "GAT", "MLP", "DAGNN" or "our"
    patience=1000,
    region_insert=False,
    dataset_path="../data/mydata_region/",
    dataset_path2="../data/mydata/",
    year="2023",
    region="SZ",
    k=20,
    refresh_every=15,
    warmup_prune_epochs=12
):
    print(device)
    if region_insert == True:
        g, feats, labels, train_mask, val_mask, test_mask = load_dataset(
            node_path=dataset_path + year + "_" + region + "/feature_deal/node_processed.dat",
            edge_path=dataset_path + year + "_" + region + "/link.dat",
            label_path=dataset_path + year + "_" + region + "/label.dat",
            # feat_save_path=dataset_path+"/features_norm.npy",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            threshold=-1)
        g = g.to(device)
    else:
        g, feats, labels, train_mask, val_mask, test_mask = load_dataset(
            node_path=dataset_path2 + "/feature_deal/node_processed.dat",
            edge_path=dataset_path2 + "/link.dat",
            label_path=dataset_path2 + "/label.dat",
            # feat_save_path=dataset_path+"/features_norm.npy",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            threshold=-1)
        g = g.to(device)
    feats, labels = feats.to(device), labels.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    if modelname == "GCN":
        model = GCN(feats.size(1), hidden_dim, n_classes=2, dropout=dropout).to(device)
    elif modelname == "GAT":
        model = GAT(feats.size(1), hidden_dim, n_classes=2, num_heads=4, dropout=dropout).to(device)
    elif modelname == "MLP":
        model = MLP(feats.size(1), hidden_dim, n_classes=2, dropout=dropout).to(device)
    elif modelname == "GraphSAGE":
        model = GraphSAGE(feats.size(1), hidden_dim, n_classes=2, dropout=dropout,aggregator_type='mean').to(device)
    elif modelname == "our":
        model = DynamicGraphRiskModel(
            in_dim=feats.size(1),
            hidden_dim=64,
            out_dim=2,
            k=k,
            refresh_every=refresh_every,
            warmup_prune_epochs=warmup_prune_epochs).to(device)
    elif modelname == "DAGNN":
        model = DAGNN(feats.size(1), hidden_dim, 2, dropout=dropout,k=10).to(device)
    else:
        raise ValueError("Invalid model name!")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    save_strategy = "score"
    save_path = "best_model/all_2023/best_model.pth"
    best_score = -float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    current_graph_train = g
    for epoch in range(1, epochs + 1):
        model.train()

        logits,updated_graph_train = model(g, feats)
        current_graph_train = updated_graph_train
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits,g_val = model(current_graph_train, feats)
            val_loss = loss_fn(logits[val_mask], labels[val_mask]).item()
            acc, prec, rec, f1, auc, pr_auc, ks_stat = evaluate(logits, labels, val_mask)
            score = f1 + auc + pr_auc+ ks_stat
            saved_model = False

            if save_strategy == "score":
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    torch.save(current_graph_train, "best_model/all_2023/best_graph.pth")
                    saved_model = True
                    patience_counter = 0
                else:
                    patience_counter += 1

            elif save_strategy == "loss":
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    saved_model = True
                    patience_counter = 0
                else:
                    patience_counter += 1

            if epoch % 1 == 0 or epoch == 1:
                status = "[model saved]" if saved_model else ""
                print(
                    f"Epoch {epoch:03d} | Train Loss {loss.item():.4f} | Val Loss {val_loss:.4f} | "
                    f"Acc {acc:.4f} | Prec {prec:.4f} | Recall {rec:.4f} | "
                    f"F1 {f1:.4f} | AUC {auc:.4f} | PR-AUC {pr_auc:.4f} | KS {ks_stat:.4f} | Edge_numbers {current_graph_train.num_edges():.4f} | {status}"
                )
                logits, g_test = model(current_graph_train, feats)
                acc, prec, rec, f1, auc, pr_auc, ks_stat = evaluate(logits, labels, test_mask)
                print(
                    f"[Test] Acc {acc:.4f} | Prec {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | PR-AUC {pr_auc:.4f} | KS {ks_stat:.4f}| Edge_numbers {current_graph_train.num_edges():.4f}")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break
    if save_strategy == "score":
        print(f"Training complete. Best score: {best_score:.4f} at epoch {best_epoch}")
    else:
        print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    test_g = torch.load("best_model/all_2023/best_graph.pth")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        logits,g_test = model(test_g, feats)
        acc, prec, rec, f1, auc, pr_auc,ks_stat = evaluate(logits, labels, test_mask)
        print(f"[Test] Acc {acc:.4f} | Prec {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | PR-AUC {pr_auc:.4f} | KS {ks_stat:.4f}| Edge_numbers {g_test.num_edges():.4f}")
        result = {
            "Acc": round(acc, 4),
            "Prec": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "KS": round(ks_stat, 4)
        }
    print("Training completed.")
train()
