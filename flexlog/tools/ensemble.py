import numpy as np
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ..models.DT import decision_tree
from ..models.KNN import KNN
from ..models.SLFN import MLP

def mj_ensemble(predictions):
    avg_probs = np.mean(predictions, axis=0)
    return (avg_probs > 0.5).astype(int) 
    
class SNAILMetaLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.causal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size-1)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),  # 2x due to concat
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, T, D)
        x_conv = x.permute(0, 2, 1)  # (B, D, T)
        conv_out = self.causal_conv(x_conv)[:, :, :x.size(1)]  # trim extra padding
        conv_out = conv_out.permute(0, 2, 1)  # (B, T, H)

        attn_out, _ = self.attn(conv_out, conv_out, conv_out)  # (B, T, H)
        combined = torch.cat([conv_out, attn_out], dim=-1)  # (B, T, 2H)

        pooled = combined.mean(dim=1)  # global mean pooling
        return self.head(pooled).squeeze(-1)

class MetaFormer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, D) → (T, B, D)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.mean(dim=0)  # Mean pooling over sequence
        return self.head(x).squeeze(-1)

def prepare_meta_input(predictions_list):
    """
    Convert list of predictions into tensor format (B, T, D)
    where B = number of sequences, T = sequence length, D = number of models
    """
    stacked = np.stack(predictions_list, axis=-1)  # shape (N, T, D)
    return torch.tensor(stacked, dtype=torch.float32)

def train_meta_learner(meta_model, train_x, train_y, epochs=20, batch_size=64):
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    meta_model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = meta_model(xb)
            loss = criterion(preds, yb.float())
            loss.backward()
            optimizer.step()
    return meta_model

def mf_ensemble(predictions, llm_predictions_train, train_processed, train_labels):
    prediction_mlp, *_ = MLP(train_processed, train_processed, train_labels)
    prediction_knn, *_ = KNN(train_processed, train_processed, train_labels, n_neighbors=args.knn_neighbor)
    prediction_dt, *_ = decision_tree(train_processed, train_processed, train_labels)

    # Combine training predictions
    model_predictions_train = [
        np.array(prediction_dt),
        np.array(prediction_knn),
        np.array(prediction_mlp),
        np.array(llm_predictions_train)
    ]

    X_train = prepare_meta_input(model_predictions_train)
    y_train = torch.tensor(train_labels, dtype=torch.float32)

    meta_model = MetaFormer(input_dim=X_train.shape[-1])
    meta_model = train_meta_learner(meta_model, X_train, y_train)

    X_test = prepare_meta_input(predictions)
    meta_model.eval()
    with torch.no_grad():
        pred = meta_model(X_test)
    return (pred > 0.5).cpu().numpy().astype(int)

def snail_ensemble(predictions, llm_predictions_train, train_processed, train_labels):
    prediction_mlp, *_ = MLP(train_processed, train_processed, train_labels)
    prediction_knn, *_ = KNN(train_processed, train_processed, train_labels, n_neighbors=args.knn_neighbor)
    prediction_dt, *_ = decision_tree(train_processed, train_processed, train_labels)

    model_predictions_train = [
        np.array(prediction_dt),
        np.array(prediction_knn),
        np.array(prediction_mlp),
        np.array(llm_predictions_train)
    ]

    X_train = prepare_meta_input(model_predictions_train)
    y_train = torch.tensor(train_labels, dtype=torch.float32)

    meta_model = SNAILMetaLearner(input_dim=X_train.shape[-1])
    meta_model = train_meta_learner(meta_model, X_train, y_train)

    X_test = prepare_meta_input(predictions)
    meta_model.eval()
    with torch.no_grad():
        pred = meta_model(X_test)
    return (pred > 0.5).cpu().numpy().astype(int)
