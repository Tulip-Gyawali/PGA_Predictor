# src/models/train_ann.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np

DEFAULT = {
  "hidden_sizes": [428, 442, 220],
  "dropout": 0.2835776221997114,
  "epochs": 829,
  "lr": 0.0011676487575205433,
  "weight_decay": 6.370451204388144e-05
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=None, dropout=0.28):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = DEFAULT["hidden_sizes"]
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_ann(X_train, y_train, X_val=None, y_val=None, cfg=None, save_path="models/ann_model.pt", device="cpu"):
    cfg = cfg or DEFAULT
    device = torch.device(device)
    model = MLP(X_train.shape[1], hidden_sizes=cfg["hidden_sizes"], dropout=cfg["dropout"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    batch_size = 64
    n_samples = X_train.shape[0]
    for epoch in range(int(cfg["epochs"])):
        model.train()
        perm = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.float32, device=device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # (optional) validation logging every 50 epochs
        if (epoch+1) % 50 == 0 and X_val is not None:
            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
                val_loss = ((pred_val - y_val)**2).mean()
            print(f"Epoch {epoch+1}/{cfg['epochs']} val_mse={val_loss:.6f}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model

def load_ann_model(path="models/ann_model.pt", input_dim=None, device="cpu"):
    assert input_dim is not None, "Provide input_dim to load ANN architecture"
    model = MLP(input_dim, hidden_sizes=DEFAULT["hidden_sizes"], dropout=DEFAULT["dropout"])
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def ann_predict_numpy(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        out = model(t).cpu().numpy()
    return out
