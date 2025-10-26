# src/models/train_ann.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.preprocess import Preprocessor, P_WAVE_FEATURES, MODELS_DIR

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

def train_and_save_ann(df, target_col="PGA", cfg=None, save_prefix="models/ann"):
    cfg = cfg or DEFAULT
    features = P_WAVE_FEATURES
    X_df = df[features]
    y_raw = df[target_col].values
    y_log = np.log1p(y_raw)

    # preprocessor
    pre = Preprocessor(feature_list=features)
    Xp = pre.fit_transform(X_df.values, y_log)

    # torch dataset
    X_tensor = torch.tensor(Xp, dtype=torch.float32)
    y_tensor = torch.tensor(y_log, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLP(Xp.shape[1], hidden_sizes=cfg["hidden_sizes"], dropout=cfg["dropout"])
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(int(cfg["epochs"])):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0)
        if (epoch+1) % 50 == 0:
            avg = epoch_loss / len(dataset)
            print(f"Epoch {epoch+1}/{cfg['epochs']} loss={avg:.6f}")

    # save model and preprocessor
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "ann_model.pt")
    torch.save(model.state_dict(), model_path)
    pre.save(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    meta = {"features": features, "target_col": target_col, "cfg": cfg}
    with open(os.path.join(MODELS_DIR, "ann_metadata.json"), "w") as f:
        json.dump(meta, f)
    return model_path, os.path.join(MODELS_DIR, "preprocessor.joblib")

def load_ann_model(path=None, input_dim=None, device="cpu"):
    assert input_dim is not None, "pass input_dim when loading ANN architecture"
    model = MLP(input_dim, hidden_sizes=DEFAULT["hidden_sizes"], dropout=DEFAULT["dropout"])
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def ann_predict_numpy(model, X_numpy):
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_numpy, dtype=torch.float32)
        out = model(t).cpu().numpy()
    return out
