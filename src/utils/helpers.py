# src/utils/helpers.py
import os
import joblib
import torch

def ensure_models_dir(path="models"):
    os.makedirs(path, exist_ok=True)
    return path

def save_xgb(model, path="models/xgb_model.joblib"):
    ensure_models_dir(os.path.dirname(path) or "models")
    joblib.dump(model, path)

def load_xgb(path="models/xgb_model.joblib"):
    return joblib.load(path)

def save_ann(model, path="models/ann_model.pt"):
    ensure_models_dir(os.path.dirname(path) or "models")
    torch.save(model.state_dict(), path)

def load_ann(model_class, path="models/ann_model.pt", map_location="cpu"):
    model = model_class
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()
    return model
