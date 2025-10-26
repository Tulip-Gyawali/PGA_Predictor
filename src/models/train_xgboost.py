# src/models/train_xgboost.py
import joblib
import os
import json
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
# above resolves to repo_root/models

DEFAULT_PARAMS = {
  "n_estimators": 776,
  "learning_rate": 0.010590433420511285,
  "max_depth": 6,
  "subsample": 0.666852461341688,
  "colsample_bytree": 0.8724127328229327
}

def train_xgb(X_train, y_train, X_val=None, y_val=None, params=None, save_path=None):
    params = params or DEFAULT_PARAMS
    model = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        verbosity=0
    )
    model.fit(X_train, y_train)
    if save_path:
        joblib.dump(model, save_path)
    return model

def save_xgb_model(model, path=None):
    path = path or os.path.join("models", "xgb_model.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path

def load_xgb_model(path="models/xgb_model.joblib"):
    return joblib.load(path)

def evaluate_xgb(model, X, y):
    preds_log = model.predict(X)
    r2 = r2_score(y, preds_log)
    mae = mean_absolute_error(y, preds_log)
    rmse = np.sqrt(mean_squared_error(y, preds_log))
    return {"r2": r2, "mae": mae, "rmse": rmse}
