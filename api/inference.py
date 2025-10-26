# api/inference.py
import os
import pandas as pd
import numpy as np
from fastapi import UploadFile
from typing import List, Dict

# local imports - relative to repo root where src package exists
from src.models.train_xgboost import load_xgb_model
from src.models.train_ann import load_ann_model, ann_predict_numpy

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def predict_from_df(df: pd.DataFrame, model_type: str = "xgb") -> List[float]:
    """
    df: feature dataframe (only numeric feature columns)
    returns: raw PGA predictions (not-log)
    """
    X = df.select_dtypes(include=["number"]).values
    if model_type.lower() == "xgb":
        model = load_xgb_model(os.path.join(MODELS_DIR, "xgb_model.joblib"))
        preds_log = model.predict(X)
        preds = np.expm1(preds_log)
    else:
        model = load_ann_model(os.path.join(MODELS_DIR, "ann_model.pt"), input_dim=X.shape[1])
        preds_log = ann_predict_numpy(model, X)  # returns log-pga predictions
        preds = np.expm1(preds_log)
    return preds.tolist()

async def save_upload_file_tmp(upload_file: UploadFile, tmp_dir: str = "/tmp") -> str:
    out_path = os.path.join(tmp_dir, upload_file.filename)
    with open(out_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)
    return out_path

def predict_from_csv_file(csv_path: str, model_type: str = "xgb") -> (str, int):
    df = pd.read_csv(csv_path)
    preds = predict_from_df(df, model_type=model_type)
    out = csv_path.replace(".csv", f"_preds_{model_type}.csv")
    df["pga_pred"] = preds
    df.to_csv(out, index=False)
    return out, len(df)
