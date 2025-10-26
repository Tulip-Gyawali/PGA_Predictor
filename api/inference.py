# api/inference.py
import os
import tempfile
import pandas as pd
import numpy as np
from src.data.preprocess import Preprocessor
from src.models.train_xgboost import load_xgb_model
from src.models.train_ann import load_ann_model, ann_predict_numpy

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def predict_from_df(df: pd.DataFrame, model_type: str = "xgb"):
    """Run inference on a preprocessed DataFrame using the trained XGB or ANN model."""
    pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if not os.path.exists(pre_path):
        raise FileNotFoundError("Preprocessor not found. Train and save a model first.")
    pre = Preprocessor.load(pre_path)

    # Ensure we have the same feature order as during training
    X_df = df[pre.feature_list]
    Xp = pre.transform(X_df.values)

    if model_type.lower() == "xgb":
        model = load_xgb_model(os.path.join(MODELS_DIR, "xgb_model.joblib"))
        preds_log = model.predict(Xp)
        preds = np.expm1(preds_log)
    else:
        input_dim = Xp.shape[1]
        ann_path = os.path.join(MODELS_DIR, "ann_model.pt")
        model = load_ann_model(ann_path, input_dim=input_dim)
        preds_log = ann_predict_numpy(model, Xp)
        preds = np.expm1(preds_log)

    return preds.tolist()


def predict_from_csv_file(csv_path: str, model_type: str = "xgb"):
    """Load a CSV, preprocess, and predict."""
    df = pd.read_csv(csv_path)
    return predict_from_df(df, model_type=model_type)


def save_upload_file_tmp(upload_file):
    """
    Saves an uploaded FastAPI file temporarily and returns its path.
    Used in /predict_csv/ endpoint.
    """
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            tmp_path = tmp.name
        return tmp_path
    finally:
        upload_file.file.close()
