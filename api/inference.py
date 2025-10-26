# api/inference.py
import os
import pandas as pd
import numpy as np
from src.data.preprocess import Preprocessor
from src.models.train_xgboost import load_xgb_model
from src.models.train_ann import load_ann_model, ann_predict_numpy

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def predict_from_df(df: pd.DataFrame, model_type: str = "xgb"):
    # load preprocessor that was saved during training
    pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if not os.path.exists(pre_path):
        raise FileNotFoundError("Preprocessor not found. Train and save a model first.")
    pre = Preprocessor.load(pre_path)
    # Ensure we have the exact feature order
    X_df = df[pre.feature_list]
    Xp = pre.transform(X_df.values)  # this returns transformed array ready for model

    if model_type.lower() == "xgb":
        model = load_xgb_model(os.path.join(MODELS_DIR, "xgb_model.joblib"))
        preds_log = model.predict(Xp)
        preds = np.expm1(preds_log)
    else:
        # ANN: need to know input_dim used during training
        # We saved preprocessor.selector so Xp.shape[1] gives final input dim
        input_dim = Xp.shape[1]
        ann_path = os.path.join(MODELS_DIR, "ann_model.pt")
        model = load_ann_model(ann_path, input_dim=input_dim)
        preds_log = ann_predict_numpy(model, Xp)
        preds = np.expm1(preds_log)

    return preds.tolist()
