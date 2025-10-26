# src/models/train_xgboost.py
import os
import joblib
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.data.preprocess import Preprocessor, P_WAVE_FEATURES, MODELS_DIR

DEFAULT_PARAMS = {
  "n_estimators": 776,
  "learning_rate": 0.010590433420511285,
  "max_depth": 6,
  "subsample": 0.666852461341688,
  "colsample_bytree": 0.8724127328229327
}

def train_and_save_xgb(df, target_col="PGA", params=None, save_prefix="models/xgb"):
    """
    df: pandas DataFrame (should already be df.fillna(df.median()) as in notebook)
    target_col: 'PGA' (uppercase) per your notebook
    Saves: {save_prefix}.joblib (model), {save_prefix}_preprocessor.joblib
    """
    params = params or DEFAULT_PARAMS
    # Ensure features exist
    features = P_WAVE_FEATURES
    X_df = df[features]
    y_raw = df[target_col].values
    # notebook used log1p
    y_log = np.log1p(y_raw)

    # fit preprocessor on X and y_log (as in notebook selector usage)
    pre = Preprocessor(feature_list=features)
    Xp = pre.fit_transform(X_df.values, y_log)
    # train xgb on transformed features (y_log)
    model = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        verbosity=0
    )
    model.fit(Xp, y_log)

    # save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
    joblib.dump(model, model_path)
    pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    pre.save(pre_path)
    # Save metadata
    meta = {"features": features, "target_col": target_col, "params": params}
    with open(os.path.join(MODELS_DIR, "xgb_metadata.json"), "w") as f:
        json.dump(meta, f)
    return model_path, pre_path

def load_xgb_model(path=None):
    path = path or os.path.join(MODELS_DIR, "xgb_model.joblib")
    return joblib.load(path)

def evaluate_xgb(model, X_transformed, y_log_true):
    preds_log = model.predict(X_transformed)
    r2 = r2_score(y_log_true, preds_log)
    mae = mean_absolute_error(y_log_true, preds_log)
    rmse = np.sqrt(mean_squared_error(y_log_true, preds_log))
    return {"r2": r2, "mae": mae, "rmse": rmse}
