# src/models/evaluate_model.py
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_log_preds(y_true_log, y_pred_log):
    metrics = {
        "r2": r2_score(y_true_log, y_pred_log),
        "mae": mean_absolute_error(y_true_log, y_pred_log),
        "rmse": np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    }
    return metrics

def to_raw(y_log):
    import numpy as np
    return np.expm1(y_log)
