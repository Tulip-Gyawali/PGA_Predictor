# src/data/preprocess.py
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# The canonical 17 p-wave features from your notebooks
P_WAVE_FEATURES = [
    "pkev12","pkev23","durP","tauPd","tauPt",
    "PDd","PVd","PAd","PDt","PVt","PAt",
    "ddt_PDd","ddt_PVd","ddt_PAd","ddt_PDt","ddt_PVt","ddt_PAt"
]

class Preprocessor:
    """
    Replicates notebook preprocessing: df.fillna(median) (should be applied before),
    scaler -> imputer -> selector (selector uses f_regression, k='all' in notebook).
    We save all fitted objects plus the feature order.
    """
    def __init__(self, feature_list=None):
        self.feature_list = feature_list or P_WAVE_FEATURES
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.selector = SelectKBest(score_func=f_regression, k="all")
        self.fitted = False

    def fit(self, X, y=None):
        """
        X: 2D array or DataFrame, columns correspond to self.feature_list
        y: log-transformed y used for SelectKBest (not strictly needed if k='all')
        """
        X = pd.DataFrame(X, columns=self.feature_list)
        # In notebooks they already filled NA with medians before, but fit imputer anyway:
        # Fit scaler, imputer, selector in the same order used in notebook
        _ = self.scaler.fit(X.values)
        _ = self.imputer.fit(self.scaler.transform(X.values))
        # selector expects numeric arrays; pass y if available (not necessary for k='all')
        if y is None:
            y_dummy = np.zeros(X.shape[0])
            self.selector.fit(self.imputer.transform(self.scaler.transform(X.values)), y_dummy)
        else:
            self.selector.fit(self.imputer.transform(self.scaler.transform(X.values)), y)
        self.fitted = True
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_list)
        Xt = self.scaler.transform(X.values)
        Xt = self.imputer.transform(Xt)
        Xt = self.selector.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "preprocessor.joblib")
        joblib.dump({
            "feature_list": self.feature_list,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "selector": self.selector
        }, path)
        # also write readable feature file
        with open(path + ".features.json", "w") as f:
            json.dump(self.feature_list, f)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(MODELS_DIR, "preprocessor.joblib")
        data = joblib.load(path)
        p = cls(feature_list=data["feature_list"])
        p.scaler = data["scaler"]
        p.imputer = data["imputer"]
        p.selector = data["selector"]
        p.fitted = True
        return p
