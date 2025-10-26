# tests/test_features.py
import numpy as np
from src.features.extract_p_wave_feature import p_wave_features_calc

def test_p_wave_features_basic():
    # create a synthetic window
    dt = 0.01
    t = np.linspace(0,1,101)
    window = np.sin(2*np.pi*5*t) * np.exp(-t*3)
    feats = p_wave_features_calc(window, dt)
    # some keys should be present and numeric
    assert "pkev12" in feats
    assert feats["pkev12"] >= 0
    assert "PDd" in feats
