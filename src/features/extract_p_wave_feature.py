# src/features/extract_p_wave_feature.py
import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

def p_wave_features_calc(window: np.ndarray, dt: float) -> dict:
    if len(window) == 0:
        return {k: np.nan for k in [
            "pkev12","pkev23","durP","tauPd","tauPt","PDd","PVd","PAd","PDt","PVt","PAt",
            "ddt_PDd","ddt_PVd","ddt_PAd","ddt_PDt","ddt_PVt","ddt_PAt"
        ]}

    durP = len(window) * dt
    PDd = np.max(window) - np.min(window)
    grad = np.gradient(window) / dt
    PVd = np.max(np.abs(grad))
    PAd = np.mean(np.abs(window))
    PDt = np.max(window)
    PVt = np.max(grad)
    PAt = np.sqrt(np.mean(window ** 2))
    tauPd = durP / PDd if PDd != 0 else 0
    tauPt = durP / PDt if PDt != 0 else 0

    ddt = lambda x: np.mean(np.abs(np.gradient(x))) if len(x) > 1 else 0
    ddt_PDd = ddt(window)
    ddt_PVd = ddt(grad)
    ddt_PAd = ddt(np.abs(window))
    ddt_PDt = ddt(np.maximum(window, 0))
    ddt_PVt = ddt(grad)
    ddt_PAt = ddt(window ** 2)

    pkev12 = np.sum(window ** 2) / len(window)
    pkev23 = np.sum(np.abs(window)) / len(window)

    return {
        "pkev12": pkev12, "pkev23": pkev23,
        "durP": durP, "tauPd": tauPd, "tauPt": tauPt,
        "PDd": PDd, "PVd": PVd, "PAd": PAd,
        "PDt": PDt, "PVt": PVt, "PAt": PAt,
        "ddt_PDd": ddt_PDd, "ddt_PVd": ddt_PVd,
        "ddt_PAd": ddt_PAd, "ddt_PDt": ddt_PDt,
        "ddt_PVt": ddt_PVt, "ddt_PAt": ddt_PAt
    }

def extract_from_trace(trace, win_len_seconds=2.0, sta=1.0, lta=10.0, on=2.5, off=1.0):
    """
    Given an ObsPy Trace object, detect p-wave and return features dict or None.
    """
    trace.detrend("demean")
    trace.filter("bandpass", freqmin=0.5, freqmax=20.0)
    dt = trace.stats.delta

    cft = classic_sta_lta(trace.data, int(sta / dt), int(lta / dt))
    trig = trigger_onset(cft, on, off)
    if len(trig) == 0:
        return None
    p_index = trig[0][0]
    win = int(win_len_seconds / dt)
    p_window = trace.data[p_index:p_index + win]
    if len(p_window) < 10:
        return None
    feats = p_wave_features_calc(p_window, dt)
    feats.update({
        "station": trace.stats.station,
        "network": trace.stats.network,
        "starttime": str(UTCDateTime(trace.stats.starttime)),
        "sampling_rate": trace.stats.sampling_rate
    })
    return feats
