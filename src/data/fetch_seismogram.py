# src/data/fetch_seismogram.py
import random, datetime
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from src.features.extract_p_wave_feature import extract_from_trace
import pandas as pd
import os

client = Client("IRIS")

def batch_fetch_and_extract(num_samples=10, stations=None, net="IU", out_csv="Data/p_wave_features_extracted.csv"):
    if stations is None:
        stations = ["ANMO", "COR", "MAJO", "KBL"]
    records = []
    for i in range(num_samples):
        starttime = UTCDateTime(datetime.datetime(
            random.choice([2022, 2023, 2024]),
            random.randint(1, 12),
            random.randint(1, 25),
            random.randint(0, 21), 0, 0
        ))
        endtime = starttime + 2 * 3600
        got = False
        for station in stations:
            try:
                st = client.get_waveforms(net, station, "*", "BHZ", starttime, endtime)
                if st and len(st) > 0:
                    tr = st[0]
                    feats = extract_from_trace(tr)
                    if feats:
                        records.append(feats)
                        got = True
                        break
            except Exception:
                continue
        if not got:
            continue
    if len(records) > 0:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        return out_csv
    return None
