# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import pandas as pd

from .schemas import PredictRequest, PredictResponse
from .inference import predict_from_df, save_upload_file_tmp, predict_from_csv_file

app = FastAPI(title="EEW PGA Inference API")

# allow frontend dev server to access (change origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...), model_type: str = Form("xgb")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file containing features")
    tmp = await save_upload_file_tmp(file)
    out, n = predict_from_csv_file(tmp, model_type=model_type)
    return PredictResponse(n_rows=n, predictions_file=out).dict()

@app.post("/predict_json/")
def predict_json(req: PredictRequest):
    if not req.features or len(req.features) == 0:
        raise HTTPException(status_code=400, detail="Send features as a non-empty list of dicts")
    df = pd.DataFrame(req.features)
    preds = predict_from_df(df, model_type=req.model_type)
    return PredictResponse(n_rows=len(preds), predictions=preds).dict()
