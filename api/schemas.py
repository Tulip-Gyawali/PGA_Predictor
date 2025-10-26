# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    # either you send features as list-of-dicts or a path to a CSV uploaded
    features: Optional[List[dict]] = None
    model_type: str = "xgb"  # "xgb" or "ann"

class PredictResponse(BaseModel):
    n_rows: int
    predictions: Optional[List[float]] = None
    predictions_file: Optional[str] = None
