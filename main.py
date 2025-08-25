from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import os, joblib, datetime as dt

app = FastAPI(title="Menstrual Health API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.pkl")

# --- Schemas ---
class LogEntry(BaseModel):
    date: str
    flow: Optional[int] = Field(default=None, description="0=light,1=medium,2=heavy")
    pain: Optional[int] = Field(default=None, description="0..10")
    mood: Optional[str] = None
    notes: Optional[str] = None

class PredictRequest(BaseModel):
    period_dates: List[str] = Field(..., description="List of historical period start dates YYYY-MM-DD")
    cycle_length_override: Optional[int] = None

class PredictResponse(BaseModel):
    next_period_date: str
    predicted_cycle_length: int
    ovulation_window_start: Optional[str] = None
    ovulation_window_end: Optional[str] = None
    method: str

class Remedy(BaseModel):
    id: Optional[int] = None
    title: str
    for_symptom: str
    description: str
    tags: List[str] = []

# --- In-memory stores (replace with DB later) ---
LOGS: List[LogEntry] = []
REMEDIES: List[Remedy] = []
RID = 1

# --- Utilities ---
def safe_parse(d: str) -> dt.date:
    return dt.datetime.strptime(d, "%Y-%m-%d").date()

def baseline_cycle_length(period_dates: List[str]) -> int:
    dates = sorted([safe_parse(d) for d in period_dates])
    if len(dates) < 2:
        return 28
    deltas = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
    # robust: trim extremes
    arr = np.array(deltas)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    mask = (arr >= q1-1.5*iqr) & (arr <= q3+1.5*iqr)
    filtered = arr[mask] if mask.any() else arr
    return int(round(filtered.mean()))

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

MODEL = load_model()

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/logs", response_model=LogEntry)
def add_log(entry: LogEntry):
    LOGS.append(entry)
    return entry

@app.get("/logs", response_model=List[LogEntry])
def get_logs():
    return LOGS

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.period_dates:
        raise HTTPException(400, "period_dates is required")
    cl = req.cycle_length_override or baseline_cycle_length(req.period_dates)

    last = safe_parse(sorted(req.period_dates)[-1])
    next_period = last + dt.timedelta(days=cl)

    # ML (if available) could refine delta days
    method = "baseline"
    if MODEL is not None:
        try:
            # simple feature: last cycle length + count
            dates = sorted([safe_parse(d) for d in req.period_dates])
            lens = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))] or [cl]
            X = np.array([[lens[-1], len(lens), np.mean(lens)]])
            delta = int(MODEL.predict(X)[0])
            next_period = last + dt.timedelta(days=delta)
            cl = delta
            method = "ml"
        except Exception:
            method = "baseline"

    ovu_start = next_period - dt.timedelta(days=14+2)
    ovu_end = next_period - dt.timedelta(days=14-2)

    return PredictResponse(
        next_period_date=next_period.isoformat(),
        predicted_cycle_length=cl,
        ovulation_window_start=ovu_start.isoformat(),
        ovulation_window_end=ovu_end.isoformat(),
        method=method,
    )

@app.post("/remedies", response_model=Remedy)
def add_remedy(r: Remedy):
    global RID
    r.id = RID
    RID += 1
    REMEDIES.append(r)
    return r

@app.get("/remedies", response_model=List[Remedy])
def list_remedies(symptom: Optional[str] = None):
    if symptom:
        return [r for r in REMEDIES if r.for_symptom.lower() == symptom.lower()]
    return REMEDIES

@app.delete("/remedies/{rid}")
def delete_remedy(rid: int):
    global REMEDIES
    before = len(REMEDIES)
    REMEDIES = [r for r in REMEDIES if r.id != rid]
    return {"deleted": before - len(REMEDIES)}
