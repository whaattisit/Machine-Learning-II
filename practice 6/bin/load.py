import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from . import preprocessing
import sys
sys.modules.setdefault('preprocessing', preprocessing)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "filter.joblib"

load_model: Pipeline = joblib.load(MODEL_PATH)

def predict_input(input: str) -> dict: 
    probs = load_model.predict_proba([input])
    preds = probs @ load_model.classes_
    result = int(np.rint(preds[0]))
    return {"msg" : f"Result for your review {result}"}