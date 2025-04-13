# api/main.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
from sklearn.metrics import classification_report, roc_auc_score

app = FastAPI()

# Load model
with open("../lgb_santander_customer_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    df = pd.read_csv(file.file)

    # Drop ID_code if present
    if "ID_code" in df.columns:
        df = df.drop(columns=["ID_code"])

    y_true = df["target"]
    X = df.drop(columns=["target"])

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    return {
        "roc_auc": round(roc_auc, 4),
        "classification_report": report
    }
