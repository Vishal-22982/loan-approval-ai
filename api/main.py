from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import os
import mysql.connector

app = FastAPI()

# -------------------------------
# ✅ ENABLE CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# ✅ LOAD MODEL (FIXED PATH)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "../ml-model/model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "../ml-model/scaler.pkl"), "rb"))
features = pickle.load(open(os.path.join(BASE_DIR, "../ml-model/features.pkl"), "rb"))

# -------------------------------
# ✅ MYSQL CONNECTION (SAFE)
# -------------------------------
try:
    db = mysql.connector.connect(
        host="localhost",  # ⚠️ change later for cloud
        user="root",
        password="Vishal2211",
        database="loan_db"
    )
    cursor = db.cursor()
    print("✅ MySQL Connected")
except:
    db = None
    cursor = None
    print("⚠️ MySQL not connected (will still run API)")

# -------------------------------
# ✅ HOME
# -------------------------------
@app.get("/")
def home():
    return {"message": "Loan Prediction API 🚀"}

# -------------------------------
# ✅ PREDICT API
# -------------------------------
@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # Encode + align features
        df = pd.get_dummies(df)
        df = df.reindex(columns=features, fill_value=0)

        # Scale
        df = scaler.transform(df)

        # Predict
        prediction = int(model.predict(df)[0])

        # -------------------------------
        # ✅ SAVE TO DB (IF AVAILABLE)
        # -------------------------------
        if cursor:
            cursor.execute("""
                INSERT INTO predictions 
                (applicant_income, coapplicant_income, age, credit_score, loan_amount, prediction)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                data.get("Applicant_Income"),
                data.get("Coapplicant_Income"),
                data.get("Age"),
                data.get("Credit_Score"),
                data.get("Loan_Amount"),
                prediction
            ))
            db.commit()

        return {
            "prediction": prediction,
            "result": "Approved ✅" if prediction == 1 else "Rejected ❌"
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# ✅ STATS API
# -------------------------------
@app.get("/stats")
def get_stats():
    if cursor:
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1")
        approved = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction=0")
        rejected = cursor.fetchone()[0]

        return {
            "approved": approved,
            "rejected": rejected,
            "total": approved + rejected
        }
    else:
        return {
            "approved": 0,
            "rejected": 0,
            "total": 0,
            "message": "Database not connected"
        }