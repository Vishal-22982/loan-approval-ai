from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import mysql.connector

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load ML model
model = pickle.load(open("../ml-model/model.pkl", "rb"))
scaler = pickle.load(open("../ml-model/scaler.pkl", "rb"))
features = pickle.load(open("../ml-model/features.pkl", "rb"))

# ✅ Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vishal2211",  # 🔴 replace with your password
    database="loan_db"
)

cursor = db.cursor()

# -------------------------------
# HOME
# -------------------------------
@app.get("/")
def home():
    return {"message": "Loan Prediction API with MySQL 🚀"}

# -------------------------------
# PREDICT API
# -------------------------------
@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=features, fill_value=0)
        df = scaler.transform(df)

        prediction = int(model.predict(df)[0])

        # ✅ Save to MySQL
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
# STATS API (REAL DATA)
# -------------------------------
@app.get("/stats")
def get_stats():
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1")
    approved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction=0")
    rejected = cursor.fetchone()[0]

    return {
        "approved": approved,
        "rejected": rejected,
        "total": approved + rejected
    }