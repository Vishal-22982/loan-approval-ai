import pandas as pd
import pickle

df = pd.read_csv("loan_approval_data.csv")

print("Columns:", df.columns)

# Clean column names
df.columns = df.columns.str.strip()

df["Loan_Approved"] = df["Loan_Approved"].astype(str).str.strip()
df["Loan_Approved"] = df["Loan_Approved"].map({"Yes": 1, "No": 0})

# REMOVE invalid rows
df = df.dropna(subset=["Loan_Approved"])

# Convert to integer
df["Loan_Approved"] = df["Loan_Approved"].astype(int)

print("Target values:", df["Loan_Approved"].unique())

# Drop ID
if "Applicant_ID" in df.columns:
    df = df.drop("Applicant_ID", axis=1)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("features.pkl", "wb"))

print("Model trained successfully on REAL DATA ")