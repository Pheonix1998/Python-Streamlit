import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

DATA_PATH = os.path.join("data", "E:\Labmentix Internship\Week 5\TOTAL POLICY DETAILS.csv")
df = pd.read_csv(DATA_PATH)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "", regex=False)
)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

df["bmi_category"] = pd.cut(
    df["bmi"], [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

df["age_group"] = pd.cut(
    df["age"], [0, 25, 40, 55, 100],
    labels=["<25", "25–40", "40–55", "55+"]
)

# ------------------------------------------------------------
# FEATURES & TARGET
# ------------------------------------------------------------

X = df.drop(columns=["charges_in_inr", "policy_no"])
y = df["charges_in_inr"]

categorical_cols = ["sex", "smoker", "region", "bmi_category", "age_group"]
numeric_cols = ["age", "bmi", "children"]

# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/insurance_model.pkl")

print("Model trained and saved successfully.")
