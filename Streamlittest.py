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

df = pd.read_csv(r"E:\Labmentix Internship\Week 5\TOTAL POLICY DETAILS.csv")

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

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

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

print("Model trained and saved to model/insurance_model.pkl")

# ============================================================
# INSURANCE COST ANALYTICS & PREDICTION – STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Insurance Cost Analytics & Prediction",
    layout="wide"
)

st.title("Insurance Cost Analysis & Prediction System")
st.markdown(
    "Business-driven exploratory analysis and machine learning–based "
    "insurance premium prediction"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    data_path = os.path.join("data", "E:\Labmentix Internship\Week 5\TOTAL POLICY DETAILS.csv")

    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.stop()

    df = pd.read_csv(data_path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )
    return df

df = load_data()

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

df["bmi_category"] = pd.cut(
    df["bmi"],
    [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

df["age_group"] = pd.cut(
    df["age"],
    [0, 25, 40, 55, 100],
    labels=["<25", "25–40", "40–55", "55+"]
)

# ------------------------------------------------------------
# KPI SECTION
# ------------------------------------------------------------

st.subheader("Key Business Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Policy Cost (INR)", f"{df['charges_in_inr'].mean():,.0f}")

with col2:
    st.metric("Median Policy Cost (INR)", f"{df['charges_in_inr'].median():,.0f}")

with col3:
    smoker_ratio = (
        df[df["smoker"] == "yes"]["charges_in_inr"].mean() /
        df[df["smoker"] == "no"]["charges_in_inr"].mean()
    )
    st.metric("Smoker Premium Ratio", f"{smoker_ratio:.2f}x")

st.divider()

# ------------------------------------------------------------
# EDA VISUALS
# ------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["Demographics", "Risk Factors", "Geographic Impact"]
)

with tab1:
    col1, col2 = st.columns(2)

    gender_avg = df.groupby("sex")["charges_in_inr"].mean().reset_index()
    fig_gender = px.pie(
        gender_avg,
        names="sex",
        values="charges_in_inr",
        hole=0.5,
        title="Average Insurance Cost Share by Gender"
    )
    col1.plotly_chart(fig_gender, use_container_width=True)

    age_avg = df.groupby("age_group")["charges_in_inr"].mean().reset_index()
    fig_age = px.line(
        age_avg,
        x="age_group",
        y="charges_in_inr",
        markers=True,
        title="Insurance Cost Trend Across Age Groups"
    )
    col2.plotly_chart(fig_age, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    smoker_avg = df.groupby("smoker")["charges_in_inr"].mean().reset_index()
    fig_smoker = px.bar(
        smoker_avg,
        x="smoker",
        y="charges_in_inr",
        title="Average Insurance Cost: Smoker vs Non-Smoker",
        text_auto=".2s"
    )
    col1.plotly_chart(fig_smoker, use_container_width=True)

    bmi_avg = df.groupby("bmi_category")["charges_in_inr"].mean().reset_index()
    fig_bmi = px.pie(
        bmi_avg,
        names="bmi_category",
        values="charges_in_inr",
        hole=0.4,
        title="Insurance Cost Distribution by BMI Category"
    )
    col2.plotly_chart(fig_bmi, use_container_width=True)

with tab3:
    region_avg = (
        df.groupby("region")["charges_in_inr"]
        .mean()
        .reset_index()
        .sort_values("charges_in_inr", ascending=False)
    )

    fig_region = px.bar(
        region_avg,
        x="region",
        y="charges_in_inr",
        title="Average Insurance Cost by Region",
        text_auto=".2s"
    )
    st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# LOAD TRAINED MODEL (ROBUST)
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model_path = os.path.join("model", "insurance_model.pkl")

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found at: {model_path}\n\n"
            "Please ensure `insurance_model.pkl` exists in the `model/` folder."
        )
        st.stop()

    return joblib.load(model_path)

model = load_model()

# ------------------------------------------------------------
# PREDICTION SECTION
# ------------------------------------------------------------

st.header("Predict Insurance Premium")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Gender", ["male", "female"])

with col2:
    bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    smoker = st.selectbox("Smoker", ["yes", "no"])

with col3:
    children = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

# Feature engineering (must match training)
bmi_category = pd.cut(
    [bmi],
    [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)[0]

age_group = pd.cut(
    [age],
    [0, 25, 40, 55, 100],
    labels=["<25", "25–40", "40–55", "55+"]
)[0]

if st.button("Predict Premium"):
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
        "bmi_category": bmi_category,
        "age_group": age_group
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Insurance Premium: INR {prediction:,.0f}")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.caption(
    "Random Forest model with business-driven feature engineering"
)

