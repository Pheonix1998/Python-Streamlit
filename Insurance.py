import pandas as pd
import joblib
import os

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# SQL SERVER CONNECTION
# ------------------------------------------------------------

def load_data_from_sql():
    server = r"TATHAGATA\SQLEXPRESS"
    database = "MyDatabase"

    connection_string = (
        "mssql+pyodbc://@{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    ).format(server=server, database=database)

    engine = create_engine(connection_string)

    query = """
        SELECT
            policy_no,
            age,
            sex,
            bmi,
            children,
            smoker,
            region,
            charges_in_inr
        FROM dbo.[POLICY DETAILS]
    """

    df = pd.read_sql(query, engine)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    return df

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = load_data_from_sql()

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
# TRAIN MODEL
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

print("Model trained and saved successfully")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sqlalchemy import create_engine

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Insurance Cost Analytics & Prediction",
    layout="wide"
)

st.title("Insurance Cost Analysis & Prediction System")
st.markdown(
    "Enterprise-grade analytics and machine learning–driven "
    "insurance premium prediction powered by SQL Server"
)

# ------------------------------------------------------------
# SQL SERVER CONNECTION
# ------------------------------------------------------------

@st.cache_data
def load_data():
    server = r"TATHAGATA\SQLEXPRESS"
    database = "MyDatabase"

    connection_string = (
        "mssql+pyodbc://@{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    ).format(server=server, database=database)

    engine = create_engine(connection_string)

    query = """
        SELECT
            policy_no,
            age,
            sex,
            bmi,
            children,
            smoker,
            region,
            charges_in_inr
        FROM dbo.[POLICY DETAILS]
    """

    df = pd.read_sql(query, engine)

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
    df["bmi"], [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

df["age_group"] = pd.cut(
    df["age"], [0, 25, 40, 55, 100],
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

    fig_gender = px.pie(
        df.groupby("sex")["charges_in_inr"].mean().reset_index(),
        names="sex",
        values="charges_in_inr",
        hole=0.5,
        title="Average Insurance Cost Share by Gender"
    )
    col1.plotly_chart(fig_gender, use_container_width=True)

    fig_age = px.line(
        df.groupby("age_group")["charges_in_inr"].mean().reset_index(),
        x="age_group",
        y="charges_in_inr",
        markers=True,
        title="Insurance Cost Trend Across Age Groups"
    )
    col2.plotly_chart(fig_age, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    fig_smoker = px.bar(
        df.groupby("smoker")["charges_in_inr"].mean().reset_index(),
        x="smoker",
        y="charges_in_inr",
        title="Smoker vs Non-Smoker Cost Comparison",
        text_auto=".2s"
    )
    col1.plotly_chart(fig_smoker, use_container_width=True)

    fig_bmi = px.pie(
        df.groupby("bmi_category")["charges_in_inr"].mean().reset_index(),
        names="bmi_category",
        values="charges_in_inr",
        hole=0.4,
        title="Cost Distribution by BMI Category"
    )
    col2.plotly_chart(fig_bmi, use_container_width=True)

with tab3:
    fig_region = px.bar(
        df.groupby("region")["charges_in_inr"]
        .mean()
        .reset_index()
        .sort_values("charges_in_inr", ascending=False),
        x="region",
        y="charges_in_inr",
        title="Average Insurance Cost by Region",
        text_auto=".2s"
    )
    st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/insurance_model.pkl")

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

bmi_category = pd.cut(
    [bmi], [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)[0]

age_group = pd.cut(
    [age], [0, 25, 40, 55, 100],
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

st.caption(
    "SQL Server–backed analytics with Random Forest–based pricing intelligence"
)
