import streamlit as st
import pandas as pd
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
    "Executive analytics and machine-learning–driven insurance premium forecasting"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():
    data_path = os.path.join("data", "E:\Labmentix Internship\Week 5\TOTAL POLICY DETAILS.csv")

    if not os.path.exists(data_path):
        st.error("Dataset not found. Please verify deployment structure.")
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

c1, c2, c3 = st.columns(3)

c1.metric("Average Policy Cost (INR)", f"{df['charges_in_inr'].mean():,.0f}")
c2.metric("Median Policy Cost (INR)", f"{df['charges_in_inr'].median():,.0f}")

smoker_ratio = (
    df[df["smoker"] == "yes"]["charges_in_inr"].mean()
    / df[df["smoker"] == "no"]["charges_in_inr"].mean()
)

c3.metric("Smoker Premium Ratio", f"{smoker_ratio:.2f}x")

st.divider()

# ------------------------------------------------------------
# EDA DASHBOARD
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
        title="Average Cost Share by Gender"
    )
    col1.plotly_chart(fig_gender, use_container_width=True)

    fig_age = px.line(
        df.groupby("age_group")["charges_in_inr"].mean().reset_index(),
        x="age_group",
        y="charges_in_inr",
        markers=True,
        title="Cost Trend Across Age Groups"
    )
    col2.plotly_chart(fig_age, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    fig_smoker = px.bar(
        df.groupby("smoker")["charges_in_inr"].mean().reset_index(),
        x="smoker",
        y="charges_in_inr",
        title="Smoker vs Non-Smoker Cost Impact",
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
        df.groupby("region")["charges_in_inr"].mean().reset_index(),
        x="region",
        y="charges_in_inr",
        title="Regional Cost Comparison",
        text_auto=".2s"
    )
    st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model_path = os.path.join("model", "insurance_model.pkl")

    if not os.path.exists(model_path):
        st.error("Model artifact missing. Train model before deployment.")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# ------------------------------------------------------------
# PREDICTION ENGINE
# ------------------------------------------------------------

st.header("Insurance Premium Prediction")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Gender", ["male", "female"])

with c2:
    bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    smoker = st.selectbox("Smoker", ["yes", "no"])

with c3:
    children = st.selectbox("Dependents", [0, 1, 2, 3, 4])
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

st.caption("Random Forest–based pricing intelligence with business-aligned features")
