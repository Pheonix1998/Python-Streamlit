import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="OLA Booking Status Dashboard",
    layout="wide"
)

st.title("OLA Booking Status Analysis Dashboard")
st.markdown("Operational overview of booking outcomes for managerial decision-making")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"E:\Labmentix Internship\Week 4\OLA Cleaned.csv")
    return df

df = load_data()

# ----------------------------------
# Data Preparation
# ----------------------------------
drop_cols = ["Date", "Time", "Booking_ID", "Customer_ID", "Vehicle Images"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Visualization copy (no encoding)
df_viz = df.copy()

# Modeling copy (with encoding – retained for future ML use)
from sklearn.preprocessing import LabelEncoder
df_model = df.copy()

le = LabelEncoder()
for col in df_model.select_dtypes(include="object").columns:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# ----------------------------------
# KPI Section
# ----------------------------------
col1, col2 = st.columns(2)

col1.metric("Total Bookings", len(df_viz))
col2.metric("Successful Bookings",
            (df_viz["Booking_Status"] == "Success").sum())

st.divider()

# ----------------------------------
# Visualization
# ----------------------------------
st.subheader("Booking Status Distribution")

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(
    data=df_viz,
    x="Booking_Status",
    order=df_viz["Booking_Status"].value_counts().index,
    ax=ax
)

ax.set_title("Booking Status Distribution")
ax.set_xlabel("Booking Status")
ax.set_ylabel("Number of Rides")
plt.xticks(rotation=30)

st.pyplot(fig)

# ----------------------------------
# Data Preview
# ----------------------------------
st.subheader("Dataset Preview")
st.dataframe(df_viz.head(), use_container_width=True)