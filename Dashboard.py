import streamlit as st
import pandas as pd

st.set_page_config(page_title="Proctor Dashboard", layout="wide")

st.title("🎓 Smart Proctoring Dashboard")

df = pd.read_csv("log.csv")

# -----------------------------
# Metrics
col1, col2 = st.columns(2)

col1.metric("Total Violations", len(df))
col2.metric("Unique Types", df["Violation"].nunique())

# -----------------------------
st.subheader("📊 Violation Summary")
st.bar_chart(df["Violation"].value_counts())

# -----------------------------
st.subheader("📄 Logs")
st.dataframe(df)

# -----------------------------
st.subheader("📸 Evidence")

for i in range(min(5, len(df))):
    st.image(df.iloc[i]["File"], width=300)