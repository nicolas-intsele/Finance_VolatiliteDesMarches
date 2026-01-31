import streamlit as st
import pandas as pd
import plotly.express as px
import os

PROJECT_ROOT = r"C:\Users\nicol\OneDrive\Bureau\Finance"
os.chdir(PROJECT_ROOT)

print("Current working directory:", os.getcwd())

st.set_page_config(page_title="Volatility Forecasting", layout="wide")

st.title("📊 Volatility Prediction Dashboard")

df = pd.read_csv("data/processed/features.csv")

model = st.selectbox("Select Model", ["GARCH", "XGBoost", "LSTM"])

fig = px.line(
    df,
    x="Date",
    y=["realized_vol", f"pred_{model.lower()}"],
    title="Realized vs Predicted Volatility"
)

st.plotly_chart(fig, use_container_width=True)
