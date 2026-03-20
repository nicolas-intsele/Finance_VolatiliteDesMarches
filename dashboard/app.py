from sklearn import metrics
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from datetime import date
import yfinance as yf


PROJECT_ROOT = r"C:\Users\nicol\OneDrive\Bureau\Finance"
os.chdir(PROJECT_ROOT)
print("Current working directory:", os.getcwd())

st.set_page_config(page_title="Prediction de la volatilité", layout="wide")

def load_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("dashboard/style.css")

st.title("Dashboard de Prédeiction de la Volatilité des Marchés Financiers")
st.sidebar.image("dashboard/Volatilite-marches-financiers.jpg", width=200)

def plot_volatility(df, model):
    fig = px.line(
        df,
        y=["target_vol", model],
        title="Realized vs Predicted Volatility"
    )
    return fig

@st.cache_resource
def load_model():
    model = joblib.load("src/models/rf_model.pkl")
    return model

model = load_model()
df_features = pd.read_csv("data/processed/features_pred.csv")
df_pred = pd.read_csv("data/processed/predictions.csv", index_col=0, parse_dates=True)

START = '2026-01-01'
TODAY = date.today().strftime("%Y-%m-%d")
stock =  ["AAPL"]
selected_stock = st.sidebar.selectbox("Selection de l'actif", stock)
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")


FEATURES = [
    "vol_5", "vol_10", "vol_21", "vol_ewma",
    "ATR", "RSI", "MACD", "MACD_signal",
    "log_volume", "volume_change"
]

latest_data = df_features[FEATURES].iloc[-1:]
predicted_vol = model.predict(latest_data)[0]
last_vol = df_pred["target_vol"].iloc[-1]
if abs(predicted_vol - last_vol) < 0.01:
    direction = "stable"
else:
    direction = "volatile"

st.subheader("📊 Prédictions")

st.write(f"Dernière volatilité: {last_vol:.6f}")
st.write(f"Volatilité prédite: {predicted_vol:.6f}")

if direction == "stable":
    st.success("Stable signal ✅")
else:
    st.error("Volatile signal ⚠️")


today_price = df_features["Volume"].iloc[-1]

# petit rendement simulé
predicted_price = [today_price - (predicted_vol * today_price), today_price + (predicted_vol * today_price)]
tomorrow_price = (predicted_price[0] + predicted_price[1]) / 2

col1, col2 = st.columns(2)
col1.metric("Aujourd'hui", f"{today_price:.1f}")
col2.metric("Prédiction pour demain", f"[{predicted_price[0]:.1f}, {predicted_price[1]:.1f}]",
            delta=f"{tomorrow_price - today_price:.6f}")

# Metriques de performance
metrics_df = pd.DataFrame({
    "Model": ["GARCH", "Random Forest", "Gradient Boosting", "LSTM", "GRU"],
    "MSE": [9.1443e-06, 4.6856e-07, 1.0002e-06, 1.2000e-06, 1.6000e-06],
    "MAE": [0.0019, 0.0004, 0.0006, 0.0007, 0.0010],
    "QLIKE": [-8.49, -8.07, -8.06, -8.07, -8.06]
})
# Graphique de comparaison des modèles
fig1 = px.bar(
    metrics_df,
    x="Model",
    y="MSE",
    title="Comparaison des Modèles - MSE"
)

st.subheader("Performance des Modèles")
col1, col2 = st.columns(2)
with col1:
    st.write("")
    st.write("")
    st.write("**Métriques de performance**")
    st.write("MSE: Mean Squared Error")
    st.write("MAE: Mean Absolute Error")
    st.write("QLIKE: Quasi-Likelihood Loss")
    st.table(metrics_df)
with col2:
    st.plotly_chart(fig1)

model_choice = st.sidebar.selectbox(
    "Choix du modèle",
    ["vol_garch", "vol_rf", "vol_gb", "vol_lstm", "vol_gru"]
)

# === Graphique volatilité ===
fig2 = px.line(
    df_pred,
    y=["target_vol", model_choice],
    title="Volatilité réelle vs Volatilité prédite"
)
st.plotly_chart(fig2, use_container_width=True)

# === Graphique du Backtesting ===
df_returns = pd.read_csv(
    "data/processed/backtest_returns.csv",
    index_col=0,
    parse_dates=True
)

fig = px.line(
    (1 + df_returns).cumprod(),
    title="Backtesting Performance"
)
st.plotly_chart(fig)
