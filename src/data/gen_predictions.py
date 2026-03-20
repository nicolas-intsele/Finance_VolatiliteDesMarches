import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from arch import arch_model

PROJECT_ROOT = r"C:\Users\nicol\OneDrive\Bureau\Finance"
os.chdir(PROJECT_ROOT)
print("Current working directory:", os.getcwd())

# Load prediction dataset
df = pd.read_csv(
    "data/processed/features_pred.csv",
    index_col=0,
    parse_dates=True,
)
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True)

# Model paths
rf_path = os.path.join(PROJECT_ROOT, "src", "models", "rf_model.pkl")
gb_path = os.path.join(PROJECT_ROOT, "src", "models", "gb_model.pkl")
lstm_path = os.path.join(PROJECT_ROOT, "src", "models", "lstm_model.keras")
gru_path = os.path.join(PROJECT_ROOT, "src", "models", "gru_model.keras")
lstm_x_scaler_path = os.path.join(PROJECT_ROOT, "src", "models", "lstm_x_scaler.pkl")
lstm_y_scaler_path = os.path.join(PROJECT_ROOT, "src", "models", "lstm_y_scaler.pkl")
gru_x_scaler_path = os.path.join(PROJECT_ROOT, "src", "models", "gru_x_scaler.pkl")
gru_y_scaler_path = os.path.join(PROJECT_ROOT, "src", "models", "gru_y_scaler.pkl")

required_paths = [
    rf_path,
    gb_path,
    lstm_path,
    gru_path,
    lstm_x_scaler_path,
    lstm_y_scaler_path,
    gru_x_scaler_path,
    gru_y_scaler_path,
]
missing_paths = [path for path in required_paths if not os.path.exists(path)]
if missing_paths:
    raise FileNotFoundError("Missing model/scaler files:\n" + "\n".join(missing_paths))

# Load models and scalers
rf_model = joblib.load(rf_path)
gb_model = joblib.load(gb_path)
lstm_model = tf.keras.models.load_model(lstm_path)
gru_model = tf.keras.models.load_model(gru_path)
if hasattr(rf_model, "n_jobs"):
    rf_model.n_jobs = 1
if hasattr(gb_model, "n_jobs"):
    gb_model.n_jobs = 1
lstm_x_scaler = joblib.load(lstm_x_scaler_path)
lstm_y_scaler = joblib.load(lstm_y_scaler_path)
gru_x_scaler = joblib.load(gru_x_scaler_path)
gru_y_scaler = joblib.load(gru_y_scaler_path)

FEATURES_ML = [
    "vol_5",
    "vol_10",
    "vol_21",
    "vol_ewma",
    "ATR",
    "RSI",
    "MACD",
    "MACD_signal",
    "log_volume",
    "volume_change",
]
FEATURES_DL = [
    "vol_5",
    "vol_10",
    "vol_21",
    "vol_ewma",
    "log_return",
]

X_ml = df[FEATURES_ML]
X_dl = df[FEATURES_DL]
y = df["target_vol"]

# ML predictions
vol_rf = rf_model.predict(X_ml)
vol_gb = gb_model.predict(X_ml)


def create_sequences(values, seq_length):
    """Create rolling windows like the DL training notebook."""
    n_samples, n_features = values.shape
    if n_samples <= seq_length:
        return np.empty((0, seq_length, n_features))

    seqs = np.zeros((n_samples - seq_length, seq_length, n_features), dtype=values.dtype)
    for i in range(seq_length, n_samples):
        seqs[i - seq_length] = values[i - seq_length : i]
    return seqs


TIMESTEPS = 60

# Apply the same scaling used at training
X_lstm_scaled = lstm_x_scaler.transform(X_dl)
X_gru_scaled = gru_x_scaler.transform(X_dl)

seqs_lstm = create_sequences(X_lstm_scaled, TIMESTEPS)
seqs_gru = create_sequences(X_gru_scaled, TIMESTEPS)

# Predict and align to original index
n_total = len(df)
vol_lstm = np.full(n_total, np.nan)
vol_gru = np.full(n_total, np.nan)

if seqs_lstm.shape[0] > 0:
    lstm_preds_scaled = np.asarray(lstm_model.predict(seqs_lstm, verbose=0)).reshape(-1, 1)
    lstm_preds = np.clip(
        lstm_y_scaler.inverse_transform(lstm_preds_scaled).flatten(),
        1e-8,
        None,
    )
    start = TIMESTEPS
    vol_lstm[start : start + len(lstm_preds)] = lstm_preds

if seqs_gru.shape[0] > 0:
    gru_preds_scaled = np.asarray(gru_model.predict(seqs_gru, verbose=0)).reshape(-1, 1)
    gru_preds = np.clip(
        gru_y_scaler.inverse_transform(gru_preds_scaled).flatten(),
        1e-8,
        None,
    )
    start = TIMESTEPS
    vol_gru[start : start + len(gru_preds)] = gru_preds

# GARCH predictions
returns = df["log_return"].dropna()
if not isinstance(returns.index, pd.DatetimeIndex):
    returns.index = pd.to_datetime(returns.index)

garch = arch_model(returns, vol="Garch", p=1, q=1)
garch_fit = garch.fit(disp="off")
garch_forecast = garch_fit.forecast(start=returns.index[0])
vol_garch = np.sqrt(garch_forecast.variance.loc[df.index]).values.flatten()

# Final table
predictions = pd.DataFrame(
    {
        "target_vol": y,
        "vol_garch": vol_garch,
        "vol_rf": vol_rf,
        "vol_gb": vol_gb,
        "vol_lstm": vol_lstm,
        "vol_gru": vol_gru,
    },
    index=y.index,
).dropna()

os.makedirs("data/processed", exist_ok=True)
candidate_paths = [
    "data/processed/predictions.csv",
    "data/processed/predictions_new.csv",
    "predictions_new.csv",
]
saved_path = None
for output_path in candidate_paths:
    try:
        predictions.to_csv(output_path)
        saved_path = output_path
        break
    except PermissionError:
        continue

if saved_path is None:
    print("Could not write predictions file (permission denied on all target paths).")
else:
    print(f"{saved_path} created successfully")
print(predictions.head())
