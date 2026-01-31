import numpy as np
import pandas as pd
from arch import arch_model


def train_garch(log_returns):
    """
    Entraîne un modèle GARCH(1,1) sur les log-returns
    
    Parameters
    ----------
    log_returns : pd.Series
        Rendements logarithmiques
    
    Returns
    -------
    model_fit : arch.univariate.base.ARCHModelResult
        Modèle GARCH entraîné
    """
    # Suppression des NaN
    returns = log_returns.dropna()

    # Mise à l'échelle (bonne pratique)
    returns_scaled = returns * 100

    # Définition du modèle
    model = arch_model(
        returns_scaled,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    # Entraînement
    model_fit = model.fit(disp="off")

    return model_fit


def forecast_garch(model_fit, horizon=1):
    """
    Prévision de la volatilité avec GARCH
    
    Returns
    -------
    pd.Series : volatilité prévue
    """
    forecast = model_fit.forecast(horizon=horizon)

    # Variance → volatilité
    vol_forecast = np.sqrt(forecast.variance.iloc[:, 0]) / 100

    return vol_forecast
