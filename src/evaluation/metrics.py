import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def qlike_loss(y_true, y_pred, eps=1e-8):
    """
    QLIKE loss pour la volatilité
    """
    y_pred = np.maximum(y_pred, eps)
    return np.mean(
        np.log(y_pred**2) + (y_true**2) / (y_pred**2)
    )


def evaluate_volatility(y_true, y_pred):
    """
    Calcule MSE, MAE et QLIKE
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "QLIKE": qlike_loss(y_true, y_pred)
    }
