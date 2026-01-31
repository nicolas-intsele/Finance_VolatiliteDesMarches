import numpy as np
import pandas as pd

def realized_volatility(log_returns, window=21):
    return np.sqrt(
        log_returns.rolling(window).apply(lambda x: np.sum(x**2))
    )
