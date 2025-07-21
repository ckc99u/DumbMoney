import pandas as pd

def detect_bull_to_bear(df: pd.DataFrame, window: int=5) -> pd.Series:
    """Strong bearish candle breaking support zone."""
    low_roll = df['Low'].rolling(window).min()
    cond = (df['Close'] < low_roll.shift(1)) & \
           (df['Close'] - df['Open'] < -0.5*(df['High']-df['Low']))
    return cond.astype(int)

def detect_bear_to_bull(df: pd.DataFrame, window: int=5) -> pd.Series:
    """Strong bullish candle breaking resistance zone."""
    high_roll = df['High'].rolling(window).max()
    cond = (df['Close'] > high_roll.shift(1)) & \
           (df['Close'] - df['Open'] > 0.5*(df['High']-df['Low']))
    return cond.astype(int)
