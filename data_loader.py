import pandas as pd
import yfinance as yf

def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Disable auto_adjust and force single‐level columns
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        multi_level_index=False
    )
    # Now df has simple columns: Open, High, Low, Close, Volume
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df
