import pandas as pd
import yfinance as yf
import numpy as np

def load_ohlc_data_from_csv(
    csv_path: str,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
        
    # Set datetime as index
    df.set_index('date', inplace=True)
        
    # Ensure proper column names for algorithm compatibility
    df = df.rename(columns={
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume'})
        
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df.index >= start_dt]
            
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df.index <= end_dt]
        
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna()
        
    return df


def fetch_ohlc_from_yf(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1h',
    log_prices: bool = False
) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol} with interval {interval}.")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
    
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    
    if log_prices:
        df = np.log(df)
    
    return df
