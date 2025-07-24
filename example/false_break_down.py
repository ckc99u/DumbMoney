import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List

# --- 1. PIPs Implementation ---
def find_pips(
    data: np.ndarray,
    n_pips: int,
    dist_measure: int = 2
) -> Tuple[List[int], List[float]]:
    """
    Perceptually Important Points (PIPs) extraction.
    data: 1D array of prices
    n_pips: total points to extract (including first & last)
    dist_measure: 1=Euclidean, 2=Perpendicular, 3=Vertical
    Returns:
      pips_x: list of indices of PIPs
      pips_y: list of prices at those indices
    """
    pips_x = [0, len(data) - 1]
    pips_y = [data[0], data[-1]]

    for curr_count in range(2, n_pips):
        max_d, max_i, insert_idx = -1.0, -1, -1
        # For each segment between existing PIPs
        for seg in range(len(pips_x) - 1):
            left, right = pips_x[seg], pips_x[seg + 1]
            if right - left <= 1:
                continue
            # Line parameters
            dx = right - left
            dy = pips_y[seg + 1] - pips_y[seg]
            slope = dy / dx
            intercept = pips_y[seg] - slope * left
            # Examine each point between
            for i in range(left + 1, right):
                if dist_measure == 1:
                    # Euclidean: sum of distances to both endpoints
                    d = np.hypot(i - left, data[i] - pips_y[seg]) \
                        + np.hypot(right - i, pips_y[seg + 1] - data[i])
                elif dist_measure == 2:
                    # Perpendicular distance to line
                    d = abs(slope * i + intercept - data[i]) / np.hypot(slope, 1)
                else:
                    # Vertical
                    d = abs(slope * i + intercept - data[i])
                if d > max_d:
                    max_d, max_i, insert_idx = d, i, seg + 1
        pips_x.insert(insert_idx, max_i)
        pips_y.insert(insert_idx, data[max_i])
    return pips_x, pips_y

# --- 2. False Breakout Detection ---
def detect_false_breakouts(
    df: pd.DataFrame,
    window: int = 20,
    reversal: int = 3,
    n_pips: int = 5
) -> pd.DataFrame:
    """
    Add columns to df identifying false-breakout high/low bars.
    Steps:
      1) Compute rolling high/low over 'window'
      2) Detect breakout if current high > previous rolling high (and vice versa)
      3) Validate false breakout if price closes back inside within 'reversal' bars
      4) Confirm shape by extracting PIPs over breakout window to form 'W'/'M'
    """
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df['high_roll'] = df['High'].rolling(window).max()
    df['low_roll'] = df['Low'].rolling(window).min()
    
    # Convert to numpy arrays to completely bypass pandas alignment
    high_values = df['High'].values
    low_values = df['Low'].values
    high_roll_shifted = df['high_roll'].shift(1).values
    low_roll_shifted = df['low_roll'].shift(1).values
    
    # Create boolean arrays using numpy comparison
    break_high_values = np.where(np.isnan(high_roll_shifted), False, high_values > high_roll_shifted)
    break_low_values = np.where(np.isnan(low_roll_shifted), False, low_values < low_roll_shifted)
    
    # Assign back to dataframe
    df['break_high'] = break_high_values
    df['break_low'] = break_low_values
    
    df['false_high'] = False
    df['false_low'] = False

    for idx in df.index[window + reversal:]:
        i = df.index.get_loc(idx)
        # False-high breakout
        if df.at[idx, 'break_high']:
            # Within reversal, closing inside previous high_roll
            window_slice = df.iloc[i - reversal:i + 1]
            prev_high_roll = df.iloc[i-1]['high_roll']
            if (window_slice['Close'].max() < df.at[idx, 'high_roll']) and \
               (window_slice['Close'].min() > prev_high_roll):
                # Validate 'M' shape via PIPs
                prices = df['Close'].iloc[i - window : i + 1].to_numpy()
                if len(prices) >= n_pips:  # Ensure we have enough data points
                    px, py = find_pips(prices, n_pips, dist_measure=3)
                    # Expect a down-up-down shape: peak in middle
                    if len(py) >= 4 and py[2] > py[1] and py[2] > py[3]:
                        df.at[idx, 'false_high'] = True

        # False-low breakout
        if df.at[idx, 'break_low']:
            window_slice = df.iloc[i - reversal:i + 1]
            prev_low_roll = df.iloc[i-1]['low_roll']
            if (window_slice['Close'].min() > df.at[idx, 'low_roll']) and \
               (window_slice['Close'].max() < prev_low_roll):
                # Validate 'W' shape via PIPs
                prices = df['Close'].iloc[i - window : i + 1].to_numpy()
                if len(prices) >= n_pips:  # Ensure we have enough data points
                    px, py = find_pips(prices, n_pips, dist_measure=3)
                    if len(py) >= 4 and py[2] < py[1] and py[2] < py[3]:
                        df.at[idx, 'false_low'] = True

    return df

# --- 3. Debug Plotting ---
def plot_false_breakouts(df: pd.DataFrame, title: str):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close', color='black')
    fh = df[df['false_high']]
    fl = df[df['false_low']]
    plt.scatter(fh.index, fh['High'], marker='v', color='red', s=100, label='False High')
    plt.scatter(fl.index, fl['Low'], marker='^', color='green', s=100, label='False Low')
    plt.legend()
    plt.title(title)
    plt.show()

# --- 4. Backtesting Interface ---
def backtest_false_breakout(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1d',
    window: int = 20,
    reversal: int = 3,
    n_pips: int = 5
):
    """
    Fetch data, detect patterns, plot, and return annotated DataFrame.
    """
    df = yf.download(symbol, start=start, end=end, interval=interval)
    df = detect_false_breakouts(df, window, reversal, n_pips)
    plot_false_breakouts(df, f'{symbol} False Breakouts ({interval})')
    return df

if __name__ == '__main__':
    # Example usage:
    # Backtest Apple daily from 2022-01-01 to 2023-01-01
    result_df = backtest_false_breakout(
        symbol='AAPL',
        start='2022-05-01',
        end='2025-07-01',
        interval='1h',
        window=50,
        reversal=3,
        n_pips=5
    )
