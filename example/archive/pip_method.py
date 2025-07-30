import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from datetime import datetime
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class PIPsPattern:
    """Container for PIPs analysis results"""
    start_idx: int
    end_idx: int
    pip_indices: List[int]
    pip_prices: List[float]
    window_data: np.ndarray
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    trend_direction: str  # 'bullish', 'bearish', 'sideways'
    volatility_score: float

def find_pips(data: np.ndarray, n_pips: int, dist_measure: int = 2) -> Tuple[List[int], List[float]]:
    """
    Extract perceptually important points from price data.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of price values
    n_pips : int
        Number of PIPs to extract (including first and last points)
    dist_measure : int
        1 = Euclidean Distance
        2 = Perpendicular Distance  
        3 = Vertical Distance
    
    Returns:
    --------
    Tuple[List[int], List[float]]
        (indices, prices) of the PIPs
    """
    if len(data) < n_pips:
        return list(range(len(data))), data.tolist()
    
    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]]  # Price
    
    for curr_point in range(2, n_pips):
        md = 0.0  # Max distance
        md_i = -1  # Max distance index
        insert_index = -1
        
        for k in range(0, curr_point - 1):
            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1
            
            if pips_x[right_adj] - pips_x[left_adj] <= 1:
                continue
                
            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope
            
            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                d = 0.0  # Distance
                
                if dist_measure == 1:  # Euclidean distance
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                elif dist_measure == 2:  # Perpendicular distance
                    d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                else:  # Vertical distance
                    d = abs((slope * i + intercept) - data[i])
                
                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj
        
        if md_i != -1:
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])
    
    return pips_x, pips_y

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


def detect_pips_patterns(
    df: pd.DataFrame,
    window: int = 50,
    n_pips: int = 7,
    dist_measure: int = 2,
    step_size: int = 10
) -> List[PIPsPattern]:

    patterns = []
    close_prices = df['close'].values
    
    for i in range(window, len(df), step_size):
        start_idx = i - window
        end_idx = i
        
        window_data = close_prices[start_idx:end_idx]
        pip_indices, pip_prices = find_pips(window_data, n_pips, dist_measure)
        
        # Convert relative indices to absolute DataFrame indices
        abs_pip_indices = [start_idx + idx for idx in pip_indices]
        
        # Analyze trend and volatility
        trend, volatility = analyze_pips_trend(pip_indices, pip_prices)
        
        pattern = PIPsPattern(
            start_idx=start_idx,
            end_idx=end_idx,
            pip_indices=abs_pip_indices,
            pip_prices=pip_prices,
            window_data=window_data,
            timestamp_start=df.index[start_idx],
            timestamp_end=df.index[end_idx-1],
            trend_direction=trend,
            volatility_score=volatility
        )
        
        patterns.append(pattern)
    
    return patterns

def backtest_pips_analysis(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1h',
    window: int = 50,
    n_pips: int = 7,
    dist_measure: int = 2,
    step_size: int = 10,
    plot_each: bool = True    # new flag to control per-window plotting
):
    print(f"Fetching {symbol} data from {start} to {end} ({interval})...")
    df = fetch_ohlc_from_yf(symbol, start, end, interval, log_prices=False)
    print(f"Data fetched: {len(df)} bars")
    
    patterns = []
    close_prices = df['close'].values

    # Slide window and plot each
    for i in range(window, len(df), step_size):
        start_idx = i - window
        end_idx = i
        window_data = close_prices[start_idx:end_idx]
        
        # 3. Extract PIPs
        pip_indices, pip_prices = find_pips(window_data, n_pips, dist_measure)
        abs_pip_indices = [start_idx + idx for idx in pip_indices]

        # 4. Package into PIPsPattern
        pattern = PIPsPattern(
            start_idx       = start_idx,
            end_idx         = end_idx,
            pip_indices     = abs_pip_indices,
            pip_prices      = pip_prices,
            window_data     = window_data,
            timestamp_start = df.index[start_idx],
            timestamp_end   = df.index[end_idx-1],
            trend_direction = None,    # we no longer compute trend
            volatility_score= None     # we no longer compute volatility
        )
        patterns.append(pattern)

        # 5. Plot each window immediately
        if plot_each:
            # Override color choice to a single consistent color, e.g. yellow
            # so the connecting lines stand out regardless of trend
            plot_pips_pattern(df, pattern, symbol, interval)
    
    print(f"Total windows processed: {len(patterns)}")
    return df, patterns

def plot_pips_pattern(df: pd.DataFrame, pattern: PIPsPattern, symbol: str, interval: str):
    segment = df.iloc[pattern.start_idx:pattern.end_idx].copy()
    
    # Ensure numeric
    for col in ['open','high','low','close']:
        segment[col] = segment[col].astype(float)

    # Build connector lines
    pattern_lines = []
    for i in range(len(pattern.pip_indices)-1):
        idx0 = pattern.pip_indices[i]
        idx1 = pattern.pip_indices[i+1]
        if idx0 < len(df) and idx1 < len(df):
            pattern_lines.append([
                (df.index[idx0], float(pattern.pip_prices[i])),
                (df.index[idx1], float(pattern.pip_prices[i+1]))
            ])

    # Create properly indexed scatter data
    # Map PIP indices to segment indices for correct positioning
    segment_pip_positions = []
    segment_pip_prices = []
    
    for i, abs_idx in enumerate(pattern.pip_indices):
        if abs_idx < len(df):
            # Convert absolute index to relative position within segment
            relative_idx = abs_idx - pattern.start_idx
            if 0 <= relative_idx < len(segment):
                segment_pip_positions.append(relative_idx)
                segment_pip_prices.append(float(pattern.pip_prices[i]))

    # Create scatter plot data aligned with segment
    scatter_data = pd.Series(data=np.nan, index=segment.index)
    for pos, price in zip(segment_pip_positions, segment_pip_prices):
        if pos < len(scatter_data):
            scatter_data.iloc[pos] = price

    # Only create scatter if we have valid data
    addplots = []
    if not scatter_data.dropna().empty:
        scatter = mpf.make_addplot(
            scatter_data,
            type='scatter', 
            markersize=50, 
            marker='o', 
            color='yellow'
        )
        addplots.append(scatter)

    # Plot with error handling
    try:
        mpf.plot(
            segment,
            type='candle',
            style='classic',
            title=f"{symbol} PIPs Window {pattern.start_idx}-{pattern.end_idx}",
            ylabel='Price',
            addplot=addplots if addplots else None,
            alines=dict(
                alines=pattern_lines, 
                colors=['yellow']*len(pattern_lines), 
                linewidths=2
            ) if pattern_lines else None,
            show_nontrading=False
        )
    except Exception as e:
        print(f"Plot error for window {pattern.start_idx}-{pattern.end_idx}: {e}")
        # Fallback: simple candlestick without PIPs
        mpf.plot(
            segment,
            type='candle',
            style='classic',
            title=f"{symbol} Window {pattern.start_idx}-{pattern.end_idx} (Simplified)",
            ylabel='Price',
            show_nontrading=False
        )

# Example usage and testing
if __name__ == '__main__':
    # Example 1: Analyze AAPL with hourly data
    print("Example 1: AAPL Hourly PIPs Analysis")
    print("=" * 40)
    
    df_aapl, patterns_aapl = backtest_pips_analysis(
        symbol='NQ=F',
        start='2025-7-15',
        end='2025-07-20',
        interval='5m',
        window=50,
        n_pips=12,
        dist_measure=2,
        step_size=1,
        plot_each=True
    )
    
    # Analyze performance
    summary_aapl = analyze_pattern_performance(patterns_aapl)
