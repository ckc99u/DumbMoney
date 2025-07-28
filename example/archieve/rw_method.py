import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from datetime import datetime
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class RollingWindowPattern:
    """Container for Rolling Window analysis results"""
    start_idx: int
    end_idx: int
    tops_indices: List[int]
    tops_prices: List[float]
    bottoms_indices: List[int]
    bottoms_prices: List[float]
    all_extrema_indices: List[int]
    all_extrema_prices: List[float]
    window_data: np.ndarray
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    trend_direction: str
    volatility_score: float

def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    """Checks if there is a local top detected at curr index"""
    if curr_index < order * 2 + 1:
        return False
    
    top = True
    k = curr_index - order
    v = data[k]
    
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break
    return top

def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    """Checks if there is a local bottom detected at curr index"""
    if curr_index < order * 2 + 1:
        return False
        
    bottom = True
    k = curr_index - order
    v = data[k]
    
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break
    return bottom

def find_rolling_window_extrema(data: np.ndarray, order: int = 10) -> Tuple[List[int], List[float], List[int], List[float]]:
    """
    Extract rolling window extrema from price data.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of price values
    order : int
        Rolling window order for extrema detection
        
    Returns:
    --------
    Tuple[List[int], List[float], List[int], List[float]]
        (tops_indices, tops_prices, bottoms_indices, bottoms_prices)
    """
    tops_indices = []
    tops_prices = []
    bottoms_indices = []
    bottoms_prices = []
    
    for i in range(len(data)):
        if rw_top(data, i, order):
            top_idx = i - order
            tops_indices.append(top_idx)
            tops_prices.append(data[top_idx])
            
        if rw_bottom(data, i, order):
            bottom_idx = i - order
            bottoms_indices.append(bottom_idx)
            bottoms_prices.append(data[bottom_idx])
    
    return tops_indices, tops_prices, bottoms_indices, bottoms_prices

def fetch_ohlc_from_yf(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1h',
    log_prices: bool = False
) -> pd.DataFrame:
    """Download OHLCV data with enhanced data type handling."""
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol} with interval {interval}.")
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Select and rename columns
    df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    
    # Ensure all columns are proper float types
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    
    if log_prices:
        df = np.log(df)
    
    return df

def analyze_extrema_trend(extrema_indices: List[int], extrema_prices: List[float]) -> Tuple[str, float]:
    """
    Analyze the overall trend direction and volatility from extrema points.
    Enhanced with robust error handling for None and invalid values.
    """
    # Handle empty or insufficient data
    if not extrema_prices or len(extrema_prices) < 3:
        return 'sideways', 0.0
    
    try:
        # Convert to numpy array and ensure float type, filtering out None values
        valid_prices = [p for p in extrema_prices if p is not None and np.isfinite(p)]
        
        if len(valid_prices) < 3:
            return 'sideways', 0.0
            
        prices = np.array(valid_prices, dtype=float)
        
        # Simple trend analysis
        first_third = np.mean(prices[:len(prices)//3 + 1])
        last_third = np.mean(prices[-len(prices)//3:])
        
        # Handle potential division by zero or invalid values
        if first_third == 0 or not np.isfinite(first_third) or not np.isfinite(last_third):
            price_change_pct = 0
        else:
            price_change_pct = (last_third - first_third) / abs(first_third)
        
        if price_change_pct > 0.02:  # 2% threshold
            trend = 'bullish'
        elif price_change_pct < -0.02:
            trend = 'bearish'
        else:
            trend = 'sideways'
        
        # Enhanced volatility calculation with comprehensive error handling
        if len(prices) <= 1:
            volatility = 0.0
        else:
            price_changes = np.diff(prices)
            # Filter out any NaN or infinite values
            price_changes = price_changes[np.isfinite(price_changes)]
            
            if len(price_changes) == 0:
                volatility = 0.0
            else:
                volatility = float(np.std(price_changes))
        
        # Final safety check - ensure we don't return NaN or infinite values
        if not np.isfinite(volatility):
            volatility = 0.0
            
        return trend, volatility
        
    except Exception as e:
        print(f"Error in trend analysis: {e}")
        return 'sideways', 0.0

def plot_rolling_window_pattern(df: pd.DataFrame, pattern: RollingWindowPattern, symbol: str, interval: str):
    """Plot rolling window extrema pattern with enhanced error handling"""
    segment = df.iloc[pattern.start_idx:pattern.end_idx].copy()
    
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close']:
        segment[col] = segment[col].astype(float)
    
    # Create connecting lines between extrema points
    pattern_lines = []
    if len(pattern.all_extrema_indices) > 1:
        for i in range(len(pattern.all_extrema_indices) - 1):
            idx0 = pattern.all_extrema_indices[i]
            idx1 = pattern.all_extrema_indices[i + 1]
            
            if idx0 < len(df) and idx1 < len(df):
                pattern_lines.append([
                    (df.index[idx0], float(pattern.all_extrema_prices[i])),
                    (df.index[idx1], float(pattern.all_extrema_prices[i + 1]))
                ])
    
    # Create scatter data for tops (green)
    tops_scatter_data = pd.Series(data=np.nan, index=segment.index)
    for i, abs_idx in enumerate(pattern.tops_indices):
        relative_idx = abs_idx - pattern.start_idx
        if 0 <= relative_idx < len(segment):
            tops_scatter_data.iloc[relative_idx] = float(pattern.tops_prices[i])
    
    # Create scatter data for bottoms (red)
    bottoms_scatter_data = pd.Series(data=np.nan, index=segment.index)
    for i, abs_idx in enumerate(pattern.bottoms_indices):
        relative_idx = abs_idx - pattern.start_idx
        if 0 <= relative_idx < len(segment):
            bottoms_scatter_data.iloc[relative_idx] = float(pattern.bottoms_prices[i])
    
    # Prepare addplots
    addplots = []
    
    # Add tops scatter if we have data
    if not tops_scatter_data.dropna().empty:
        tops_scatter = mpf.make_addplot(
            tops_scatter_data,
            type='scatter',
            markersize=50,
            marker='o',
            color='green'
        )
        addplots.append(tops_scatter)
    
    # Add bottoms scatter if we have data
    if not bottoms_scatter_data.dropna().empty:
        bottoms_scatter = mpf.make_addplot(
            bottoms_scatter_data,
            type='scatter',
            markersize=50,
            marker='o',
            color='red'
        )
        addplots.append(bottoms_scatter)
    
    # Determine line colors based on trend
    if pattern.trend_direction == 'bullish':
        line_colors = ['lime'] * len(pattern_lines)
    elif pattern.trend_direction == 'bearish':
        line_colors = ['red'] * len(pattern_lines)
    else:
        line_colors = ['yellow'] * len(pattern_lines)
    
    # **FIXED**: Enhanced volatility display with proper None/invalid value handling
    try:
        if pattern.volatility_score is None or not np.isfinite(pattern.volatility_score):
            vol_display = "0.0000"
        else:
            vol_display = f"{pattern.volatility_score:.4f}"
    except (TypeError, ValueError):
        vol_display = "0.0000"
    
    # **FIXED**: Safe trend direction handling
    trend_display = pattern.trend_direction if pattern.trend_direction else "UNKNOWN"
    
    # Create title
    title = (f"{symbol} Rolling Window Analysis (Order={10}) - {trend_display.upper()}\n"
             f"{pattern.timestamp_start.strftime('%Y-%m-%d %H:%M')} to "
             f"{pattern.timestamp_end.strftime('%Y-%m-%d %H:%M')}\n"
             f"Tops: {len(pattern.tops_indices)} | Bottoms: {len(pattern.bottoms_indices)} | "
             f"Volatility: {vol_display}")
    
    # Plot with error handling
    try:
        mpf.plot(
            segment,
            type='candle',
            style='classic',
            title=title,
            ylabel='Price',
            addplot=addplots if addplots else None,
            alines=dict(
                alines=pattern_lines,
                colors=line_colors,
                linewidths=2
            ) if pattern_lines else None,
            show_nontrading=False
        )
    except Exception as e:
        print(f"Plot error for window {pattern.start_idx}-{pattern.end_idx}: {e}")
        # Fallback: simple candlestick without extrema
        mpf.plot(
            segment,
            type='candle',
            style='classic',
            title=f"{symbol} Window {pattern.start_idx}-{pattern.end_idx} (Simplified)",
            ylabel='Price',
            show_nontrading=False
        )

def backtest_rolling_window_analysis(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1h',
    window: int = 50,
    order: int = 10,
    step_size: int = 10,
    plot_each: bool = True
):

    print(f"Fetching {symbol} data from {start} to {end} ({interval})...")
    df = fetch_ohlc_from_yf(symbol, start, end, interval, log_prices=False)
    print(f"Data fetched: {len(df)} bars")
    
    patterns = []
    close_prices = df['close'].values
    total_bars = len(df)
    
    # **ENHANCED**: Start from index 0 and ensure we cover ALL data
    start_positions = []
    
    # Generate all possible window positions to ensure complete coverage
    for i in range(0, total_bars, step_size):
        start_positions.append(i)
    
    # Ensure the last window ends exactly at the last candle
    if start_positions[-1] + window < total_bars:
        start_positions.append(total_bars - window)
    
    print(f"Processing {len(start_positions)} windows for complete coverage...")
    
    for pos_idx, start_idx in enumerate(start_positions):
        # Adaptive window sizing for edge cases
        if start_idx + window > total_bars:
            # For the final window, ensure it ends at the last candle
            end_idx = total_bars
            actual_start_idx = max(0, total_bars - window)
        else:
            end_idx = start_idx + window
            actual_start_idx = start_idx
        
        # Ensure minimum window size for meaningful analysis
        if end_idx - actual_start_idx < order * 2 + 1:
            continue
            
        window_data = close_prices[actual_start_idx:end_idx]
        
        # Extract rolling window extrema
        tops_indices, tops_prices, bottoms_indices, bottoms_prices = find_rolling_window_extrema(window_data, order)
        
        # Convert relative indices to absolute DataFrame indices
        abs_tops_indices = [actual_start_idx + idx for idx in tops_indices if idx >= 0]
        abs_bottoms_indices = [actual_start_idx + idx for idx in bottoms_indices if idx >= 0]
        
        # Combine all extrema points for trend analysis
        all_extrema_indices = abs_tops_indices + abs_bottoms_indices
        all_extrema_prices = tops_prices + bottoms_prices
        
        # Sort by index to maintain chronological order
        if all_extrema_indices:
            combined = list(zip(all_extrema_indices, all_extrema_prices))
            combined.sort(key=lambda x: x[0])
            all_extrema_indices, all_extrema_prices = zip(*combined)
            all_extrema_indices = list(all_extrema_indices)
            all_extrema_prices = list(all_extrema_prices)
        
        # Compute trend and volatility
        trend, volatility = analyze_extrema_trend(all_extrema_indices, all_extrema_prices)
        
        # Package into RollingWindowPattern
        pattern = RollingWindowPattern(
            start_idx=actual_start_idx,
            end_idx=end_idx,
            tops_indices=abs_tops_indices,
            tops_prices=tops_prices,
            bottoms_indices=abs_bottoms_indices,
            bottoms_prices=bottoms_prices,
            all_extrema_indices=all_extrema_indices,
            all_extrema_prices=all_extrema_prices,
            window_data=window_data,
            timestamp_start=df.index[actual_start_idx],
            timestamp_end=df.index[end_idx-1],
            trend_direction=trend,
            volatility_score=volatility
        )
        
        patterns.append(pattern)
        
        # Plot each window
        if plot_each:
            plot_rolling_window_pattern(df, pattern, symbol, interval)

    return df, patterns

def analyze_pattern_performance(patterns: List[RollingWindowPattern]) -> Dict:
    """Analyze performance of rolling window patterns with enhanced error handling"""
    if not patterns:
        return {}
    
    total_patterns = len(patterns)
    trend_counts = {'bullish': 0, 'bearish': 0, 'sideways': 0}
    total_tops = 0
    total_bottoms = 0
    volatilities = []
    
    for pattern in patterns:
        # Safe trend counting
        if pattern.trend_direction and pattern.trend_direction in trend_counts:
            trend_counts[pattern.trend_direction] += 1
            
        total_tops += len(pattern.tops_indices)
        total_bottoms += len(pattern.bottoms_indices)
        
        # Safe volatility collection
        if (pattern.volatility_score is not None and 
            np.isfinite(pattern.volatility_score)):
            volatilities.append(pattern.volatility_score)
    
    avg_volatility = np.mean(volatilities) if volatilities else 0.0
    
    return {
        'total_patterns': total_patterns,
        'trend_distribution': trend_counts,
        'avg_tops_per_window': total_tops / total_patterns if total_patterns > 0 else 0,
        'avg_bottoms_per_window': total_bottoms / total_patterns if total_patterns > 0 else 0,
        'avg_volatility': avg_volatility,
        'total_extrema_detected': total_tops + total_bottoms
    }

# Example usage and testing
if __name__ == '__main__':
    # Example: Analyze NQ futures with 5-minute data
    print("Rolling Window Extrema Analysis")
    print("=" * 40)
    
    df, patterns = backtest_rolling_window_analysis(
        symbol='NQ=F',
        start='2025-7-15',
        end='2025-07-20',
        interval='5m',
        window=50,
        order=3,  # Rolling window order
        step_size=1,
        plot_each=True
    )
    
