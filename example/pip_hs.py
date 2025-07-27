import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HSPattern:
    """Enhanced Head & Shoulders pattern with PIPs integration"""
    inverted: bool  # True for inverse H&S (bullish), False for regular H&S (bearish)
    
    # PIPs indices and prices
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1
    
    # Pattern prices
    l_shoulder_p: float = -1
    r_shoulder_p: float = -1
    l_armpit_p: float = -1
    r_armpit_p: float = -1
    head_p: float = -1
    
    # Pattern boundaries
    start_i: int = -1
    break_i: int = -1
    break_p: float = -1
    neck_start: float = -1
    neck_end: float = -1
    
    # PIPs-specific attributes
    pattern_strength: float = -1  # Based on PIPs distance measures
    pip_indices: List[int] = None
    pip_prices: List[float] = None
    
    # Traditional attributes
    neck_slope: float = -1
    head_width: float = -1
    head_height: float = -1
    pattern_r2: float = -1

def find_pips(data: np.array, n_pips: int, dist_measure: int = 2) -> Tuple[List[int], List[float]]:
    """
    Extract Perceptually Important Points from price data.
    
    Parameters:
    -----------
    data : np.array
        1D array of price values
    n_pips : int
        Number of PIPs to extract
    dist_measure : int
        1 = Euclidean Distance
        2 = Perpendicular Distance  
        3 = Vertical Distance
    
    Returns:
    --------
    Tuple[List[int], List[float]]
        (indices, prices) of the PIPs
    """
    if len(data) < 2:
        return [0], [data[0]] if len(data) == 1 else [], []
    
    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]]  # Price
    
    for curr_point in range(2, min(n_pips, len(data))):
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
            
            if time_diff == 0:
                continue
                
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

def identify_hs_candidates_from_pips(pip_indices: List[int], pip_prices: List[float], 
                                   data: np.array) -> List[Tuple[int, int, int, int, int]]:
    """
    Identify potential head and shoulders patterns from PIPs.
    
    Returns:
    --------
    List of tuples: (l_shoulder, l_armpit, head, r_armpit, r_shoulder) indices
    """
    candidates = []
    
    if len(pip_indices) < 5:
        return candidates
    
    # Search for 5-point patterns in PIPs
    for i in range(len(pip_indices) - 4):
        sequence = pip_indices[i:i+5]
        prices = pip_prices[i:i+5]
        
        # Check for alternating pattern (peak-valley-peak-valley-peak or vice versa)
        # Regular H&S: high-low-high-low-high where middle high is highest
        # Inverse H&S: low-high-low-high-low where middle low is lowest
        
        # Regular H&S pattern check
        if (prices[0] > prices[1] and prices[1] < prices[2] and 
            prices[2] > prices[3] and prices[3] < prices[4]):
            # Head should be highest point
            if prices[2] == max(prices[0], prices[2], prices[4]):
                candidates.append((sequence[0], sequence[1], sequence[2], sequence[3], sequence[4]))
        
        # Inverse H&S pattern check  
        if (prices[0] < prices[1] and prices[1] > prices[2] and 
            prices[2] < prices[3] and prices[3] > prices[4]):
            # Head should be lowest point
            if prices[2] == min(prices[0], prices[2], prices[4]):
                candidates.append((sequence[0], sequence[1], sequence[2], sequence[3], sequence[4]))
    
    return candidates

def validate_hs_pattern_pips(l_shoulder: int, l_armpit: int, head: int, 
                           r_armpit: int, r_shoulder: int, data: np.array,
                           current_i: int, inverted: bool = False) -> Optional[HSPattern]:
    """
    Validate head and shoulders pattern using PIPs-identified points.
    """
    try:
        # Extract prices
        l_shoulder_p = data[l_shoulder]
        l_armpit_p = data[l_armpit]
        head_p = data[head]
        r_armpit_p = data[r_armpit]
        r_shoulder_p = data[r_shoulder]
        
        if inverted:
            # Inverse H&S validation
            # Head must be lower than both shoulders
            if head_p >= min(l_shoulder_p, r_shoulder_p):
                return None
                
            # Shoulders should be relatively level (within tolerance)
            shoulder_diff = abs(l_shoulder_p - r_shoulder_p) / max(l_shoulder_p, r_shoulder_p)
            if shoulder_diff > 0.1:  # 10% tolerance
                return None
        else:
            # Regular H&S validation
            # Head must be higher than both shoulders
            if head_p <= max(l_shoulder_p, r_shoulder_p):
                return None
                
            # Shoulders should be relatively level
            shoulder_diff = abs(l_shoulder_p - r_shoulder_p) / max(l_shoulder_p, r_shoulder_p)
            if shoulder_diff > 0.1:  # 10% tolerance
                return None
        
        # Symmetry validation - time from shoulders to head should be comparable
        l_to_h_time = head - l_shoulder
        r_to_h_time = r_shoulder - head
        
        if l_to_h_time == 0 or r_to_h_time == 0:
            return None
            
        time_ratio = max(l_to_h_time, r_to_h_time) / min(l_to_h_time, r_to_h_time)
        if time_ratio > 3.0:  # Maximum 3:1 time ratio
            return None
        
        # Calculate neckline
        neck_run = r_armpit - l_armpit
        if neck_run == 0:
            return None
            
        neck_rise = r_armpit_p - l_armpit_p
        neck_slope = neck_rise / neck_run
        
        # Neckline value at current position
        neck_val = l_armpit_p + (current_i - l_armpit) * neck_slope
        
        # Check for neckline break
        current_price = data[current_i]
        
        if inverted:
            # For inverse H&S, price should break above neckline
            if current_price <= neck_val:
                return None
        else:
            # For regular H&S, price should break below neckline
            if current_price >= neck_val:
                return None
        
        # Find pattern start (extend back from left shoulder)
        head_width = r_armpit - l_armpit
        pat_start = max(0, l_shoulder - head_width // 2)
        neck_start = l_armpit_p + (pat_start - l_armpit) * neck_slope
        
        # Calculate pattern strength based on head prominence
        if inverted:
            pattern_strength = (min(l_armpit_p, r_armpit_p) - head_p) / abs(head_p)
        else:
            pattern_strength = (head_p - max(l_armpit_p, r_armpit_p)) / abs(head_p)
        
        # Create pattern object
        pattern = HSPattern(inverted=inverted)
        pattern.l_shoulder = l_shoulder
        pattern.l_armpit = l_armpit
        pattern.head = head
        pattern.r_armpit = r_armpit
        pattern.r_shoulder = r_shoulder
        
        pattern.l_shoulder_p = l_shoulder_p
        pattern.l_armpit_p = l_armpit_p
        pattern.head_p = head_p
        pattern.r_armpit_p = r_armpit_p
        pattern.r_shoulder_p = r_shoulder_p
        
        pattern.start_i = pat_start
        pattern.break_i = current_i
        pattern.break_p = current_price
        pattern.neck_start = neck_start
        pattern.neck_end = neck_val
        pattern.neck_slope = neck_slope
        pattern.head_width = head_width
        pattern.pattern_strength = pattern_strength
        
        # Calculate head height
        neckline_at_head = l_armpit_p + (head - l_armpit) * neck_slope
        if inverted:
            pattern.head_height = neckline_at_head - head_p
        else:
            pattern.head_height = head_p - neckline_at_head
        
        return pattern
        
    except Exception as e:
        return None

def find_hs_patterns_pips(data: np.array, window: int = 100, n_pips: int = 15, 
                         dist_measure: int = 2, min_pattern_bars: int = 20) -> Tuple[List[HSPattern], List[HSPattern]]:
    """
    Find head and shoulders patterns using PIPs algorithm.
    
    Parameters:
    -----------
    data : np.array
        Price data array
    window : int
        Rolling window size for PIPs analysis
    n_pips : int
        Number of PIPs to extract per window
    dist_measure : int
        Distance measure for PIPs (1=Euclidean, 2=Perpendicular, 3=Vertical)
    min_pattern_bars : int
        Minimum bars required for a valid pattern
        
    Returns:
    --------
    Tuple[List[HSPattern], List[HSPattern]]
        (regular_hs_patterns, inverse_hs_patterns)
    """
    hs_patterns = []
    ihs_patterns = []
    
    # Sliding window analysis
    step_size = window // 4  # 75% overlap
    
    for i in range(window, len(data), step_size):
        # Extract window data
        window_start = i - window
        window_data = data[window_start:i]
        
        if len(window_data) < min_pattern_bars:
            continue
        
        # Find PIPs in current window
        pip_indices_rel, pip_prices = find_pips(window_data, n_pips, dist_measure)
        
        if len(pip_indices_rel) < 5:
            continue
        
        # Convert relative indices to absolute
        pip_indices_abs = [window_start + idx for idx in pip_indices_rel]
        
        # Identify H&S candidates from PIPs
        candidates = identify_hs_candidates_from_pips(pip_indices_abs, pip_prices, data)
        
        # Validate each candidate
        for l_shoulder, l_armpit, head, r_armpit, r_shoulder in candidates:
            # Skip if pattern is too close to current position
            if i - r_shoulder < 5:
                continue
            
            # Try regular H&S
            hs_pattern = validate_hs_pattern_pips(
                l_shoulder, l_armpit, head, r_armpit, r_shoulder, 
                data, i, inverted=False
            )
            
            if hs_pattern is not None:
                # Check for duplicates
                is_duplicate = False
                for existing in hs_patterns:
                    if abs(existing.head - hs_pattern.head) < 10:  # Within 10 bars
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    hs_pattern.pip_indices = pip_indices_abs
                    hs_pattern.pip_prices = pip_prices
                    hs_patterns.append(hs_pattern)
            
            # Try inverse H&S
            ihs_pattern = validate_hs_pattern_pips(
                l_shoulder, l_armpit, head, r_armpit, r_shoulder, 
                data, i, inverted=True
            )
            
            if ihs_pattern is not None:
                # Check for duplicates
                is_duplicate = False
                for existing in ihs_patterns:
                    if abs(existing.head - ihs_pattern.head) < 10:  # Within 10 bars
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    ihs_pattern.pip_indices = pip_indices_abs
                    ihs_pattern.pip_prices = pip_prices
                    ihs_patterns.append(ihs_pattern)
    
    return hs_patterns, ihs_patterns

def plot_hs_pips(candle_data: pd.DataFrame, pat: HSPattern, pad: int = 2):
    """
    Plot head and shoulders pattern with PIPs visualization.
    """
    if pad < 0:
        pad = 0
    
    idx = candle_data.index
    
    # Create segment data
    segment = candle_data.iloc[pat.start_i:pat.break_i + 1 + pad].copy()
    
    # Ensure proper data types
    for col in ['open', 'high', 'low', 'close']:
        if col in segment.columns:
            segment[col] = pd.to_numeric(segment[col], errors='coerce').astype(np.float64)
    
    segment = segment.dropna()
    
    if len(segment) < 2:
        print("Insufficient data for plotting")
        return
    
    # Create pattern lines
    pattern_lines = [
        [(idx[pat.start_i], float(pat.neck_start)), (idx[pat.l_shoulder], float(pat.l_shoulder_p))],
        [(idx[pat.l_shoulder], float(pat.l_shoulder_p)), (idx[pat.l_armpit], float(pat.l_armpit_p))],
        [(idx[pat.l_armpit], float(pat.l_armpit_p)), (idx[pat.head], float(pat.head_p))],
        [(idx[pat.head], float(pat.head_p)), (idx[pat.r_armpit], float(pat.r_armpit_p))],
        [(idx[pat.r_armpit], float(pat.r_armpit_p)), (idx[pat.r_shoulder], float(pat.r_shoulder_p))],
        [(idx[pat.r_shoulder], float(pat.r_shoulder_p)), (idx[pat.break_i], float(pat.neck_end))],
        [(idx[pat.start_i], float(pat.neck_start)), (idx[pat.break_i], float(pat.neck_end))]  # Neckline
    ]
    
    # Create PIPs scatter plot if available
    addplots = []
    if pat.pip_indices and pat.pip_prices:
        # Filter PIPs within the segment range
        segment_pips_prices = []
        for pip_idx, pip_price in zip(pat.pip_indices, pat.pip_prices):
            if pat.start_i <= pip_idx <= pat.break_i + pad:
                segment_pips_prices.append(float(pip_price))
        
        if segment_pips_prices:
            pip_scatter = mpf.make_addplot(
                segment_pips_prices[:len(segment)],  # Ensure matching length
                type='scatter',
                markersize=50,
                marker='o',
                color='cyan',
                panel=0
            )
            addplots.append(pip_scatter)
    
    # Configure style
    mc = mpf.make_marketcolors(up='g', down='r')
    style = mpf.make_mpf_style(marketcolors=mc)
    
    # Plot title
    pattern_type = "Inverse H&S (Bullish)" if pat.inverted else "H&S (Bearish)"
    title = (f"{pattern_type} - PIPs Detected\n"
             f"Strength: {pat.pattern_strength:.3f} | "
             f"{idx[pat.start_i].strftime('%Y-%m-%d %H:%M')} - "
             f"{idx[pat.break_i].strftime('%Y-%m-%d %H:%M')}")
    
    try:
        # Plot with pattern lines and PIPs
        mpf.plot(
            segment,
            type='candle',
            style=style,
            title=title,
            ylabel='Price',
            addplot=addplots if addplots else None,
            alines=dict(
                alines=pattern_lines,
                colors=['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'red'],
                linewidths=[1, 1, 1, 1, 1, 1, 2]
            ),
            show_nontrading=False,
            tight_layout=True
        )
    except Exception as e:
        print(f"Plotting error: {e}")
        # Fallback basic plot
        mpf.plot(
            segment,
            type='candle',
            style=style,
            title=title,
            ylabel='Price',
            show_nontrading=False
        )

def fetch_ohlc_from_yf(symbol: str, start: str, end: str, interval: str = '1h', 
                      log_prices: bool = True) -> pd.DataFrame:
    """Download OHLC data from Yahoo Finance with proper formatting."""
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol} with interval {interval}.")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Rename columns
    df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    
    # Ensure proper data types
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
    
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    
    if log_prices:
        df = np.log(df)
    
    return df

# Example usage
if __name__ == '__main__':
    # Fetch data
    symbol = 'NQ=F'
    start_date = '2025-06-01'
    end_date = '2025-07-20'
    interval = '5m'
    
    print(f"Fetching {symbol} data using PIPs-based H&S detection...")
    data = fetch_ohlc_from_yf(symbol, start_date, end_date, interval, log_prices=True)
    
    # Extract close prices for pattern detection
    close_prices = data['close'].to_numpy()
    
    # Find patterns using PIPs
    print("Detecting H&S patterns using PIPs algorithm...")
    hs_patterns, ihs_patterns = find_hs_patterns_pips(
        close_prices, 
        window=70,      # Larger window for PIPs analysis
        n_pips=12,       # More PIPs for better pattern detection
        dist_measure=5,  # Perpendicular distance
        min_pattern_bars=20
    )
    
    print(f"\nResults:")
    print(f"Regular H&S patterns found: {len(hs_patterns)}")
    print(f"Inverse H&S patterns found: {len(ihs_patterns)}")
    
    # Plot patterns
    for i, pattern in enumerate(hs_patterns[:3]):  # Plot first 3 regular H&S
        print(f"\nPlotting Regular H&S Pattern {i+1}")
        print(f"Strength: {pattern.pattern_strength:.3f}")
        plot_hs_pips(data, pattern, pad=5)
    
    for i, pattern in enumerate(ihs_patterns[:3]):  # Plot first 3 inverse H&S
        print(f"\nPlotting Inverse H&S Pattern {i+1}")
        print(f"Strength: {pattern.pattern_strength:.3f}")
        plot_hs_pips(data, pattern, pad=5)
