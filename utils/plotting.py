import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from models import PIPsPattern

def plot_pips_pattern(df: pd.DataFrame, pattern: PIPsPattern, symbol: str, interval: str, stop_loss=None, take_profit_levels=None):
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
    
    # Define color mapping based on PIP position
    def get_pip_color(position_index):
        """Return color based on PIP position"""
        color_rules = {
            0: 'blue',  # First point = blue
            -1: 'black',  # Last point = green (handle with len check)
        }
        
        # Handle last position dynamically
        if position_index == len(pattern.pip_indices) - 3:
            return 'red'
        return color_rules.get(position_index, 'yellow')  # Default yellow
    
    # Create position-based scatter data
    position_colors = {}
    segment_pip_data = {}
    
    for i, abs_idx in enumerate(pattern.pip_indices):
        if abs_idx < len(df):
            # Convert absolute index to relative position within segment
            relative_idx = abs_idx - pattern.start_idx
            if 0 <= relative_idx < len(segment):
                color = get_pip_color(i)
                # Group by color
                if color not in position_colors:
                    position_colors[color] = []
                    segment_pip_data[color] = []
                position_colors[color].append((relative_idx, float(pattern.pip_prices[i])))
    
    # Create separate scatter plots for each color
    addplots = []
    
    stop_loss_line = pd.Series(data=stop_loss, index=segment.index)
    stop_loss_plot = mpf.make_addplot(
        stop_loss_line,
        color='red',
        linestyle='--',
        width=2,
        alpha=0.8,
        secondary_y=False)
    addplots.append(stop_loss_plot)
    
    # Add Take Profit Lines
    tp_colors = ['green', 'blue']  # Different shades for multiple TPs
    for i in range(len(take_profit_levels)):
        tp_line = pd.Series(data=take_profit_levels[i], index=segment.index)
        tp_plot = mpf.make_addplot(
            tp_line,
            color=tp_colors[i],
            linestyle=':' if i == 0 else '-.',  # Different line styles
            width=2,
            alpha=0.7,
            secondary_y=False)
        addplots.append(tp_plot)
    
    for color, positions in position_colors.items():
        if positions:
            # Create scatter data for this color
            scatter_data = pd.Series(data=np.nan, index=segment.index)
            for pos, price in positions:
                if pos < len(scatter_data):
                    scatter_data.iloc[pos] = price
            
            # Create addplot for this color group
            if not scatter_data.dropna().empty:
                # Different marker sizes for emphasis
                marker_size = 80 if color == 'red' else 50
                scatter = mpf.make_addplot(
                    scatter_data,
                    type='scatter',
                    markersize=marker_size,
                    marker='o',
                    color=color,
                    alpha=0.8
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
                colors=['blue']*len(pattern_lines),
                linewidths=2,
                alpha=0.7
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
