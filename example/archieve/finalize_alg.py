import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from datetime import datetime
from typing import Tuple, List, Dict
from dataclasses import dataclass
import math

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

@dataclass
class TradeResult:
    """Container for trade execution results"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    action: str = None # 'Buy' or 'Sell'
    exit_reason: str = None # 'STOP_LOSS', 'TAKE_PROFIT_1', 'TAKE_PROFIT_2', 'END_OF_DATA'
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    duration_bars: int = 0
    is_winner: bool = False

@dataclass
class BacktestMetrics:
    """Container for comprehensive backtest metrics"""
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    cagr: float
    sharpe_ratio: float
    profit_loss_ratio: float
    total_return: float
    equity_curve: pd.Series
    trades_per_month: float
    avg_trade_duration: float
    max_consecutive_wins: int
    max_consecutive_losses: int

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
    close_data = data['close'].values


    pips_x = [0, len(close_data) - 1]  # Index
    pips_y = [close_data[0], close_data[-1]]  # Price
    
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
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - close_data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - close_data[i]) ** 2) ** 0.5
                elif dist_measure == 2:  # Perpendicular distance
                    d = abs((slope * i + intercept) - close_data[i]) / (slope ** 2 + 1) ** 0.5
                else:  # Vertical distance
                    d = abs((slope * i + intercept) - close_data[i])
                
                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj
        
        if md_i != -1:
            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, close_data[md_i])

    high_data = data['high'].values
    low_data = data['low'].values


    for i, (pip_index, pip_price) in enumerate(zip(pips_x, pips_y)):
        if i > 0:
            prev_price = pips_y[i - 1]
            trend = pip_price - prev_price
            if trend >= 0:
                pips_y[i] = high_data[pip_index]
            else:
                pips_y[i] = low_data[pip_index]

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

def calculate_angle_at_middle_point(x1, y1, x2, y2, x3, y3):

    vector1 = np.array([x1 - x2, y1 - y2])  # P2 -> P1
    vector2 = np.array([x3 - x2, y3 - y2])  # P2 -> P3
    
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    dot_product = np.dot(vector1, vector2)
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_radians = math.acos(cos_angle)
    return (math.degrees(angle_radians))


def is_valid_standard_pattern(py, px, window_data) -> bool:
    #check break value
    if len(px) < 6:
        return False
    rear = len(px) - 1

    pt1 = px[rear]
    pt2 = px[rear - 1]
    pt3 = px[rear - 2]
    pt4 = px[rear - 3]
    pt5 = px[rear - 4]
    pt6 = px[rear - 5]
    
    y1 = py[rear]
    y2 = py[rear - 1]
    y3 = py[rear - 2]
    y4 = py[rear - 3]
    y5 = py[rear - 4]
    y6 = py[rear - 5]
    slope_1_5 = (pt5 - pt1) / (y5 - y1)
    angle_radians_1_5 = math.atan(slope_1_5)

    angle2 = calculate_angle_at_middle_point(y1, pt1, y2, pt2, y3, pt3)
    angle4 = calculate_angle_at_middle_point(y3, pt3, y4, pt4, y5, pt5)
    angle5 = calculate_angle_at_middle_point(y4, pt4, y5, pt5, y6, pt6)

    angle3 = calculate_angle_at_middle_point(y1, pt1, y3, pt3, y5, pt5)

    if(abs(math.degrees(angle_radians_1_5)) < 20 and max(angle2, angle4, angle5) < 90 and angle3 < 100):
        if (pt3 > max(pt1, pt5) and pt2 < min(pt3, pt1) and pt4 < min(pt5, pt4)):# Bear
            return True, "Sell"
        elif (pt3 < max(pt1, pt5) and pt2 > max(pt1, pt3) and pt4 > max(pt5, pt3)): #Bull
            return True, "Buy"
    return False, "Hold"

def merge_near_collinear_pips(px: List[int], py: List[float], angle_thresh_deg=3):
    k = 1
    while k < len(px) - 1:
        angle = calculate_angle_at_middle_point(px[k-1], py[k-1],
                               px[k],   py[k],
                               px[k+1], py[k+1])
        if (angle > angle_thresh_deg):
            # remove the middle pivot
            px.pop(k)
            py.pop(k)
            # do not increment k → re-test new triple
        else:
            k += 1
    return px, py

def create_stop_loss_profit(py, px, window_data, action):
    rear = len(px) - 1
    pt1 = px[rear]
    pt2 = px[rear - 1]
    pt3 = px[rear - 2]
    pt4 = px[rear - 3]
    pt5 = px[rear - 4]
    pt6 = px[rear - 5]
    
    y1 = py[rear]
    y2 = py[rear - 1]
    y3 = py[rear - 2]
    y4 = py[rear - 3]
    y5 = py[rear - 4]
    y6 = py[rear - 5]
    
    take_profit_pos = []
    points4_6 = window_data[y6 : y4]
    points1_2 = window_data[y2 : y1]

    if action == 'Buy':
        stop_loss = min(points4_6['low'].values.min(), points1_2['low'].values.min()) * 0.999
        if (stop_loss < pt3):
            stop_loss = (pt3 + pt5 + pt1)/3
        take_profit_pos.append(max(pt2, pt4)) # first profit line
        diff = abs(take_profit_pos[0] - stop_loss)# adam theory
        take_profit_pos.append(take_profit_pos[0] + diff)
    else:
        stop_loss = max(points4_6['high'].values.min(), points1_2['high'].values.min()) * 1.001
        if (stop_loss > pt3):
            stop_loss = (pt3 + pt5 + pt1)/3
        take_profit_pos.append(min(pt2, pt4))
        diff = abs(take_profit_pos[0] - stop_loss)# adam theory
        take_profit_pos.append(take_profit_pos[0] - diff)


    return stop_loss, take_profit_pos

def assign_trade_result(price, diff_points, exit_reason, point_value, trade_result):
    trade_result.exit_price = price
    trade_result.exit_reason = exit_reason
    trade_result.pnl_points += diff_points
    trade_result.pnl_dollars += trade_result.pnl_points * point_value

    return trade_result

def analyze(data_frame, start_idx, stop_loss, take_profit_pos, action):
    
    POINT_VALUE = 20.0  # $20 per point move from Chicago Mercantile Exchange (CME) for NQ=F
    
    # Initialize trade result
    entry_time = data_frame.index[start_idx]
    entry_price = (data_frame.iloc[start_idx - 1]['low'] + data_frame.iloc[start_idx - 1]['high']) * 0.5
    
    trade_result = TradeResult(
        entry_time=entry_time,
        entry_price=entry_price,
        action=action
    )
    # Track from next bar onwards
    for i in range(start_idx + 1, len(data_frame)):
        current_bar = data_frame.iloc[i]
        current_time = data_frame.index[i]
        
        if action == 'Buy':
            if current_bar['low'] <= stop_loss:
                if trade_result.exit_reason == None:
                    reason = 'STOP_LOSS'
                else:
                    reason = 'TAKE_PROFIT_1 + STOP_LOSS'
                assign_trade_result(stop_loss, stop_loss - entry_price, reason, POINT_VALUE,  trade_result)
                return i, trade_result
            
            if current_bar['high'] >= take_profit_pos[0] or current_bar['high'] >= take_profit_pos[1]:
                if (trade_result.exit_reason == None):
                    if (current_bar['high'] >= take_profit_pos[0]):
                        assign_trade_result(take_profit_pos[0], take_profit_pos[0] - entry_price, 'TAKE_PROFIT_1', POINT_VALUE,  trade_result)
                        stop_loss = current_bar['low'] # adjust stop loss
                    else:
                        assign_trade_result(take_profit_pos[1], take_profit_pos[1] - entry_price, 'TAKE_PROFIT_2', POINT_VALUE,  trade_result)
                        return i, trade_result
                else:
                    assign_trade_result(take_profit_pos[1], take_profit_pos[1] - entry_price, 'TAKE_PROFIT_2', POINT_VALUE,  trade_result)
                    return i, trade_result
                
        else:  # 'Sell'
            # Check stop loss (price goes above stop)
            if current_bar['high'] >= stop_loss:
                if trade_result.exit_reason == None:
                    reason = 'STOP_LOSS'
                else:
                    reason = 'TAKE_PROFIT_1 + STOP_LOSS'

                assign_trade_result(stop_loss, entry_price - stop_loss, reason, POINT_VALUE,  trade_result)
                return i, trade_result

            if current_bar['low'] <= take_profit_pos[0] or current_bar['low'] <= take_profit_pos[1]:
                if (trade_result.exit_reason == None):
                    if (current_bar['low'] <= take_profit_pos[0]):
                        assign_trade_result(take_profit_pos[0], entry_price - take_profit_pos[0], 'TAKE_PROFIT_1', POINT_VALUE,  trade_result)
                        stop_loss = current_bar['high'] # adjust stop loss
                    else:
                        assign_trade_result(take_profit_pos[1], entry_price - take_profit_pos[1], 'TAKE_PROFIT_2', POINT_VALUE,  trade_result)
                        return i, trade_result
                else:
                    assign_trade_result(take_profit_pos[1], entry_price - take_profit_pos[1], 'TAKE_PROFIT_2', POINT_VALUE,  trade_result)
                    return i, trade_result
            
    
    # If we reach end of data without hitting stops or targets
    final_price = data_frame.iloc[-1]['close']
    trade_result.exit_time = data_frame.index[-1]
    trade_result.exit_price = final_price
    trade_result.exit_reason = 'END_OF_DATA'
    trade_result.duration_bars = len(data_frame) - 1 - start_idx
    
    if action == 'Buy':
        trade_result.pnl_points = final_price - entry_price
    else:
        trade_result.pnl_points = entry_price - final_price
    
    trade_result.pnl_dollars = trade_result.pnl_points * POINT_VALUE
    
    return len(data_frame), trade_result


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
            0: 'blue',       # First point = blue
            -1: 'black',     # Last point = green (handle with len check)
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


def calculate_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve"""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_cagr(equity_curve: pd.Series, total_days: int) -> float:
    """Calculate Compound Annual Growth Rate"""
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    years = total_days / 365.25  # Account for leap years
    if years <= 0 or start_value <= 0:
        return 0.0
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_consecutive_streaks(trades: List[TradeResult]) -> Tuple[int, int]:
    """Calculate maximum consecutive wins and losses"""
    if not trades:
        return 0, 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for trade in trades:
        if trade.pnl_dollars > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    
    return max_wins, max_losses

def calculate_comprehensive_metrics(trades: List[TradeResult], start_date: pd.Timestamp, end_date: pd.Timestamp, 
                                  initial_capital: float = 100000) -> BacktestMetrics:
    """Calculate comprehensive backtesting metrics"""
    
    if not trades:
        return BacktestMetrics(
            total_trades=0, win_rate=0, profit_factor=0, max_drawdown=0,
            cagr=0, sharpe_ratio=0, profit_loss_ratio=0, total_return=0,
            equity_curve=pd.Series([initial_capital]), trades_per_month=0,
            avg_trade_duration=0, max_consecutive_wins=0, max_consecutive_losses=0
        )
    
    # Basic trade statistics
    pnl_dollars = [trade.pnl_dollars for trade in trades]
    wins = [p for p in pnl_dollars if p > 0]
    losses = [p for p in pnl_dollars if p <= 0]
    
    total_trades = len(trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    # Profit metrics
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Equity curve
    cumulative_pnl = np.cumsum(pnl_dollars)
    equity_curve = pd.Series(cumulative_pnl + initial_capital)
    
    # Drawdown and returns
    max_drawdown = calculate_drawdown(equity_curve)
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    
    # Time-based metrics
    total_days = (end_date - start_date).days
    cagr = calculate_cagr(equity_curve, total_days)
    
    # Sharpe ratio
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    
    # Additional metrics
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Trade frequency
    total_months = total_days / 30.44  # Average days per month
    trades_per_month = total_trades / total_months if total_months > 0 else 0
    
    # Average trade duration
    durations = [trade.duration_bars for trade in trades if hasattr(trade, 'duration_bars')]
    avg_trade_duration = np.mean(durations) if durations else 0
    
    # Consecutive streaks
    max_wins, max_losses = calculate_consecutive_streaks(trades)
    
    return BacktestMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        cagr=cagr,
        sharpe_ratio=sharpe_ratio,
        profit_loss_ratio=profit_loss_ratio,
        total_return=total_return,
        equity_curve=equity_curve,
        trades_per_month=trades_per_month,
        avg_trade_duration=avg_trade_duration,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses
    )

def print_performance_report(metrics: BacktestMetrics):
    """Print comprehensive performance report"""
    print("\n" + "="*60)
    print("         COMPREHENSIVE BACKTEST REPORT")
    print("="*60)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Win Rate: {metrics.win_rate:.2%}")
    print(f"   Trades per Month: {metrics.trades_per_month:.1f}")
    print(f"   Avg Trade Duration: {metrics.avg_trade_duration:.1f} bars")
    
    print(f"\n💰 PROFITABILITY:")
    print(f"   Total Return: {metrics.total_return:.2%}")
    print(f"   CAGR: {metrics.cagr:.2%}")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Profit/Loss Ratio: {metrics.profit_loss_ratio:.2f}")
    
    print(f"\n⚠️  RISK METRICS:")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    
    print(f"\n🔄 STREAKS:")
    print(f"   Max Consecutive Wins: {metrics.max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {metrics.max_consecutive_losses}")
    
    print(f"\n✅ REQUIREMENTS CHECK:")
    print(f"   Max Drawdown ≤ 12%: {'✓ PASS' if abs(metrics.max_drawdown) <= 0.12 else '✗ FAIL'}")
    print(f"   CAGR ≥ 18%: {'✓ PASS' if metrics.cagr >= 0.18 else '✗ FAIL'}")
    print(f"   Sharpe ≥ 1.0: {'✓ PASS' if metrics.sharpe_ratio >= 1.0 else '✗ FAIL'}")
    
    print("="*60)

def backtest_pips_analysis(
    symbol: str,
    start: str,
    end: str,
    interval: str = '1h',
    window: int = 50,
    n_pips: int = 7,
    dist_measure: int = 2,
    step_size: int = 10,
    enable_debug_plot: bool = False,
    initial_capital: float = 100000
):
    """Enhanced backtest function with comprehensive metrics"""
    
    print(f"Fetching {symbol} data from {start} to {end} ({interval})...")
    df = fetch_ohlc_from_yf(symbol, start, end, interval, log_prices=False)
    
    patterns = []
    trades = []
    i = window
    total = 0
    while i < len(df):
        start_idx = i - window
        end_idx = i
        window_data = df[start_idx:end_idx]

        pip_indices, pip_prices = find_pips(window_data, n_pips, dist_measure)
        pip_indices, pip_prices = merge_near_collinear_pips(
            pip_indices, pip_prices, angle_thresh_deg=130)

        result, signal = is_valid_standard_pattern(pip_indices, pip_prices, window_data)
        if not result:
            i += step_size
            continue

        stop_loss, take_profit = create_stop_loss_profit(pip_indices, pip_prices, window_data, signal)
        next_iter, trade_result = analyze(df, end_idx, stop_loss, take_profit, signal)

        # Calculate trade duration
        trade_result.duration_bars = next_iter - end_idx
        trade_result.is_winner = trade_result.pnl_dollars > 0

        i = next_iter + window - 1
        abs_pip_indices = [start_idx + idx for idx in pip_indices]

        pattern = PIPsPattern(
            start_idx=start_idx,
            end_idx=end_idx,
            pip_indices=abs_pip_indices,
            pip_prices=pip_prices,
            window_data=window_data,
            timestamp_start=df.index[start_idx],
            timestamp_end=df.index[end_idx-1]
        )
        total += trade_result.pnl_dollars
        patterns.append(pattern)
        trades.append(trade_result)

        if enable_debug_plot:
            plot_pips_pattern(df, pattern, symbol, interval, stop_loss, take_profit)
    print(f"Total windows processed: {len(patterns)} patterns, total: {total}")
    # Calculate comprehensive metrics
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    metrics = calculate_comprehensive_metrics(trades, start_date, end_date, initial_capital)
    
    # Print performance report
    print_performance_report(metrics)
    
    return df, patterns, trades, metrics


# Example usage and testing
if __name__ == '__main__':
    # Test with 2-year period as required
    df_result, patterns_result, trades_result, metrics_result = backtest_pips_analysis(
        symbol='NQ=F',
        start='2025-06-01',  # 2-year backtest period
        end='2025-07-15',
        interval='15m',       # Can be changed to '15m', '4h', '5m' as noted
        window=50,
        n_pips=12,
        dist_measure=2,
        step_size=1,
        enable_debug_plot=False,
        initial_capital=50000
    )
    
    # Optional: Create equity curve plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(metrics_result.equity_curve.index, metrics_result.equity_curve.values)
    # plt.title(f'Equity Curve - Total Return: {metrics_result.total_return:.2%}')
    # plt.xlabel('Trade Number')
    # plt.ylabel('Portfolio Value ($)')
    # plt.grid(True)
    # plt.show()
