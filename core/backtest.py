import pandas as pd
import numpy as np
from typing import List, Tuple
from models import BacktestMetrics, TradeResult
import math

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
            equity_curve=pd.Series([initial_capital]),
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
        avg_trade_duration=avg_trade_duration,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses
    )

def print_performance_report(metrics: BacktestMetrics):
    """Print comprehensive performance report"""
    print("\n" + "="*60)
    print(" COMPREHENSIVE BACKTEST REPORT")
    print("="*60)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f" Total Trades: {metrics.total_trades}")
    print(f" Win Rate: {metrics.win_rate:.2%}")
    print(f" Avg Trade Duration: {metrics.avg_trade_duration:.1f} bars")
    
    print(f"\n💰 PROFITABILITY:")
    print(f" Total Return: {metrics.total_return:.2%}")
    print(f" CAGR: {metrics.cagr:.2%}")
    print(f" Profit Factor: {metrics.profit_factor:.2f}")
    print(f" Profit/Loss Ratio: {metrics.profit_loss_ratio:.2f}")
    
    print(f"\n⚠️ RISK METRICS:")
    print(f" Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f" Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    
    print(f"\n🔄 STREAKS:")
    print(f" Max Consecutive Wins: {metrics.max_consecutive_wins}")
    print(f" Max Consecutive Losses: {metrics.max_consecutive_losses}")
    
    print(f"\n✅ REQUIREMENTS CHECK:")
    print(f" Max Drawdown ≤ 12%: {'✓ PASS' if abs(metrics.max_drawdown) <= 0.12 else '✗ FAIL'}")
    print(f" CAGR ≥ 18%: {'✓ PASS' if metrics.cagr >= 0.18 else '✗ FAIL'}")
    print(f" Sharpe ≥ 1.0: {'✓ PASS' if metrics.sharpe_ratio >= 1.0 else '✗ FAIL'}")
    print("="*60)

# Original backtest_pips_analysis function adapted for structure
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
    from utils.data_utils import fetch_ohlc_from_yf, load_ohlc_data_from_csv
    from utils.plotting import plot_pips_pattern
    from core.pips_functions import (
        find_pips, merge_near_collinear_pips, is_valid_standard_pattern,
        create_stop_loss_profit, analyze
    )
    import yfinance as yf
    from models import PIPsPattern
    
    print(f"Fetching {symbol} data from {start} to {end} ({interval})...")
    # # test existance
    # data = yf.download(symbol, period='5d', progress=False)
    # if not data.empty:
    #     df = fetch_ohlc_from_yf(symbol, start, end, interval, log_prices=False)
    # else:
    df = load_ohlc_data_from_csv(symbol, start, end)
    
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

        enable_trade, signal = is_valid_standard_pattern(pip_indices, pip_prices, window_data) 
        if not enable_trade:
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
        
        # if enable_debug_plot or trade_result.pnl_dollars < 0:
        #     print(signal, trade_result.pnl_dollars)
        #     plot_pips_pattern(df, pattern, symbol, interval, stop_loss, take_profit)
    
    print(f"Total windows processed: {len(patterns)} patterns, total: {total}")
    
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    metrics = calculate_comprehensive_metrics(trades, start_date, end_date, initial_capital)
    
    print_performance_report(metrics)
    
    return df, patterns, trades, metrics
