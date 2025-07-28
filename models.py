import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

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
    action: str = None
    exit_reason: str = None
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
    avg_trade_duration: float
    max_consecutive_wins: int
    max_consecutive_losses: int
