import pandas as pd

class Strategy:
    def __init__(self, df: pd.DataFrame):
        self.df = df; self.positions=[]
    def init(self): pass
    def next(self, idx): pass
    def analyze(self) -> dict:
        # Compute PnL, CAGR, Sharpe, MaxDD, Trades, WinRate
        return {}
