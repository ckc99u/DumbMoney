from .base import Strategy
import pandas as pd
from pattern import detect_bear_to_bull, detect_bull_to_bear

class NakedKStrategy(Strategy):
    def init(self):
        self.df['bull'] = detect_bear_to_bull(self.df)
        self.df['bear'] = detect_bull_to_bear(self.df)
        self.position = None

    def next(self, idx):
        price = self.df.at[idx,'Close']
        if self.position is None:
            if self.df.at[idx,'bull']:
                self.position = ('long', price, idx)
            elif self.df.at[idx,'bear']:
                self.position = ('short', price, idx)
        else:
            # Adam exit: small body + long opposite wick
            o,h,l,c = [self.df.at[idx,col] for col in ['Open','High','Low','Close']]
            body = abs(c-o); wick = (h-l)-body
            if body < 0.3*(h-l) and wick > 2*body:
                entry_type, entry_price, entry_idx = self.position
                pnl = (c-entry_price) if entry_type=='long' else (entry_price-c)
                self.positions.append((entry_idx, idx, entry_type, pnl))
                self.position=None

    def analyze(self):
        pnl = [p for *_,p in self.positions]
        returns = pd.Series(pnl)
        stats = {
            'Total Trades': len(pnl),
            'Win Rate': (returns>0).mean(),
            'Avg Win': returns[returns>0].mean(),
            'Avg Loss': returns[returns<0].mean(),
            'Profit Factor': returns[returns>0].sum()/(-returns[returns<0].sum()),
            # ... CAGR, Sharpe, MaxDD ...
        }
        return stats
