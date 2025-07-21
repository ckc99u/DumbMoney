import pandas as pd
from report import generate_report

class Backtester:
    def __init__(self, df: pd.DataFrame, strategy):
        self.df = df.copy()
        self.strat = strategy(self.df)
        self.trades = []

    def run(self):
        self.strat.init()
        for idx in self.df.index:
            self.strat.next(idx)
        self.results = self.strat.analyze()
        return self.results

    def report(self, out_path='report.html'):
        generate_report(self.results, out_path)
