from data_loader import load_data
from backtester import Backtester
from strategies.nakedk_strategy import NakedKStrategy
from debug import plot_transitions

# Client input
product = 'NQ=F'
big_tf = '2023-01-01'
small_tf = '2025-07-20'

# Load data
df = load_data(product, start=big_tf, end=small_tf)

# Debug plot
strat = NakedKStrategy(df)
strat.init()
plot_transitions(df, df['bull'], df['bear'])

# Backtest
bt = Backtester(df, NakedKStrategy)
results = bt.run()
print(results)

# # Generate report
# bt.report('nq_report.html')
