import matplotlib.pyplot as plt

def plot_transitions(df, bull_signals, bear_signals):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close Price')
    plt.scatter(df.index[bear_signals==1], df['Close'][bear_signals==1],
                marker='v', color='r', label='Bull→Bear', s=50)
    plt.scatter(df.index[bull_signals==1], df['Close'][bull_signals==1],
                marker='^', color='g', label='Bear→Bull', s=50)
    plt.legend(); plt.title('Transition Signals'); plt.show()
