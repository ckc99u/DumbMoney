import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
def get_nq_data(days=60):
    """
    Get NQ futures data. Since NQ1 15k isn't directly available from free sources,
    we'll use NQ=F (continuous futures) as a proxy
    """
    try:
        import yfinance as yf
        # Download NQ futures data (15-minute intervals)
        ticker = "NQ=F"  # Nasdaq 100 E-mini futures
        data = yf.download(ticker, period=f"{days}d", interval="15m")
        data.dropna(inplace=True)
        return data
    except ImportError:
        # Generate sample data if yfinance is not available
        print("yfinance not available, generating sample NQ-like data...")
        return generate_sample_data()

def generate_sample_data(num_bars=2000):
    """Generate sample OHLC data that resembles NQ futures for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='15min')
    
    # Start price around typical NQ levels
    base_price = 15000
    prices = [base_price]
    
    # Generate realistic price movements
    for i in range(num_bars - 1):
        change = np.random.normal(0, 20)  # 20-point average moves
        new_price = prices[-1] + change
        prices.append(max(new_price, 10000))  # Floor price
    
    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        high_offset = abs(np.random.normal(0, 15))
        low_offset = abs(np.random.normal(0, 15))
        close_change = np.random.normal(0, 10)
        
        open_price = price
        high_price = price + high_offset
        low_price = price - low_offset
        close_price = price + close_change
        
        data.append([open_price, high_price, low_price, close_price, 
                    np.random.randint(1000, 5000)])  # Volume
    
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    df.index = dates
    return df

def plot_patterns_with_candlesticks(data, patterns):
    """Create an interactive candlestick chart with pattern annotations"""
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='NQ 15min',
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    # Add pattern annotations
    for pattern in patterns:
        if pattern['pattern_type'] == 'Double Top':
            # Mark the double top pattern
            first_idx = pattern['first_peak']
            trough_idx = pattern['trough']
            second_idx = pattern['second_peak']
            
            # Add markers for peaks
            fig.add_trace(go.Scatter(
                x=[data.index[first_idx], data.index[second_idx]],
                y=[data['High'].iloc[first_idx], data['High'].iloc[second_idx]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down'),
                name='Double Top',
                showlegend=True
            ))
            
            # Add connecting line
            fig.add_trace(go.Scatter(
                x=[data.index[first_idx], data.index[trough_idx], data.index[second_idx]],
                y=[data['High'].iloc[first_idx], data['Low'].iloc[trough_idx], 
                   data['High'].iloc[second_idx]],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Double Top Pattern',
                showlegend=False
            ))
            
            # Add SELL annotation
            fig.add_annotation(
                x=data.index[second_idx],
                y=data['High'].iloc[second_idx],
                text="SELL",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(color="red", size=12, family="Arial Black")
            )
        
        elif pattern['pattern_type'] == 'Double Bottom':
            # Mark the double bottom pattern
            first_idx = pattern['first_trough']
            peak_idx = pattern['peak']
            second_idx = pattern['second_trough']
            
            # Add markers for troughs
            fig.add_trace(go.Scatter(
                x=[data.index[first_idx], data.index[second_idx]],
                y=[data['Low'].iloc[first_idx], data['Low'].iloc[second_idx]],
                mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                name='Double Bottom',
                showlegend=True
            ))
            
            # Add connecting line
            fig.add_trace(go.Scatter(
                x=[data.index[first_idx], data.index[peak_idx], data.index[second_idx]],
                y=[data['Low'].iloc[first_idx], data['High'].iloc[peak_idx], 
                   data['Low'].iloc[second_idx]],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='Double Bottom Pattern',
                showlegend=False
            ))
            
            # Add BUY annotation
            fig.add_annotation(
                x=data.index[second_idx],
                y=data['Low'].iloc[second_idx],
                text="BUY",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                font=dict(color="green", size=12, family="Arial Black")
            )
    
    # Update layout
    fig.update_layout(
        title='NQ Futures 15-Minute Chart - Double Top/Bottom Pattern Detection',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig


class DoublePatternDetector:
    def __init__(self, price_tolerance=0.008, min_separation=10, lookback_window=5):
        """
        Initialize the Double Top/Bottom Pattern Detector
        
        Parameters:
        price_tolerance: Maximum price difference between peaks/troughs (as percentage)
        min_separation: Minimum bars between pattern points
        lookback_window: Window size for finding local extrema
        """
        self.price_tolerance = price_tolerance
        self.min_separation = min_separation
        self.lookback_window = lookback_window
    
    def find_local_extrema(self, data, column):
        """Find local maxima and minima in the data"""
        if column == 'High':
            local_extrema = argrelextrema(data[column].values, np.greater_equal, 
                                        order=self.lookback_window)[0]
        else:  # Low
            local_extrema = argrelextrema(data[column].values, np.less_equal, 
                                        order=self.lookback_window)[0]
        return local_extrema
    
    def detect_double_top(self, data):
        """
        Detect double top patterns (Sell signals)
        Pattern: High-Low-High where both highs are approximately equal
        """
        highs_idx = self.find_local_extrema(data, 'High')
        lows_idx = self.find_local_extrema(data, 'Low')
        
        double_tops = []
        
        for i in range(len(highs_idx) - 1):
            first_high_idx = int(highs_idx[i])  # Ensure integer indexing
            second_high_idx = int(highs_idx[i + 1])  # Ensure integer indexing
            
            # Check minimum separation
            if second_high_idx - first_high_idx < self.min_separation:
                continue
            
            # Use .iloc with integer indices and convert to scalar with .item()
            first_high_price = data['High'].iloc[first_high_idx]
            second_high_price = data['High'].iloc[second_high_idx]
            
            # Ensure these are scalar values
            if hasattr(first_high_price, 'item'):
                first_high_price = first_high_price.item()
            if hasattr(second_high_price, 'item'):
                second_high_price = second_high_price.item()
            
            # Calculate price difference and average (now guaranteed scalars)
            price_diff = abs(first_high_price - second_high_price)
            avg_price = (first_high_price + second_high_price) / 2
            
            # Safe boolean comparison with scalar values
            if price_diff / avg_price <= self.price_tolerance:
                # Find the low between the two highs
                intermediate_lows = [int(idx) for idx in lows_idx 
                                   if first_high_idx < idx < second_high_idx]
                
                if intermediate_lows:
                    low_idx = min(intermediate_lows, 
                                key=lambda x: data['Low'].iloc[x])
                    low_price = data['Low'].iloc[low_idx]
                    
                    # Ensure scalar value
                    if hasattr(low_price, 'item'):
                        low_price = low_price.item()
                    
                    # Ensure the intermediate low is significantly lower
                    if (first_high_price - low_price) / first_high_price > 0.01:
                        double_tops.append({
                            'pattern_type': 'Double Top',
                            'first_peak': first_high_idx,
                            'trough': low_idx,
                            'second_peak': second_high_idx,
                            'signal': 'SELL',
                            'confirmation_idx': second_high_idx,
                            'price_level': avg_price
                        })
        
        return double_tops
    
    def detect_double_bottom(self, data):
        """
        Detect double bottom patterns (Buy signals)
        Pattern: Low-High-Low where both lows are approximately equal
        """
        highs_idx = self.find_local_extrema(data, 'High')
        lows_idx = self.find_local_extrema(data, 'Low')
        
        double_bottoms = []
        
        for i in range(len(lows_idx) - 1):
            first_low_idx = int(lows_idx[i])  # Ensure integer indexing
            second_low_idx = int(lows_idx[i + 1])  # Ensure integer indexing
            
            # Check minimum separation
            if second_low_idx - first_low_idx < self.min_separation:
                continue
            
            # Use .iloc with integer indices and convert to scalar
            first_low_price = data['Low'].iloc[first_low_idx]
            second_low_price = data['Low'].iloc[second_low_idx]
            
            # Ensure these are scalar values
            if hasattr(first_low_price, 'item'):
                first_low_price = first_low_price.item()
            if hasattr(second_low_price, 'item'):
                second_low_price = second_low_price.item()
            
            # Calculate price difference and average (now guaranteed scalars)
            price_diff = abs(first_low_price - second_low_price)
            avg_price = (first_low_price + second_low_price) / 2
            
            # Safe boolean comparison with scalar values
            if price_diff / avg_price <= self.price_tolerance:
                # Find the high between the two lows
                intermediate_highs = [int(idx) for idx in highs_idx 
                                    if first_low_idx < idx < second_low_idx]
                
                if intermediate_highs:
                    high_idx = max(intermediate_highs, 
                                 key=lambda x: data['High'].iloc[x])
                    high_price = data['High'].iloc[high_idx]
                    
                    # Ensure scalar value
                    if hasattr(high_price, 'item'):
                        high_price = high_price.item()
                    
                    # Ensure the intermediate high is significantly higher
                    if (high_price - first_low_price) / first_low_price > 0.01:
                        double_bottoms.append({
                            'pattern_type': 'Double Bottom',
                            'first_trough': first_low_idx,
                            'peak': high_idx,
                            'second_trough': second_low_idx,
                            'signal': 'BUY',
                            'confirmation_idx': second_low_idx,
                            'price_level': avg_price
                        })
        
        return double_bottoms
    
    def detect_patterns(self, data):
        """Detect both double top and double bottom patterns"""
        double_tops = self.detect_double_top(data)
        double_bottoms = self.detect_double_bottom(data)
        return double_tops + double_bottoms

# Alternative robust approach using numpy operations
class RobustDoublePatternDetector:
    def __init__(self, price_tolerance=0.008, min_separation=10, lookback_window=5):
        self.price_tolerance = price_tolerance
        self.min_separation = min_separation
        self.lookback_window = lookback_window
    
    def find_local_extrema(self, data, column):
        """Find local maxima and minima using numpy arrays directly"""
        values = data[column].values  # Convert to numpy array
        if column == 'High':
            extrema_idx = argrelextrema(values, np.greater_equal, 
                                      order=self.lookback_window)[0]
        else:
            extrema_idx = argrelextrema(values, np.less_equal, 
                                      order=self.lookback_window)[0]
        return extrema_idx
    
    def detect_double_top(self, data):
        """Robust double top detection using numpy operations"""
        highs_idx = self.find_local_extrema(data, 'High')
        lows_idx = self.find_local_extrema(data, 'Low')
        
        # Convert to numpy arrays for robust indexing
        high_values = data['High'].values
        low_values = data['Low'].values
        
        double_tops = []
        
        for i in range(len(highs_idx) - 1):
            first_high_idx = highs_idx[i]
            second_high_idx = highs_idx[i + 1]
            
            if second_high_idx - first_high_idx < self.min_separation:
                continue
            
            # Direct numpy array access ensures scalar values
            first_high_price = high_values[first_high_idx]
            second_high_price = high_values[second_high_idx]
            
            price_diff = abs(first_high_price - second_high_price)
            avg_price = (first_high_price + second_high_price) / 2
            
            # Now guaranteed to be scalar comparison
            if price_diff / avg_price <= self.price_tolerance:
                intermediate_lows = lows_idx[
                    (lows_idx > first_high_idx) & (lows_idx < second_high_idx)
                ]
                
                if len(intermediate_lows) > 0:
                    low_idx = intermediate_lows[np.argmin(low_values[intermediate_lows])]
                    low_price = low_values[low_idx]
                    
                    if (first_high_price - low_price) / first_high_price > 0.01:
                        double_tops.append({
                            'pattern_type': 'Double Top',
                            'first_peak': int(first_high_idx),
                            'trough': int(low_idx),
                            'second_peak': int(second_high_idx),
                            'signal': 'SELL',
                            'confirmation_idx': int(second_high_idx),
                            'price_level': float(avg_price)
                        })
        
        return double_tops
    
    def detect_double_bottom(self, data):
        """Robust double bottom detection using numpy operations"""
        highs_idx = self.find_local_extrema(data, 'High')
        lows_idx = self.find_local_extrema(data, 'Low')
        
        # Convert to numpy arrays for robust indexing
        high_values = data['High'].values
        low_values = data['Low'].values
        
        double_bottoms = []
        
        for i in range(len(lows_idx) - 1):
            first_low_idx = lows_idx[i]
            second_low_idx = lows_idx[i + 1]
            
            if second_low_idx - first_low_idx < self.min_separation:
                continue
            
            # Direct numpy array access ensures scalar values
            first_low_price = low_values[first_low_idx]
            second_low_price = low_values[second_low_idx]
            
            price_diff = abs(first_low_price - second_low_price)
            avg_price = (first_low_price + second_low_price) / 2
            
            # Now guaranteed to be scalar comparison
            if price_diff / avg_price <= self.price_tolerance:
                intermediate_highs = highs_idx[
                    (highs_idx > first_low_idx) & (highs_idx < second_low_idx)
                ]
                
                if len(intermediate_highs) > 0:
                    high_idx = intermediate_highs[np.argmax(high_values[intermediate_highs])]
                    high_price = high_values[high_idx]
                    
                    if (high_price - first_low_price) / first_low_price > 0.01:
                        double_bottoms.append({
                            'pattern_type': 'Double Bottom',
                            'first_trough': int(first_low_idx),
                            'peak': int(high_idx),
                            'second_trough': int(second_low_idx),
                            'signal': 'BUY',
                            'confirmation_idx': int(second_low_idx),
                            'price_level': float(avg_price)
                        })
        
        return double_bottoms
    
    def detect_patterns(self, data):
        """Detect both patterns"""
        double_tops = self.detect_double_top(data)
        double_bottoms = self.detect_double_bottom(data)
        return double_tops + double_bottoms


def main():
    """Main execution function with error handling"""
    print("NQ Futures Double Top/Bottom Pattern Detection")
    print("=" * 50)
    
    # Use the robust detector
    detector = RobustDoublePatternDetector(
        price_tolerance=0.008,
        min_separation=10,
        lookback_window=5
    )
    
    # Get NQ data
    print("Fetching NQ futures data...")
    data = get_nq_data(days=60)
    print(f"Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    
    # Detect patterns with error handling
    print("\nDetecting double top/bottom patterns...")
    try:
        patterns = detector.detect_patterns(data)
        print(f"Successfully detected {len(patterns)} patterns")
    except Exception as e:
        print(f"Error in pattern detection: {e}")
        return None, [], None
    
    # Display results
    if patterns:
        print(f"\nFound {len(patterns)} pattern(s):")
        for i, pattern in enumerate(patterns, 1):
            print(f"\n{i}. {pattern['pattern_type']} - {pattern['signal']}")
            print(f"   Confirmation: {data.index[pattern['confirmation_idx']]}")
            print(f"   Price Level: ${pattern['price_level']:.2f}")
    else:
        print("\nNo double top/bottom patterns found in the current dataset.")
    
    # Create and display chart
    print("\nGenerating candlestick chart with pattern annotations...")
    fig = plot_patterns_with_candlesticks(data, patterns)
    fig.show()
    fig.write_html("nq_patterns.html")
    print("Chart saved as 'nq_patterns.html'")
    
    return data, patterns, fig

if __name__ == "__main__":
    data, patterns, chart = main()
