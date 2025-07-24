import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class ProfessionalSupportResistanceDetector:
    def __init__(self, min_touches=4, max_break_percentage=0.015, 
                 return_threshold=0.02, min_coverage=0.75, max_lines_per_type=3,
                 price_tolerance=0.005, time_overlap_threshold=0.7):
        """
        Professional-grade S/R detector with overlap elimination
        
        Parameters:
        - max_lines_per_type: Maximum number of S/R lines to return
        - price_tolerance: Price similarity threshold for consolidation (0.5%)
        - time_overlap_threshold: Time overlap threshold for consolidation (70%)
        """
        self.min_touches = min_touches
        self.max_break_percentage = max_break_percentage
        self.return_threshold = return_threshold
        self.min_coverage = min_coverage
        self.max_lines_per_type = max_lines_per_type
        self.price_tolerance = price_tolerance
        self.time_overlap_threshold = time_overlap_threshold
    
    def find_pivot_points(self, data, window=5):
        """Find local highs and lows using scipy"""
        highs = argrelextrema(data['High'].values, np.greater, order=window)[0]
        lows = argrelextrema(data['Low'].values, np.less, order=window)[0]
        
        pivot_highs = [(i, data['High'].iloc[i]) for i in highs]
        pivot_lows = [(i, data['Low'].iloc[i]) for i in lows]
        
        return pivot_highs, pivot_lows
    
    def calculate_line_strength(self, data, line_slope, line_intercept, is_support=True):
        """Calculate how well a line fits the price action"""
        touches = 0
        total_candles = len(data)
        covered_candles = 0
        touch_points = []
        
        for idx, (_, row) in enumerate(data.iterrows()):
            line_price = line_slope * idx + line_intercept
            
            if is_support:
                distance = abs(row['Low'] - line_price)
                if distance <= (line_price * 0.005):
                    touches += 1
                    touch_points.append((idx, row['Low'], line_price))
                if row['Low'] >= line_price * 0.98:
                    covered_candles += 1
            else:
                distance = abs(row['High'] - line_price)
                if distance <= (line_price * 0.005):
                    touches += 1
                    touch_points.append((idx, row['High'], line_price))
                if row['High'] <= line_price * 1.02:
                    covered_candles += 1
        
        coverage_ratio = covered_candles / total_candles
        return touches, coverage_ratio, touch_points
    
    def calculate_time_overlap(self, line1, line2):
        """Calculate temporal overlap between two lines"""
        start1, end1 = line1['start_idx'], line1['end_idx']
        start2, end2 = line2['start_idx'], line2['end_idx']
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        overlap_length = overlap_end - overlap_start
        total_length = max(end1 - start1, end2 - start2)
        
        return overlap_length / total_length if total_length > 0 else 0.0
    
    def consolidate_overlapping_lines(self, lines):
        """Consolidate lines that are too similar in price and time"""
        if not lines:
            return []
        
        consolidated = []
        
        for line in lines:
            should_merge = False
            
            for i, existing in enumerate(consolidated):
                # Check price similarity
                price_diff = abs(line['start_price'] - existing['start_price']) / existing['start_price']
                
                # Check time overlap
                time_overlap = self.calculate_time_overlap(line, existing)
                
                if price_diff < self.price_tolerance and time_overlap > self.time_overlap_threshold:
                    # Merge with stronger line (more touches)
                    if line['touches'] > existing['touches']:
                        consolidated[i] = line  # Replace with stronger line
                    should_merge = True
                    break
            
            if not should_merge:
                consolidated.append(line)
        
        return consolidated
    
    def rank_line_strength(self, line, current_price=None):
        """Calculate composite strength score for line ranking"""
        base_score = (
            line['touches'] * 2.0 +                    # Touch weight
            line['coverage'] * 1.5 +                   # Coverage weight  
            (1.0 / (abs(line['slope']) + 0.001)) * 0.5 # Prefer horizontal lines
        )
        
        # Add recency bonus if we have current price context
        if current_price:
            price_relevance = 1.0 - min(
                abs(current_price - line['start_price']) / current_price,
                abs(current_price - line['end_price']) / current_price
            )
            base_score *= (1.0 + price_relevance * 0.3)
        
        return base_score
    
    def check_significant_overlap(self, line1, line2):
        """Check if two lines have significant overlap"""
        price_diff = abs(line1['start_price'] - line2['start_price']) / line2['start_price']
        time_overlap = self.calculate_time_overlap(line1, line2)
        
        return price_diff < self.price_tolerance * 2 or time_overlap > 0.3
    
    def select_best_lines(self, lines, current_price=None):
        """Select top N strongest non-overlapping lines"""
        if not lines:
            return []
        
        # Sort by strength
        sorted_lines = sorted(
            lines, 
            key=lambda x: self.rank_line_strength(x, current_price), 
            reverse=True
        )
        
        selected = []
        
        for line in sorted_lines:
            if len(selected) >= self.max_lines_per_type:
                break
                
            # Check if this line overlaps significantly with already selected
            overlaps = any(
                self.check_significant_overlap(line, sel) 
                for sel in selected
            )
            
            if not overlaps:
                selected.append(line)
        
        return selected
    
    def get_datetime_format(self, interval):
        """Determine datetime format based on interval"""
        intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        
        if interval in intraday_intervals:
            return '%Y-%m-%d %H:%M'
        else:
            return '%Y-%m-%d'
    
    def find_support_resistance_lines(self, data, interval='1d'):
        """Find potential support and resistance lines with professional filtering"""
        pivot_highs, pivot_lows = self.find_pivot_points(data)
        datetime_format = self.get_datetime_format(interval)
        current_price = data['Close'].iloc[-1]
        
        # Raw line detection
        raw_support_lines = []
        raw_resistance_lines = []
        
        # Find support lines with strategic pair selection
        if len(pivot_lows) >= 2:
            # Limit pairs to reduce computational complexity
            max_pairs = min(50, len(pivot_lows) * (len(pivot_lows) - 1) // 2)
            pair_count = 0
            
            for i in range(len(pivot_lows)):
                for j in range(i + 1, len(pivot_lows)):
                    if pair_count >= max_pairs:
                        break
                        
                    idx1, price1 = pivot_lows[i]
                    idx2, price2 = pivot_lows[j]
                    
                    # Skip pairs that are too close in time
                    if abs(idx2 - idx1) < 3:
                        continue
                    
                    slope = (price2 - price1) / (idx2 - idx1)
                    intercept = price1 - slope * idx1
                    
                    touches, coverage, touch_points = self.calculate_line_strength(
                        data, slope, intercept, is_support=True
                    )
                    
                    if touches >= self.min_touches and coverage >= self.min_coverage:
                        start_datetime = data.index[idx1].strftime(datetime_format)
                        end_datetime = data.index[idx2].strftime(datetime_format)
                        
                        raw_support_lines.append({
                            'datetime_range': f"{start_datetime} to {end_datetime}",
                            'start_price': price1,
                            'end_price': price2,
                            'touches': touches,
                            'coverage': coverage,
                            'slope': slope,
                            'intercept': intercept,
                            'start_idx': idx1,
                            'end_idx': idx2,
                            'touch_points': touch_points
                        })
                    
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
        
        # Find resistance lines with strategic pair selection
        if len(pivot_highs) >= 2:
            max_pairs = min(50, len(pivot_highs) * (len(pivot_highs) - 1) // 2)
            pair_count = 0
            
            for i in range(len(pivot_highs)):
                for j in range(i + 1, len(pivot_highs)):
                    if pair_count >= max_pairs:
                        break
                        
                    idx1, price1 = pivot_highs[i]
                    idx2, price2 = pivot_highs[j]
                    
                    # Skip pairs that are too close in time
                    if abs(idx2 - idx1) < 3:
                        continue
                    
                    slope = (price2 - price1) / (idx2 - idx1)
                    intercept = price1 - slope * idx1
                    
                    touches, coverage, touch_points = self.calculate_line_strength(
                        data, slope, intercept, is_support=False
                    )
                    
                    if touches >= self.min_touches and coverage >= self.min_coverage:
                        start_datetime = data.index[idx1].strftime(datetime_format)
                        end_datetime = data.index[idx2].strftime(datetime_format)
                        
                        raw_resistance_lines.append({
                            'datetime_range': f"{start_datetime} to {end_datetime}",
                            'start_price': price1,
                            'end_price': price2,
                            'touches': touches,
                            'coverage': coverage,
                            'slope': slope,
                            'intercept': intercept,
                            'start_idx': idx1,
                            'end_idx': idx2,
                            'touch_points': touch_points
                        })
                    
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
        
        # Apply professional filtering
        # Step 1: Consolidate overlapping lines
        consolidated_support = self.consolidate_overlapping_lines(raw_support_lines)
        consolidated_resistance = self.consolidate_overlapping_lines(raw_resistance_lines)
        
        print(f"🔧 Consolidated {len(raw_support_lines)} → {len(consolidated_support)} support lines")
        print(f"🔧 Consolidated {len(raw_resistance_lines)} → {len(consolidated_resistance)} resistance lines")
        
        # Step 2: Select best non-overlapping lines
        final_support = self.select_best_lines(consolidated_support, current_price)
        final_resistance = self.select_best_lines(consolidated_resistance, current_price)
        
        return final_support, final_resistance

def load_data_utc(symbol, interval="1d", lookback_days=365):
    """Load yfinance data and convert to UTC"""
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Clean column names
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    data.columns = [col.title() for col in data.columns]  # Standardize to Title case
    
    # Handle timezone conversion
    if data.index.tz is None:
        # Default to US Eastern for most symbols
        data.index = data.index.tz_localize('America/New_York')
    
    # Convert to UTC
    data.index = data.index.tz_convert('UTC')
    
    # Clean data
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna(subset=ohlc_cols)
    data = data.sort_index()
    
    return data

def print_professional_results(symbol, support_lines, resistance_lines, interval):
    """Print clean, professional results"""
    interval_name = {
        '1m': '1-Minute', '5m': '5-Minute', '15m': '15-Minute',
        '30m': '30-Minute', '60m': '1-Hour', '1h': '1-Hour', '1d': 'Daily'
    }.get(interval, interval)
    
    print(f"📊 {symbol} TOP PATTERN RANGES ({interval_name}) - UTC")
    print("=" * 55)
    
    print(f"\n{len(support_lines)} SUPPORT LINES:")
    for i, line in enumerate(support_lines, 1):
        strength = line['touches'] * line['coverage']
        print(f"  {i}. {line['datetime_range']} (${line['start_price']:.2f} - ${line['end_price']:.2f}) [coverage: {line['coverage']:.1f}]")

    
    print(f"\n {len(resistance_lines)} RESISTANCE LINES:")
    for i, line in enumerate(resistance_lines, 1):
        strength = line['touches'] * line['coverage']
        print(f"  {i}. {line['datetime_range']} (${line['start_price']:.2f} - ${line['end_price']:.2f}) [coverage: {line['coverage']:.1f}]")


# Professional Analysis Presets
ANALYSIS_PRESETS = {
    'conservative': {
        'min_touches': 5,
        'max_break_percentage': 0.01,
        'return_threshold': 0.025,
        'min_coverage': 0.80,
        'max_lines_per_type': 2,
        'price_tolerance': 0.003
    },
    'balanced': {
        'min_touches': 4,
        'max_break_percentage': 0.015,
        'return_threshold': 0.02,
        'min_coverage': 0.75,
        'max_lines_per_type': 3,
        'price_tolerance': 0.005
    },
    'aggressive': {
        'min_touches': 3,
        'max_break_percentage': 0.02,
        'return_threshold': 0.015,
        'min_coverage': 0.65,
        'max_lines_per_type': 4,
        'price_tolerance': 0.008
    }
}

def run_professional_analysis(symbol, interval="15m", lookback_days=5, preset='balanced'):
    """Run professional pattern analysis with overlap elimination"""
    
    print(f"🎯 PROFESSIONAL ANALYSIS - {preset.upper()} MODE")
    print(f"🔄 Loading {symbol} data ({interval}, {lookback_days} days)...")
    
    # Load data
    df = load_data_utc(symbol, interval, lookback_days)
    print(f"✅ Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    # Initialize detector with preset
    params = ANALYSIS_PRESETS[preset]
    detector = ProfessionalSupportResistanceDetector(**params)
    
    # Find patterns
    support_lines, resistance_lines = detector.find_support_resistance_lines(df, interval)
    
    # Print results
    print_professional_results(symbol, support_lines, resistance_lines, interval)
    
    return {
        'support_lines': support_lines,
        'resistance_lines': resistance_lines,
        'data': df,
        'preset': preset,
        'total_patterns': len(support_lines) + len(resistance_lines)
    }

def run_multi_symbol_analysis(symbols, interval="15m", lookback_days=5, preset='balanced'):
    """Run analysis on multiple symbols"""
    results = {}
    
    print("🎯 MULTI-SYMBOL PROFESSIONAL ANALYSIS")
    print("=" * 60)
    
    for symbol in symbols:
        print(f"\n📈 Analyzing {symbol}...")
        try:
            result = run_professional_analysis(symbol, interval, lookback_days, preset)
            results[symbol] = result
            print(f"✅ {symbol}: Found {result['total_patterns']} high-quality patterns")
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
    
    # Summary
    print(f"\n📋 ANALYSIS SUMMARY ({preset.upper()} MODE)")
    print("-" * 40)
    total_symbols = len([r for r in results.values() if r])
    total_patterns = sum(r['total_patterns'] for r in results.values() if r)
    
    print(f"Symbols Analyzed: {total_symbols}/{len(symbols)}")
    print(f"Total Quality Patterns: {total_patterns}")
    print(f"Average Patterns per Symbol: {total_patterns/total_symbols:.1f}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Single symbol with different presets
    print("=== CONSERVATIVE ANALYSIS ===")
    conservative_result = run_professional_analysis("AAPL", "15m", 3, 'conservative')
    
