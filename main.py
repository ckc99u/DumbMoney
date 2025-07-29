import logging
from typing import Dict, Any
from config import TradingConfig
from core.backtest import backtest_pips_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantitativeTradingSystem:
    """Main system interface using original functions only"""
    
    def __init__(self, custom_params: Dict[str, Any] = None):
        """Initialize with optional custom parameters"""
        if custom_params:
            self.config = TradingConfig.from_dict(custom_params)
        else:
            self.config = TradingConfig()
    
    def run_analysis(self, symbol: str, start: str, end: str, interval: str = '1h') -> Dict[str, Any]:
        """Run complete quantitative analysis using original backtest_pips_analysis"""
        logger.info("="*60)
        logger.info("QUANTITATIVE PIPs TRADING ANALYSIS")
        logger.info("="*60)
        
        try:
            # Use original backtest_pips_analysis function
            df, patterns, trades, metrics = backtest_pips_analysis(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                window=self.config.window,
                n_pips=self.config.n_pips,
                dist_measure=self.config.dist_measure,
                step_size=self.config.step_size,
                enable_debug_plot=self.config.enable_debug_plot,
                initial_capital=self.config.initial_capital,
                point_value = self.config.point_value,
                max_loss = self.config.max_loss,
                max_lot = self.config.max_lot,
                round_turn = self.config.round_turn,
            )
            
            return {
                'data': df,
                'patterns': patterns,  
                'trades': trades,
                'metrics': metrics,
                'meets_requirements': self._check_requirements(metrics)
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _check_requirements(self, metrics) -> Dict[str, bool]:
        """Check performance requirements"""
        return {
            'max_drawdown_ok': abs(metrics.max_drawdown) <= 0.12,
            'cagr_ok': metrics.cagr >= 0.18,
            'sharpe_ok': metrics.sharpe_ratio >= 1.0,
            'overall_pass': (abs(metrics.max_drawdown) <= 0.12 and 
                           metrics.cagr >= 0.18 and metrics.sharpe_ratio >= 1.0)
        }

def main():
    """Example usage - same as original"""
    custom_params = {
        'window': 60,
        'n_pips': 9, 
        'step_size': 1,
        'initial_capital': 50000,
        'point_value': 20,
        'max_loss':  800,
        'max_lot': 2,
        'round_turn': 15
    }
    
    system = QuantitativeTradingSystem(custom_params)
    results = system.run_analysis('example/NQ_2024_2025.csv', '2024-01-01', '2025-07-21', '15m') ##symbol='NQ=F',example/NQ_2024_2025.csv
    

if __name__ == '__main__':
    main()
