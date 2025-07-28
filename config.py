from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    """Configuration using original parameter names"""
    # Original parameters from finalize.py
    window: int = 50
    n_pips: int = 12
    dist_measure: int = 2
    step_size: int = 1
    enable_debug_plot: bool = False
    initial_capital: float = 50000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
