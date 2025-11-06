"""Bounded Counterfactual Engine"""
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Literal
import logging

logger = logging.getLogger(__name__)

class CounterfactualEngine:
    """Compute realistic what-if scenarios with bounded constraints"""
    
    def __init__(self, config: Dict, costs_config: Dict):
        self.config = config
        self.costs_config = costs_config
        
        cf_config = config.get('counterfactual', {})
        self.entry_shifts = cf_config.get('entry_shifts', [-2, -1, 0, 1, 2, 3])
        self.exit_modes = cf_config.get('exit_modes', ['tp_sl', 'time_8', 'time_12', 'atr_trail'])
        self.tp_multipliers = cf_config.get('tp_multipliers', [0.75, 1.0, 1.25])
        self.sl_multipliers = cf_config.get('sl_multipliers', [0.75, 1.0, 1.25])
        
        # Cost params
        self.total_cost_bps = costs_config.get('fees', {}).get('equity', {}).get('total', 6.3) + \
                              costs_config.get('slippage', {}).get('base_bps', 2.0)
        self.cost_decimal = self.total_cost_bps / 10000
    
    def simulate_trade(
        self,
        bars: pl.DataFrame,
        entry_idx: int,
        entry_shift: int,
        exit_mode: str,
        tp_mult: float,
        sl_mult: float,
        side: str = 'long',
        base_tp_pct: float = 0.015,
        base_sl_pct: float = 0.0075
    ) -> Dict:
        """Simulate a single trade scenario
        
        Args:
            bars: DataFrame with OHLC data
            entry_idx: Original entry bar index
            entry_shift: Bars to shift entry (negative = earlier, positive = later)
            exit_mode: Exit strategy ('tp_sl', 'time_N', 'atr_trail')
            tp_mult: TP multiplier
            sl_mult: SL multiplier
            side: 'long' or 'short'
            base_tp_pct: Base take-profit %
            base_sl_pct: Base stop-loss %
            
        Returns:
            Dict with simulation results
        """
        # Adjust entry
        actual_entry_idx = max(0, min(entry_idx + entry_shift, len(bars) - 2))
        entry_price = bars[actual_entry_idx, 'c']
        
        # Adjusted TP/SL levels
        tp_pct = base_tp_pct * tp_mult
        sl_pct = base_sl_pct * sl_mult
        
        if side == 'long':
            tp_price = entry_price * (1 + tp_pct - self.cost_decimal)
            sl_price = entry_price * (1 - sl_pct - self.cost_decimal)
        else:
            tp_price = entry_price * (1 - tp_pct - self.cost_decimal)
            sl_price = entry_price * (1 + sl_pct + self.cost_decimal)
        
        # Parse exit mode
        if exit_mode.startswith('time_'):
            max_hold_bars = int(exit_mode.split('_')[1])
        elif exit_mode == 'tp_sl':
            max_hold_bars = 20  # default
        else:  # atr_trail
            max_hold_bars = 30
        
        # Simulate forward from entry
        exit_idx = actual_entry_idx
        exit_price = entry_price
        exit_reason = 'timeout'
        
        for i in range(actual_entry_idx + 1, min(actual_entry_idx + max_hold_bars + 1, len(bars))):
            high = bars[i, 'h']
            low = bars[i, 'l']
            close = bars[i, 'c']
            
            if exit_mode == 'atr_trail':
                # Dynamic trailing stop based on ATR
                atr = bars[i, 'atr'] if 'atr' in bars.columns else (high - low)
                if side == 'long':
                    sl_price = max(sl_price, close - 2 * atr)
                else:
                    sl_price = min(sl_price, close + 2 * atr)
            
            # Check exits
            if side == 'long':
                if high >= tp_price:
                    exit_price = tp_price
                    exit_idx = i
                    exit_reason = 'tp'
                    break
                if low <= sl_price:
                    exit_price = sl_price
                    exit_idx = i
                    exit_reason = 'sl'
                    break
            else:
                if low <= tp_price:
                    exit_price = tp_price
                    exit_idx = i
                    exit_reason = 'tp'
                    break
                if high >= sl_price:
                    exit_price = sl_price
                    exit_idx = i
                    exit_reason = 'sl'
                    break
            
            # Time exit
            if i == actual_entry_idx + max_hold_bars:
                exit_price = close
                exit_idx = i
                exit_reason = 'time'
                break
        
        # Compute return
        if side == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price - self.cost_decimal
        else:
            pnl_pct = (entry_price - exit_price) / entry_price - self.cost_decimal
        
        return {
            'entry_shift': entry_shift,
            'exit_mode': exit_mode,
            'tp_mult': tp_mult,
            'sl_mult': sl_mult,
            'entry_idx': actual_entry_idx,
            'entry_price': entry_price,
            'exit_idx': exit_idx,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'bars_held': exit_idx - actual_entry_idx
        }
    
    def run_grid_search(
        self,
        bars: pl.DataFrame,
        entry_idx: int,
        side: str = 'long'
    ) -> List[Dict]:
        """Run full counterfactual grid for a trade
        
        Returns:
            List of scenario results, sorted by PnL
        """
        results = []
        
        for entry_shift in self.entry_shifts:
            for exit_mode in self.exit_modes:
                for tp_mult in self.tp_multipliers:
                    for sl_mult in self.sl_multipliers:
                        result = self.simulate_trade(
                            bars, entry_idx, entry_shift, exit_mode, tp_mult, sl_mult, side
                        )
                        results.append(result)
        
        # Sort by PnL descending
        results = sorted(results, key=lambda x: x['pnl_pct'], reverse=True)
        
        logger.info(f"Counterfactual grid: {len(results)} scenarios for entry_idx={entry_idx}")
        return results
    
    def get_best_policy(self, results: List[Dict]) -> Dict:
        """Extract best admissible policy
        
        Returns:
            Best scenario dict
        """
        if not results:
            return {}
        
        return results[0]  # Already sorted by PnL
    
    def compute_delta_r(self, actual_pnl_pct: float, best_pnl_pct: float, risk_pct: float = 1.0) -> float:
        """Compute delta R (improvement in R-multiples)
        
        Args:
            actual_pnl_pct: Actual realized return
            best_pnl_pct: Best counterfactual return
            risk_pct: Initial risk per trade (default 1%)
            
        Returns:
            Delta R value
        """
        actual_r = actual_pnl_pct / risk_pct
        best_r = best_pnl_pct / risk_pct
        return best_r - actual_r
