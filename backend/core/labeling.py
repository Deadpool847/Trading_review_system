"""Triple-Barrier Labeling with Costs"""
import polars as pl
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    """Triple-barrier meta-labeling with realistic costs"""
    
    def __init__(self, config: Dict, costs_config: Dict):
        self.config = config
        self.costs_config = costs_config
        
        # Barrier params
        self.tp_pct = config.get('target_profit_pct', 1.5) / 100
        self.sl_pct = config.get('target_loss_pct', 0.75) / 100
        self.timeout_bars = config.get('timeout_bars', 16)
        
        # Costs
        self.total_fees_bps = costs_config.get('fees', {}).get('equity', {}).get('total', 6.3)
        self.slippage_base_bps = costs_config.get('slippage', {}).get('base_bps', 2.0)
    
    def compute_total_cost_bps(self, avg_volume: float = None, order_qty: float = None) -> float:
        """Compute total transaction cost in basis points"""
        # Base cost (fees + base slippage)
        total_cost = self.total_fees_bps + self.slippage_base_bps
        
        # Add market impact if volume data available
        if avg_volume and order_qty:
            impact_coef = self.costs_config.get('slippage', {}).get('impact_coef', 5.0)
            max_slip = self.costs_config.get('slippage', {}).get('max_bps', 20.0)
            impact = impact_coef * np.sqrt(order_qty / max(avg_volume, 1))
            total_cost += min(impact, max_slip)
        
        return total_cost / 10000  # Convert to decimal
    
    def label_bars(
        self,
        df: pl.DataFrame,
        side: str = 'long',
        entry_idx: int = 0
    ) -> Tuple[int, float, str]:
        """Apply triple-barrier method on bars starting from entry_idx
        
        Args:
            df: DataFrame with OHLC data
            side: 'long' or 'short'
            entry_idx: Index to start barrier logic
            
        Returns:
            (exit_bar_idx, realized_return, exit_reason)
        """
        if entry_idx >= len(df) - 1:
            return entry_idx, 0.0, 'no_exit'
        
        entry_price = df[entry_idx, 'c']
        cost = self.compute_total_cost_bps()
        
        # Set barriers
        if side == 'long':
            tp_price = entry_price * (1 + self.tp_pct - cost)
            sl_price = entry_price * (1 - self.sl_pct - cost)
        else:
            tp_price = entry_price * (1 - self.tp_pct - cost)
            sl_price = entry_price * (1 + self.sl_pct + cost)
        
        # Scan forward bars
        max_idx = min(entry_idx + self.timeout_bars, len(df) - 1)
        
        for i in range(entry_idx + 1, max_idx + 1):
            high = df[i, 'h']
            low = df[i, 'l']
            close = df[i, 'c']
            
            if side == 'long':
                # Check TP hit
                if high >= tp_price:
                    realized_return = (tp_price - entry_price) / entry_price - cost
                    return i, realized_return, 'tp'
                # Check SL hit
                if low <= sl_price:
                    realized_return = (sl_price - entry_price) / entry_price - cost
                    return i, realized_return, 'sl'
            else:  # short
                # Check TP hit
                if low <= tp_price:
                    realized_return = (entry_price - tp_price) / entry_price - cost
                    return i, realized_return, 'tp'
                # Check SL hit
                if high >= sl_price:
                    realized_return = (entry_price - sl_price) / entry_price - cost
                    return i, realized_return, 'sl'
        
        # Timeout exit
        exit_price = df[max_idx, 'c']
        if side == 'long':
            realized_return = (exit_price - entry_price) / entry_price - cost
        else:
            realized_return = (entry_price - exit_price) / entry_price - cost
        
        return max_idx, realized_return, 'timeout'
    
    def create_meta_labels(
        self,
        df: pl.DataFrame,
        trades_df: pl.DataFrame = None
    ) -> pl.DataFrame:
        """Create meta-labeling dataset
        
        For each bar with a base signal, determine if taking the trade would be profitable
        under triple-barrier logic.
        
        Args:
            df: Feature-rich DataFrame
            trades_df: Optional actual trades to align with
            
        Returns:
            DataFrame with meta_label column (1=take, 0=skip)
        """
        if 'base_score' not in df.columns:
            logger.warning("No base_score column found, cannot create meta labels")
            return df
        
        # Filter bars with positive base score (signal candidates)
        signal_mask = df['base_score'] > 0.5
        signal_indices = df.with_row_index().filter(signal_mask)['index'].to_list()
        
        labels = []
        for idx in signal_indices:
            exit_idx, realized_return, reason = self.label_bars(df, side='long', entry_idx=idx)
            label = 1 if realized_return > 0 else 0
            labels.append({
                'index': idx,
                'meta_label': label,
                'realized_return': realized_return,
                'exit_reason': reason
            })
        
        # Create labels DataFrame and join
        labels_df = pl.DataFrame(labels)
        df = df.with_row_index()
        df = df.join(labels_df, on='index', how='left')
        df = df.drop('index')
        
        # Fill unlabeled rows
        df = df.with_columns([
            pl.col('meta_label').fill_null(0),
            pl.col('realized_return').fill_null(0),
            pl.col('exit_reason').fill_null('no_signal')
        ])
        
        logger.info(f"Created {len(labels)} meta labels, positive: {sum([l['meta_label'] for l in labels])}")
        return df
