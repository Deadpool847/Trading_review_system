"""KPI Computation Module"""
import polars as pl
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class KPIComputer:
    """Compute trading performance KPIs"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def compute_trade_kpis(self, trades_df: pl.DataFrame) -> Dict:
        """Compute KPIs from trades DataFrame
        
        Expected columns: pnl_pct, exit_reason, bars_held, etc.
        
        Returns:
            Dict with KPI values
        """
        if len(trades_df) == 0:
            return self._empty_kpis()
        
        try:
            # Basic metrics
            total_trades = len(trades_df)
            winners = trades_df.filter(pl.col('pnl_pct') > 0)
            losers = trades_df.filter(pl.col('pnl_pct') <= 0)
            
            win_rate = len(winners) / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = trades_df['pnl_pct'].sum()
            avg_pnl = trades_df['pnl_pct'].mean()
            
            avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
            avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
            
            # Risk metrics
            profit_factor = abs(winners['pnl_pct'].sum() / losers['pnl_pct'].sum()) if len(losers) > 0 and losers['pnl_pct'].sum() != 0 else 0
            
            # R-multiples (assuming 1% risk per trade)
            trades_df = trades_df.with_columns([
                (pl.col('pnl_pct') / 0.01).alias('r_multiple')
            ])
            avg_r = trades_df['r_multiple'].mean()
            
            # Exit reason breakdown
            exit_counts = trades_df.group_by('exit_reason').count().sort('count', descending=True)
            exit_breakdown = {row['exit_reason']: row['count'] for row in exit_counts.to_dicts()}
            
            # Holding period
            if 'bars_held' in trades_df.columns:
                avg_bars_held = trades_df['bars_held'].mean()
            else:
                avg_bars_held = 0
            
            kpis = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl_pct': total_pnl,
                'avg_pnl_pct': avg_pnl,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'profit_factor': profit_factor,
                'avg_r_multiple': avg_r,
                'avg_bars_held': avg_bars_held,
                'exit_breakdown': exit_breakdown
            }
            
            logger.info(f"Computed KPIs for {total_trades} trades")
            return kpis
            
        except Exception as e:
            logger.error(f"Error computing KPIs: {e}")
            return self._empty_kpis()
    
    def compute_counterfactual_uplift(
        self,
        actual_trades: pl.DataFrame,
        best_scenarios: List[Dict]
    ) -> Dict:
        """Compute uplift metrics from counterfactual analysis
        
        Args:
            actual_trades: DataFrame with actual trade results
            best_scenarios: List of best counterfactual scenarios per trade
            
        Returns:
            Dict with uplift metrics
        """
        if len(actual_trades) == 0 or len(best_scenarios) == 0:
            return {'avoidable_loss': 0, 'expected_uplift': 0, 'avg_delta_r': 0}
        
        try:
            # Compute delta PnL and delta R
            deltas = []
            for i, scenario in enumerate(best_scenarios):
                if i < len(actual_trades):
                    actual_pnl = actual_trades[i, 'pnl_pct']
                    best_pnl = scenario.get('pnl_pct', actual_pnl)
                    delta_pnl = best_pnl - actual_pnl
                    delta_r = delta_pnl / 0.01  # Assuming 1% risk
                    deltas.append({'delta_pnl': delta_pnl, 'delta_r': delta_r})
            
            deltas_df = pl.DataFrame(deltas)
            
            avoidable_loss = deltas_df.filter(pl.col('delta_pnl') > 0)['delta_pnl'].sum()
            expected_uplift = deltas_df['delta_pnl'].sum()
            avg_delta_r = deltas_df['delta_r'].mean()
            
            uplift = {
                'avoidable_loss': avoidable_loss,
                'expected_uplift': expected_uplift,
                'avg_delta_r': avg_delta_r,
                'improvable_trades': len(deltas_df.filter(pl.col('delta_pnl') > 0))
            }
            
            logger.info(f"Counterfactual uplift: {uplift['expected_uplift']:.2%} expected improvement")
            return uplift
            
        except Exception as e:
            logger.error(f"Error computing uplift: {e}")
            return {'avoidable_loss': 0, 'expected_uplift': 0, 'avg_delta_r': 0}
    
    def compute_regime_performance(
        self,
        trades_df: pl.DataFrame,
        regime_col: str = 'regime'
    ) -> Dict[str, Dict]:
        """Compute KPIs per regime
        
        Returns:
            Dict with regime -> KPIs mapping
        """
        if regime_col not in trades_df.columns:
            return {}
        
        regime_kpis = {}
        for regime in ['trend', 'chop', 'breakout', 'unknown']:
            regime_trades = trades_df.filter(pl.col(regime_col) == regime)
            if len(regime_trades) > 0:
                regime_kpis[regime] = self.compute_trade_kpis(regime_trades)
        
        return regime_kpis
    
    def _empty_kpis(self) -> Dict:
        """Return empty KPI dict"""
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'avg_pnl_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'profit_factor': 0,
            'avg_r_multiple': 0,
            'avg_bars_held': 0,
            'exit_breakdown': {}
        }
