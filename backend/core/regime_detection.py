"""Regime Detection Module"""
import polars as pl
import numpy as np
from typing import Dict, Literal
import logging

logger = logging.getLogger(__name__)

RegimeType = Literal['trend', 'chop', 'breakout', 'unknown']

class RegimeDetector:
    """Detect market regime: Trend, Chop, Breakout"""
    
    def __init__(self, config: Dict):
        self.config = config.get('regime', {})
        
        # Trend thresholds
        self.trend_vwap_slope = self.config.get('trend', {}).get('min_vwap_slope', 0.0015)
        self.trend_directional_bars = self.config.get('trend', {}).get('min_directional_bars', 5)
        
        # Chop thresholds
        self.chop_max_vol = self.config.get('chop', {}).get('max_realized_vol', 0.025)
        self.chop_max_atr = self.config.get('chop', {}).get('max_atr_pct', 0.02)
        
        # Breakout thresholds
        self.breakout_vol_ratio = self.config.get('breakout', {}).get('min_volume_ratio', 2.0)
        self.breakout_atr_expansion = self.config.get('breakout', {}).get('min_atr_expansion', 1.5)
    
    def detect_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Classify each bar into a regime
        
        Args:
            df: DataFrame with features (vwap_slope, realized_vol, atr_pct, volume_ratio_*)
            
        Returns:
            DataFrame with 'regime' column
        """
        if len(df) < 20:
            logger.warning("Insufficient data for regime detection")
            return df.with_columns([pl.lit('unknown').alias('regime')])
        
        try:
            # Compute directional persistence (how many consecutive up/down bars)
            df = df.with_columns([
                (pl.col('c') > pl.col('c').shift(1)).cast(pl.Int8).alias('up_bar')
            ])
            
            # Rolling count of directional bars
            df = df.with_columns([
                pl.col('up_bar').rolling_sum(self.trend_directional_bars).alias('directional_count')
            ])
            
            # ATR expansion (current ATR vs rolling average)
            df = df.with_columns([
                (pl.col('atr_pct') / pl.col('atr_pct').rolling_mean(10)).alias('atr_expansion')
            ])
            
            # Get volume ratio (use the 5-bar one if available)
            vol_col = 'volume_ratio_5' if 'volume_ratio_5' in df.columns else 'v'
            
            # Regime classification logic
            df = df.with_columns([
                pl.when(
                    # TREND: Strong directional movement
                    (pl.col('vwap_slope').abs() > self.trend_vwap_slope) &
                    ((pl.col('directional_count') >= self.trend_directional_bars) |
                     (pl.col('directional_count') <= (self.trend_directional_bars - self.trend_directional_bars)))
                ).then(pl.lit('trend'))
                .when(
                    # CHOP: Low volatility, range-bound
                    (pl.col('realized_vol') < self.chop_max_vol) &
                    (pl.col('atr_pct') < self.chop_max_atr)
                ).then(pl.lit('chop'))
                .when(
                    # BREAKOUT: High volume + ATR expansion
                    (pl.col(vol_col) > self.breakout_vol_ratio) &
                    (pl.col('atr_expansion') > self.breakout_atr_expansion)
                ).then(pl.lit('breakout'))
                .otherwise(pl.lit('unknown'))
                .alias('regime')
            ])
            
            logger.info(f"Regime detection complete for {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return df.with_columns([pl.lit('unknown').alias('regime')])
    
    def get_regime_stats(self, df: pl.DataFrame) -> Dict[str, Dict]:
        """Compute statistics per regime
        
        Returns:
            Dict with regime-wise stats
        """
        if 'regime' not in df.columns:
            return {}
        
        stats = {}
        for regime in ['trend', 'chop', 'breakout', 'unknown']:
            regime_df = df.filter(pl.col('regime') == regime)
            if len(regime_df) == 0:
                continue
            
            stats[regime] = {
                'count': len(regime_df),
                'pct': len(regime_df) / len(df) * 100,
                'avg_vol': regime_df['realized_vol'].mean() if 'realized_vol' in regime_df.columns else 0,
                'avg_atr': regime_df['atr_pct'].mean() if 'atr_pct' in regime_df.columns else 0
            }
        
        return stats
