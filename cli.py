#!/usr/bin/env python3
"""CLI Automation for Daily Review Machine"""
import click
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from core.data_ingestion import DataIngestion
from core.feature_engineering import FeatureEngineer
from core.labeling import TripleBarrierLabeler
from core.regime_detection import RegimeDetector
from core.ml_trainer import MLTrainer
from core.counterfactual import CounterfactualEngine
from core.kpi_computer import KPIComputer
from utils.data_loader import load_config, load_costs_config, save_trades_review, save_summary_report
from utils.helpers import format_date_for_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Daily Review Machine CLI"""
    pass

@cli.command()
@click.option('--date', default=None, help='Review date (YYYY-MM-DD)')
@click.option('--scope', type=click.Choice(['daily', 'weekly', 'monthly']), default='daily', help='Review scope')
@click.option('--symbol', default='RELIANCE', help='Trading symbol')
@click.option('--exchange', default='NSE', help='Exchange')
@click.option('--segment', default='CASH', help='Segment')
def run(date, scope, symbol, exchange, segment):
    """Run complete review pipeline"""
    try:
        # Parse date
        if date:
            review_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            review_date = datetime.now()
        
        click.echo(f"\n🚀 Starting {scope} review for {symbol} on {review_date.strftime('%Y-%m-%d')}...\n")
        
        # Load configs
        config = load_config()
        costs_config = load_costs_config()
        
        # Initialize components
        ingestion = DataIngestion()
        engineer = FeatureEngineer(config)
        labeler = TripleBarrierLabeler(config, costs_config)
        regime_detector = RegimeDetector(config)
        trainer = MLTrainer(config)
        cf_engine = CounterfactualEngine(config, costs_config)
        kpi_computer = KPIComputer(config)
        
        # Determine time range based on scope
        if scope == 'daily':
            interval = 5
            start_time = review_date.replace(hour=9, minute=15, second=0)
            end_time = review_date.replace(hour=15, minute=30, second=0)
        elif scope == 'weekly':
            interval = 5
            start_time = review_date - timedelta(days=7)
            end_time = review_date
        else:  # monthly
            interval = 15
            start_time = review_date.replace(day=1)
            end_time = review_date
        
        click.echo("📥 Step 1: Fetching market data...")
        
        # For CLI without actual Groww connection, create dummy data
        # In production, replace with actual GrowwAPI call
        click.echo("⚠️  Warning: Using mock data (no Groww connection in CLI mode)")
        click.echo("    To use real data, run via Streamlit app with Groww authentication.\n")
        
        # Create synthetic bars for demo
        from datetime import timezone as tz
        import numpy as np
        
        n_bars = 78 if scope == 'daily' else 300
        timestamps = [start_time + timedelta(minutes=i*interval) for i in range(n_bars)]
        base_price = 2500.0
        
        bars_data = []
        for i, ts in enumerate(timestamps):
            price_change = np.random.randn() * 10
            close = base_price + price_change
            high = close + abs(np.random.randn() * 5)
            low = close - abs(np.random.randn() * 5)
            open_price = bars_data[-1]['c'] if bars_data else close
            volume = int(10000 + np.random.randn() * 2000)
            
            bars_data.append({
                'ts': ts.replace(tzinfo=tz.utc),
                'symbol': symbol,
                'o': open_price,
                'h': high,
                'l': low,
                'c': close,
                'v': volume,
                'vwap': (high + low + close) / 3
            })
            base_price = close
        
        bars_df = pl.DataFrame(bars_data)
        
        click.echo(f"✅ Fetched {len(bars_df)} bars\n")
        
        click.echo("🔧 Step 2: Computing features...")
        bars_df = engineer.compute_features(bars_df, base_score=0.5)
        click.echo(f"✅ Computed {len(engineer.get_feature_columns())} features\n")
        
        click.echo("🎯 Step 3: Detecting regimes...")
        bars_df = regime_detector.detect_regime(bars_df)
        regime_stats = regime_detector.get_regime_stats(bars_df)
        
        for regime, stats in regime_stats.items():
            click.echo(f"   {regime.capitalize()}: {stats['pct']:.1f}%")
        click.echo()
        
        click.echo("🏷️  Step 4: Creating meta-labels...")
        bars_df = labeler.create_meta_labels(bars_df)
        click.echo("✅ Labels created\n")
        
        click.echo("🤖 Step 5: ML inference (skipping training for CLI demo)...")
        click.echo("   Use Streamlit app for full ML training\n")
        
        click.echo("🔬 Step 6: Running counterfactuals (sample)...")
        sample_idx = len(bars_df) // 2
        cf_results = cf_engine.run_grid_search(bars_df, sample_idx)
        best_policy = cf_engine.get_best_policy(cf_results)
        click.echo(f"✅ Best policy PnL: {best_policy['pnl_pct']:.2%}\n")
        
        click.echo("📊 Step 7: Computing KPIs...")
        # For demo, extract some sample trades
        sample_trades = bars_df.sample(min(10, len(bars_df) // 10)).with_columns([
            pl.col('meta_label').alias('label'),
            pl.col('realized_return').alias('pnl_pct'),
            pl.col('exit_reason').alias('exit_reason'),
            pl.lit(8).alias('bars_held')
        ])
        
        kpis = kpi_computer.compute_trade_kpis(sample_trades)
        
        click.echo(f"   Total Trades: {kpis['total_trades']}")
        click.echo(f"   Win Rate: {kpis['win_rate']:.1%}")
        click.echo(f"   Avg R: {kpis['avg_r_multiple']:.2f}")
        click.echo()
        
        click.echo("💾 Step 8: Generating reports...")
        report_dir = Path(f"reports/{review_date.strftime('%Y-%m-%d')}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = f"""# Daily Review Machine - {scope.capitalize()} Review

**Date:** {review_date.strftime('%Y-%m-%d')}  
**Symbol:** {symbol}  
**Scope:** {scope}  

## Key Metrics

- **Total Bars Analyzed:** {len(bars_df)}  
- **Win Rate:** {kpis['win_rate']:.1%}  
- **Average R-Multiple:** {kpis['avg_r_multiple']:.2f}  
- **Profit Factor:** {kpis['profit_factor']:.2f}  

## Regime Distribution

{chr(10).join([f'- **{r.capitalize()}:** {s["pct"]:.1f}%' for r, s in regime_stats.items()])}

## Top Recommendations

1. **Best Counterfactual Policy:** {best_policy['exit_mode']} with entry_shift={best_policy['entry_shift']}
2. **Optimal TP/SL:** TP={best_policy['tp_mult']}x, SL={best_policy['sl_mult']}x
3. **Expected Improvement:** {best_policy['pnl_pct']:.2%}

## Next Steps

- Review high-probability setups in What-If Lab
- Retrain models with latest data
- Adjust regime thresholds based on recent market behavior

---
*Generated by Daily Review Machine*
"""
        
        save_summary_report(summary, str(report_dir / 'summary.md'))
        save_trades_review(sample_trades, str(report_dir / 'trades_review.csv'))
        
        click.echo(f"✅ Reports saved to {report_dir}\n")
        
        click.echo("✨ Review complete! \n")
        click.echo(f"📄 Summary: {report_dir / 'summary.md'}")
        click.echo(f"📊 Trades: {report_dir / 'trades_review.csv'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during review: {e}", exc_info=True)
        click.echo(f"\n❌ Error: {e}\n", err=True)
        return 1

if __name__ == '__main__':
    cli()
