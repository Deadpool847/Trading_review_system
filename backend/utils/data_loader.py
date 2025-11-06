"""Data Loading Utilities"""
import polars as pl
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration YAML"""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config not found: {config_path}")
        return {}
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_costs_config(costs_path: str = "config/costs.yaml") -> Dict:
    """Load costs configuration"""
    path = Path(costs_path)
    if not path.exists():
        logger.error(f"Costs config not found: {costs_path}")
        return {}
    
    with open(path, 'r') as f:
        costs = yaml.safe_load(f)
    
    return costs

def save_trades_review(
    trades_df: pl.DataFrame,
    output_path: str
) -> bool:
    """Save trades review to CSV"""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        trades_df.write_csv(output_path)
        logger.info(f"Saved trades review: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving trades review: {e}")
        return False

def save_summary_report(
    summary_text: str,
    output_path: str
) -> bool:
    """Save summary report to markdown"""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(summary_text)
        logger.info(f"Saved summary report: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        return False
