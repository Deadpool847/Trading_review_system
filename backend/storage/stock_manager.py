"""Stock Data Manager - Local storage per stock"""
import polars as pl
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class StockManager:
    """Manage stock data and models locally"""
    
    def __init__(self, base_dir: str = "data/stocks"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_stock_data(self, symbol: str, df: pl.DataFrame):
        """Save stock data"""
        stock_dir = self.base_dir / symbol
        stock_dir.mkdir(exist_ok=True)
        
        data_path = stock_dir / "historical_data.parquet"
        df.write_parquet(data_path)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'rows': len(df),
            'start_date': str(df['timestamp'].min()),
            'end_date': str(df['timestamp'].max()),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(stock_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved {len(df)} bars for {symbol}")
    
    def load_stock_data(self, symbol: str) -> Optional[pl.DataFrame]:
        """Load stock data"""
        data_path = self.base_dir / symbol / "historical_data.parquet"
        
        if not data_path.exists():
            return None
        
        return pl.read_parquet(data_path)
    
    def get_all_stocks(self) -> List[str]:
        """Get list of all stored stocks"""
        stocks = []
        
        for stock_dir in self.base_dir.iterdir():
            if stock_dir.is_dir() and (stock_dir / "metadata.json").exists():
                stocks.append(stock_dir.name)
        
        return sorted(stocks)
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get stock metadata"""
        meta_path = self.base_dir / symbol / "metadata.json"
        
        if not meta_path.exists():
            return None
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def has_model(self, symbol: str) -> bool:
        """Check if model exists for stock"""
        model_path = Path("models") / symbol / "metadata.json"
        return model_path.exists()
