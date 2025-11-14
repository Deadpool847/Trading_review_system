"""Super-Powered Data Validator - Handles ANY DataFrame Format"""
import polars as pl
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """Intelligent data validator that normalizes any OHLCV format"""
    
    # Common column name variations
    TIMESTAMP_COLS = ['timestamp', 'ts', 'time', 'datetime', 'date', 'Date', 'Time', 'Timestamp']
    OPEN_COLS = ['open', 'o', 'Open', 'O', 'OPEN']
    HIGH_COLS = ['high', 'h', 'High', 'H', 'HIGH']
    LOW_COLS = ['low', 'l', 'Low', 'L', 'LOW']
    CLOSE_COLS = ['close', 'c', 'Close', 'C', 'CLOSE', 'ltp', 'price', 'last']
    VOLUME_COLS = ['volume', 'v', 'vol', 'Volume', 'V', 'Vol', 'VOLUME']
    SYMBOL_COLS = ['symbol', 'Symbol', 'ticker', 'Ticker', 'stock', 'Stock', 'SYMBOL']
    
    def __init__(self):
        self.detected_format = {}
    
    def detect_column(self, df_columns: List[str], possible_names: List[str]) -> Optional[str]:
        """Intelligently detect column name from variations"""
        for col in df_columns:
            if col in possible_names:
                return col
        
        # Fuzzy matching
        for col in df_columns:
            col_lower = col.lower().strip()
            for possible in possible_names:
                if possible.lower() in col_lower or col_lower in possible.lower():
                    return col
        
        return None
    
    def load_file(self, file_path: str, file_type: str = None) -> pl.DataFrame:
        """Load any file format (CSV, Excel, Parquet)
        
        Args:
            file_path: Path to file or uploaded file object
            file_type: Optional file type hint ('csv', 'excel', 'parquet')
            
        Returns:
            Polars DataFrame
        """
        try:
            # Auto-detect from extension
            if file_type is None:
                if hasattr(file_path, 'name'):
                    file_path_str = file_path.name
                else:
                    file_path_str = str(file_path)
                
                if file_path_str.endswith('.csv'):
                    file_type = 'csv'
                elif file_path_str.endswith(('.xls', '.xlsx')):
                    file_type = 'excel'
                elif file_path_str.endswith('.parquet'):
                    file_type = 'parquet'
                else:
                    file_type = 'csv'  # Default
            
            # Load based on type
            if file_type == 'csv':
                if hasattr(file_path, 'read'):
                    # File upload object
                    import io
                    df = pl.read_csv(io.BytesIO(file_path.read()))
                else:
                    df = pl.read_csv(file_path)
            
            elif file_type == 'excel':
                if hasattr(file_path, 'read'):
                    # Use pandas for Excel, convert to polars
                    import io
                    pdf = pd.read_excel(io.BytesIO(file_path.read()))
                    df = pl.from_pandas(pdf)
                else:
                    pdf = pd.read_excel(file_path)
                    df = pl.from_pandas(pdf)
            
            elif file_type == 'parquet':
                if hasattr(file_path, 'read'):
                    import io
                    df = pl.read_parquet(io.BytesIO(file_path.read()))
                else:
                    df = pl.read_parquet(file_path)
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"✅ Loaded file: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            raise
    
    def validate_and_normalize(self, df: pl.DataFrame, symbol: str = "UNKNOWN") -> pl.DataFrame:
        """Validate and normalize OHLCV data from any format
        
        This is the SUPER-POWERED function that handles ANY dataframe!
        
        Args:
            df: Input dataframe in any format
            symbol: Symbol name (used if not in data)
            
        Returns:
            Normalized dataframe with standard columns: [ts, symbol, o, h, l, c, v, vwap]
        """
        if len(df) == 0:
            raise ValueError("Empty dataframe")
        
        logger.info(f"🔍 Validating dataframe: {len(df)} rows")
        logger.info(f"📋 Input columns: {df.columns}")
        
        try:
            # Detect columns
            ts_col = self.detect_column(df.columns, self.TIMESTAMP_COLS)
            open_col = self.detect_column(df.columns, self.OPEN_COLS)
            high_col = self.detect_column(df.columns, self.HIGH_COLS)
            low_col = self.detect_column(df.columns, self.LOW_COLS)
            close_col = self.detect_column(df.columns, self.CLOSE_COLS)
            volume_col = self.detect_column(df.columns, self.VOLUME_COLS)
            symbol_col = self.detect_column(df.columns, self.SYMBOL_COLS)
            
            # Log detection
            self.detected_format = {
                'timestamp': ts_col,
                'open': open_col,
                'high': high_col,
                'low': low_col,
                'close': close_col,
                'volume': volume_col,
                'symbol': symbol_col
            }
            
            logger.info(f"🎯 Detected columns: {self.detected_format}")
            
            # Validate required columns
            if not close_col:
                raise ValueError("❌ No close/price column found. Required columns: close/c/ltp/price")
            
            # Build normalized dataframe
            normalized_data = {}
            
            # Timestamp
            if ts_col:
                ts_series = df[ts_col]
                # Try to parse if string
                if ts_series.dtype == pl.Utf8:
                    try:
                        normalized_data['ts'] = pl.col(ts_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                    except:
                        try:
                            normalized_data['ts'] = pl.col(ts_col).str.strptime(pl.Datetime, "%Y-%m-%d")
                        except:
                            # Try pandas parsing as fallback
                            pdf = df.to_pandas()
                            pdf[ts_col] = pd.to_datetime(pdf[ts_col], errors='coerce')
                            df = pl.from_pandas(pdf)
                            normalized_data['ts'] = pl.col(ts_col)
                else:
                    normalized_data['ts'] = pl.col(ts_col)
            else:
                # Generate timestamps if missing
                logger.warning("⚠️ No timestamp column found. Generating sequential timestamps.")
                start_ts = datetime.now(timezone.utc)
                timestamps = [start_ts + timedelta(minutes=i*5) for i in range(len(df))]
                df = df.with_columns([pl.lit(timestamps).alias('ts')])
                normalized_data['ts'] = pl.col('ts')
            
            # Symbol
            if symbol_col:
                normalized_data['symbol'] = pl.col(symbol_col)
            else:
                normalized_data['symbol'] = pl.lit(symbol)
            
            # OHLC
            if open_col:
                normalized_data['o'] = pl.col(open_col).cast(pl.Float64)
            else:
                # Use close as open if missing
                normalized_data['o'] = pl.col(close_col).cast(pl.Float64)
                logger.warning("⚠️ No open column. Using close as open.")
            
            if high_col:
                normalized_data['h'] = pl.col(high_col).cast(pl.Float64)
            else:
                # Use close as high if missing
                normalized_data['h'] = pl.col(close_col).cast(pl.Float64)
                logger.warning("⚠️ No high column. Using close as high.")
            
            if low_col:
                normalized_data['l'] = pl.col(low_col).cast(pl.Float64)
            else:
                # Use close as low if missing
                normalized_data['l'] = pl.col(close_col).cast(pl.Float64)
                logger.warning("⚠️ No low column. Using close as low.")
            
            normalized_data['c'] = pl.col(close_col).cast(pl.Float64)
            
            # Volume
            if volume_col:
                normalized_data['v'] = pl.col(volume_col).cast(pl.Int64)
            else:
                # Default volume if missing
                normalized_data['v'] = pl.lit(10000)
                logger.warning("⚠️ No volume column. Using default value 10000.")
            
            # VWAP (calculate if not present)
            normalized_data['vwap'] = (pl.col('h') + pl.col('l') + pl.col('c')) / 3.0
            
            # Apply transformations step by step to avoid column reference errors
            result_df = df
            
            for col_name, col_expr in normalized_data.items():
                if isinstance(col_expr, pl.Expr):
                    result_df = result_df.with_columns([col_expr.alias(col_name)])
                else:
                    result_df = result_df.with_columns([pl.lit(col_expr).alias(col_name)])
            
            # Select only the normalized columns
            normalized_df = result_df.select(['ts', 'symbol', 'o', 'h', 'l', 'c', 'v', 'vwap'])
            
            # Remove nulls
            normalized_df = normalized_df.drop_nulls()
            
            # Sort by timestamp
            normalized_df = normalized_df.sort('ts')
            
            logger.info(f"✅ Normalized to standard format: {len(normalized_df)} rows")
            logger.info(f"📊 Columns: {normalized_df.columns}")
            
            return normalized_df
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            logger.error(f"Available columns: {df.columns}")
            logger.error(f"Sample data:\n{df.head()}")
            raise ValueError(f"Could not normalize dataframe: {e}")
    
    def validate_ohlc_integrity(self, df: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLC data integrity
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check high >= low
            invalid_hl = df.filter(pl.col('h') < pl.col('l'))
            if len(invalid_hl) > 0:
                issues.append(f"⚠️ {len(invalid_hl)} bars have high < low")
            
            # Check high >= close, low <= close
            invalid_hc = df.filter(pl.col('h') < pl.col('c'))
            if len(invalid_hc) > 0:
                issues.append(f"⚠️ {len(invalid_hc)} bars have high < close")
            
            invalid_lc = df.filter(pl.col('l') > pl.col('c'))
            if len(invalid_lc) > 0:
                issues.append(f"⚠️ {len(invalid_lc)} bars have low > close")
            
            # Check for negative prices
            negative = df.filter(
                (pl.col('o') < 0) | (pl.col('h') < 0) | (pl.col('l') < 0) | (pl.col('c') < 0)
            )
            if len(negative) > 0:
                issues.append(f"❌ {len(negative)} bars have negative prices")
            
            # Check for zero volume
            zero_vol = df.filter(pl.col('v') == 0)
            if len(zero_vol) > 0:
                issues.append(f"⚠️ {len(zero_vol)} bars have zero volume")
            
            is_valid = len([i for i in issues if i.startswith('❌')]) == 0
            
            if is_valid:
                logger.info("✅ OHLC data integrity validated")
            else:
                logger.warning(f"⚠️ Data integrity issues: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating integrity: {e}")
            return False, [f"Validation error: {e}"]
