# 🔧 GrowwAPI Integration Guide

## Quick Start

### Option 1: Integrate Your Working GrowwAPI Code

**File to Edit:** `/app/backend/core/groww_helper.py`

**Function:** `fetch_historical_data()`

**Steps:**

1. Open `/app/backend/core/groww_helper.py`
2. Find the `fetch_historical_data()` method
3. Replace the `NotImplementedError` section with your working code
4. Follow the template provided in the docstring

**Template Variables (Already Created for You):**

```python
# These are automatically passed to your function:
trading_symbol = "NSE-RELIANCE"  # Symbol with exchange prefix
start_time = "2025-01-01 09:15:00"  # Start time string
end_time = "2025-01-01 15:30:00"    # End time string
interval_in_minutes = 5              # Candle interval
```

**Your Code Should:**

1. Import your GrowwAPI module
2. Initialize client (if needed)
3. Call the working API method
4. Parse response
5. Return Polars DataFrame with columns: `[ts, symbol, o, h, l, c, v, vwap]`

### Option 2: Use File Upload (No API Required)

If you have data in CSV/Excel/Parquet:

1. Launch Streamlit app
2. Choose "Upload Historical Data File"
3. Upload your file
4. System will auto-detect format and normalize

**Supported formats:**
- CSV (any delimiter)
- Excel (.xlsx, .xls)
- Parquet

**Flexible column names:**
- Timestamp: `timestamp`, `ts`, `time`, `datetime`, `date`
- Open: `open`, `o`, `Open`
- High: `high`, `h`, `High`
- Low: `low`, `l`, `Low`
- Close: `close`, `c`, `Close`, `ltp`, `price`
- Volume: `volume`, `v`, `vol`
- Symbol: `symbol`, `ticker`, `stock`

---

## Detailed Integration Example

### If you have this working code:

```python
import growwapi as groww

groww_client = groww.Groww()

response = groww_client.get_historical_candles(
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    groww_symbol="NSE-WIPRO",
    start_time="2025-09-24 10:56:00",
    end_time="2025-09-24 12:00:00",
    candle_interval=groww.CANDLE_INTERVAL_MIN_30
)
```

### Paste it like this:

```python
# In /app/backend/core/groww_helper.py
# Inside fetch_historical_data() method:

import growwapi as groww

# Initialize client
groww_client = groww.Groww()

# Map interval to constant
interval_map = {
    1: groww.CANDLE_INTERVAL_MIN_1,
    5: groww.CANDLE_INTERVAL_MIN_5,
    30: groww.CANDLE_INTERVAL_MIN_30,
    # ... add others as needed
}
candle_interval = interval_map.get(interval_in_minutes, interval_in_minutes)

# Call API
response = groww_client.get_historical_candles(
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    groww_symbol=trading_symbol,  # Use the parameter
    start_time=start_time,         # Use the parameter
    end_time=end_time,             # Use the parameter
    candle_interval=candle_interval
)

# Parse response
from datetime import datetime, timezone, timedelta

candles = response.get('candles', [])
data = []

for candle in candles:
    ts_epoch = candle[0]
    dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
    ist_dt = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
    
    data.append({
        'ts': ist_dt,
        'symbol': trading_symbol,
        'o': float(candle[1]),
        'h': float(candle[2]),
        'l': float(candle[3]),
        'c': float(candle[4]),
        'v': int(candle[5]),
        'vwap': (float(candle[2]) + float(candle[3]) + float(candle[4])) / 3
    })

df = pl.DataFrame(data)
return df
```

---

## Testing Your Integration

```bash
# Test the helper directly
python -c "
import sys
sys.path.insert(0, 'backend')
from core.groww_helper import GrowwHelper

helper = GrowwHelper()
df = helper.fetch_historical_data(
    trading_symbol='NSE-RELIANCE',
    start_time='2025-01-01 09:15:00',
    end_time='2025-01-01 15:30:00',
    interval_in_minutes=5
)

print(f'✅ Fetched {len(df)} bars')
print(df.head())
"
```

---

## Troubleshooting

### Error: "No module named 'growwapi'"

```bash
pip install growwapi
```

### Error: "Authentication failed"

Make sure your Groww credentials are configured:
```bash
# Check GrowwAPI documentation for setup
```

### Error: "Candles list is empty"

- Check if market is open
- Verify date range
- Ensure symbol format is correct (NSE-SYMBOL)

### Data Format Issues

If your API returns different format:
1. Print the `response` object
2. Adjust the parsing logic accordingly
3. Ensure final DataFrame has columns: `[ts, symbol, o, h, l, c, v, vwap]`

---

## Need Help?

Check `/app/backend/core/groww_helper.py` for detailed template and comments.
