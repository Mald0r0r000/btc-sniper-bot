"""
Data Provider for Backtesting
Fetches and caches historical OHLCV data from exchanges
"""

import ccxt
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import time

class DataProvider:
    """
    Provides historical market data for backtesting.
    Features:
    - Multi-timeframe support (5m, 1h, 4h, 1d)
    - Local caching for performance
    - Multiple exchange fallback
    """
    
    def __init__(self, cache_dir: str = "backtest/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Exchange priority for data fetching
        self.exchanges = self._init_exchanges()
        
    def _init_exchanges(self) -> List[ccxt.Exchange]:
        """Initialize exchange connections with fallback priority"""
        exchanges = []
        
        # OKX - Primary (reliable, no geo-block issues)
        try:
            okx = ccxt.okx({'enableRateLimit': True})
            exchanges.append(okx)
        except:
            pass
            
        # Bybit - Secondary
        try:
            bybit = ccxt.bybit({'enableRateLimit': True})
            exchanges.append(bybit)
        except:
            pass
            
        # Bitget - Tertiary
        try:
            bitget = ccxt.bitget({'enableRateLimit': True})
            exchanges.append(bitget)
        except:
            pass
            
        return exchanges
    
    def _get_cache_path(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generate cache file path"""
        symbol_safe = symbol.replace("/", "_").replace(":", "_")
        return os.path.join(
            self.cache_dir,
            f"{symbol_safe}_{timeframe}_{start_date}_{end_date}.json"
        )
    
    def _load_from_cache(self, cache_path: str) -> Optional[List]:
        """Load data from cache if exists and valid"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    print(f"   📂 Loaded from cache: {os.path.basename(cache_path)}")
                    return data
            except:
                pass
        return None
    
    def _save_to_cache(self, cache_path: str, data: List):
        """Save data to cache"""
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        print(f"   💾 Saved to cache: {os.path.basename(cache_path)}")
    
    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Fetch OHLCV data for a given symbol and timeframe.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT:USDT")
            timeframe: Candle timeframe (5m, 15m, 1h, 4h, 1d)
            start_date: Start date (YYYY-MM-DD), defaults to 6 months ago
            end_date: End date (YYYY-MM-DD), defaults to now
            use_cache: Whether to use local cache
            
        Returns:
            List of OHLCV dicts with keys: timestamp, open, high, low, close, volume
        """
        # Default dates
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
        
        print(f"\n📊 Fetching {symbol} {timeframe} data: {start_date} → {end_date}")
        
        # Check cache
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                return cached
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        # Fetch from exchanges with fallback
        all_candles = []
        for exchange in self.exchanges:
            try:
                print(f"   🔄 Trying {exchange.id}...")
                all_candles = self._fetch_all_candles(exchange, symbol, timeframe, start_ts, end_ts)
                if len(all_candles) > 0:
                    print(f"   ✅ Fetched {len(all_candles)} candles from {exchange.id}")
                    break
            except Exception as e:
                print(f"   ❌ {exchange.id} failed: {str(e)[:50]}")
                continue
        
        if len(all_candles) == 0:
            print("   ⚠️ No data fetched from any exchange")
            return []
        
        # Convert to dict format
        ohlcv_data = []
        for candle in all_candles:
            ohlcv_data.append({
                "timestamp": candle[0],
                "datetime": datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc).isoformat(),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            })
        
        # Save to cache
        if use_cache and len(ohlcv_data) > 0:
            self._save_to_cache(cache_path, ohlcv_data)
        
        return ohlcv_data
    
    def _fetch_all_candles(
        self, 
        exchange: ccxt.Exchange, 
        symbol: str, 
        timeframe: str,
        start_ts: int,
        end_ts: int
    ) -> List:
        """Fetch all candles with pagination"""
        all_candles = []
        current_ts = start_ts
        
        # Timeframe to milliseconds
        tf_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "1h": 3600000,
            "4h": 14400000,
            "1d": 86400000
        }
        
        limit = 1000  # Most exchanges support 1000 candles per request
        step = tf_ms.get(timeframe, 3600000) * limit
        
        while current_ts < end_ts:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=limit
                )
                
                if len(candles) == 0:
                    break
                    
                all_candles.extend(candles)
                current_ts = candles[-1][0] + tf_ms.get(timeframe, 3600000)
                
                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"      Error at {current_ts}: {e}")
                break
        
        # Remove duplicates and sort
        seen = set()
        unique_candles = []
        for c in all_candles:
            if c[0] not in seen and c[0] <= end_ts:
                seen.add(c[0])
                unique_candles.append(c)
        
        return sorted(unique_candles, key=lambda x: x[0])
    
    def fetch_multi_timeframe(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframes: List[str] = ["5m", "1h", "4h", "1d"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Fetch data for multiple timeframes.
        
        Returns:
            Dict with timeframe as key and OHLCV list as value
        """
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, start_date, end_date)
        return result
    
    def get_price_at_timestamp(self, ohlcv: List[Dict], timestamp: int) -> Optional[Dict]:
        """Get the candle containing a specific timestamp"""
        for i, candle in enumerate(ohlcv):
            if candle["timestamp"] >= timestamp:
                return candle
        return ohlcv[-1] if ohlcv else None
    
    def resample(self, ohlcv_5m: List[Dict], target_tf: str) -> List[Dict]:
        """Resample 5m data to higher timeframe (for consistency)"""
        # Implementation for resampling if needed
        pass


# Test function
def test_data_provider():
    print("=" * 60)
    print("Testing Data Provider")
    print("=" * 60)
    
    provider = DataProvider()
    
    # Test 1h data fetch for 1 week
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    
    data = provider.fetch_ohlcv(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date
    )
    
    if data:
        print(f"\n📈 Data summary:")
        print(f"   Candles: {len(data)}")
        print(f"   First: {data[0]['datetime']} @ ${data[0]['close']:,.2f}")
        print(f"   Last:  {data[-1]['datetime']} @ ${data[-1]['close']:,.2f}")
        
        # Price range
        highs = [c['high'] for c in data]
        lows = [c['low'] for c in data]
        print(f"   Range: ${min(lows):,.2f} - ${max(highs):,.2f}")


if __name__ == "__main__":
    test_data_provider()
