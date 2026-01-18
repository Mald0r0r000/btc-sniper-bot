"""
Module de connexion à l'exchange Bitget
"""
import ccxt
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

import config


class BitgetConnector:
    """Connecteur pour l'exchange Bitget"""
    
    def __init__(self):
        self.exchange = ccxt.bitget({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'password': config.API_PASSWORD,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Futures perpetuels
            }
        })
        self.symbol = config.SYMBOL
    
    def fetch_ohlcv(self, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """
        Récupère les données OHLCV
        
        Args:
            timeframe: Intervalle (1m, 5m, 1h, 1d, etc.)
            limit: Nombre de bougies
            
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception as e:
            print(f"❌ Erreur fetch_ohlcv ({timeframe}): {e}")
            return pd.DataFrame()
    
    def fetch_history_candles(self, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """
        Récupère les données OHLCV historiques via l'API history-candles
        Supporte jusqu'à 200 bougies, mais Bitget limite à 90 jours max par requête
        Pour obtenir plus de données, on fait plusieurs requêtes avec pagination
        
        Args:
            timeframe: Intervalle (1m, 5m, 1h, 1d, 3d, etc.)
            limit: Nombre de bougies souhaité (max 200)
            
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        try:
            from datetime import datetime, timedelta
            import time
            import requests
            
            # Map timeframe to Bitget granularity format
            # Note: Bitget limits historical data to 90 days per request
            # For 3D candles: 30 candles × 3 days = 90 days
            granularity_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '4h': '4H', '6h': '6H', '12h': '12H',
                '1d': '1D', '3d': '3D', '1w': '1W'
            }
            
            granularity = granularity_map.get(timeframe, timeframe)
            
            # Clean symbol format: BTC/USDT:USDT -> BTCUSDT
            clean_symbol = self.symbol.replace('/', '').split(':')[0]
            
            # Calculate how many requests needed based on 90-day limit
            # For 3D: each batch = 30 candles, so 3 batches = 90 candles
            candles_per_batch = 30 if timeframe == '3d' else 200
            num_batches = min((limit + candles_per_batch - 1) // candles_per_batch, 10)  # Max 10 batches
            
            all_candles = []
            current_end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            print(f"   🔍 Fetching {timeframe} candles from Bitget API ({num_batches} batches)...")
            
            for batch_num in range(num_batches):
                url = "https://api.bitget.com/api/v2/mix/market/history-candles"
                params = {
                    'symbol': clean_symbol,
                    'granularity': granularity,
                    'limit': str(200),
                    'productType': 'usdt-futures',
                    'endTime': str(current_end_time)
                }
                
                print(f"      Batch {batch_num + 1}/{num_batches}: endTime={current_end_time}")
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    print(f"      ⚠️ API error ({response.status_code}): {response.text[:100]}")
                    break
                
                data = response.json()
                
                if not data or 'data' not in data or not data['data']:
                    print(f"      ⚠️ No more data available")
                    break
                
                batch_candles = data['data']
                print(f"      ✅ Received {len(batch_candles)} candles")
                
                all_candles.extend(batch_candles)
                
                # Update end_time to the timestamp of the oldest candle in this batch
                # Subtract 1ms to avoid duplicate
                oldest_timestamp = int(batch_candles[-1][0])
                current_end_time = oldest_timestamp - 1
                
                # Small delay to respect rate limits
                if batch_num < num_batches - 1:
                    time.sleep(0.1)
                
                # Stop if we got fewer candles than expected (reached the end)
                if len(batch_candles) < candles_per_batch:
                    break
            
            if not all_candles:
                print(f"      ❌ No candles fetched")
                return pd.DataFrame()
            
            # Parse and combine all candles
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'usdtVolume'])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Drop extra column, remove duplicates, and sort chronologically
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates('timestamp').sort_values('timestamp')
            
            print(f"      📊 Total: {len(df)} unique candles fetched")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur fetch_history_candles ({timeframe}): {e}")
            return pd.DataFrame()
    
    def fetch_order_book(self, limit: int = 50) -> Dict[str, List]:
        """
        Récupère le carnet d'ordres
        
        Returns:
            Dict avec 'bids' et 'asks' (liste de [prix, volume])
        """
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=limit)
            return {
                'bids': ob['bids'],
                'asks': ob['asks']
            }
        except Exception as e:
            print(f"❌ Erreur fetch_order_book: {e}")
            return {'bids': [], 'asks': []}
    
    def fetch_trades(self, limit: int = 1000) -> List[Dict]:
        """
        Récupère les derniers trades
        
        Returns:
            Liste de trades avec 'side', 'amount', 'price', 'timestamp'
        """
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            return trades
        except Exception as e:
            print(f"❌ Erreur fetch_trades: {e}")
            return []
    
    def fetch_open_interest(self) -> Dict[str, Any]:
        """
        Récupère l'Open Interest
        
        Returns:
            Dict avec 'openInterestAmount' et 'openInterestValue'
        """
        try:
            oi = self.exchange.fetch_open_interest(self.symbol)
            return {
                'amount': float(oi.get('openInterestAmount', 0)),
                'value': float(oi.get('openInterestValue', 0)) if oi.get('openInterestValue') else None
            }
        except Exception as e:
            print(f"❌ Erreur fetch_open_interest: {e}")
            return {'amount': 0, 'value': None}
    
    def fetch_funding_rate(self) -> Dict[str, float]:
        """
        Récupère le taux de financement
        
        Returns:
            Dict avec 'fundingRate' et 'predictedFundingRate'
        """
        try:
            funding = self.exchange.fetch_funding_rate(self.symbol)
            return {
                'current': float(funding.get('fundingRate', 0) or 0),
                'predicted': float(funding.get('predictedFundingRate', 0) or 0)
            }
        except Exception as e:
            print(f"❌ Erreur fetch_funding_rate: {e}")
            return {'current': 0, 'predicted': 0}
    
    def fetch_ticker(self) -> Dict[str, float]:
        """
        Récupère le ticker actuel
        
        Returns:
            Dict avec 'last', 'bid', 'ask', etc.
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return {
                'last': float(ticker.get('last', 0)),
                'bid': float(ticker.get('bid', 0)),
                'ask': float(ticker.get('ask', 0)),
                'change_24h': float(ticker.get('percentage', 0) or 0)
            }
        except Exception as e:
            print(f"❌ Erreur fetch_ticker: {e}")
            return {'last': 0, 'bid': 0, 'ask': 0, 'change_24h': 0}
    
    def get_current_price(self) -> float:
        """Retourne le prix actuel"""
        ticker = self.fetch_ticker()
        return ticker['last']
