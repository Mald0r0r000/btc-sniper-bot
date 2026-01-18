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
        Supporte jusqu'à 200 bougies pour toutes les timeframes incluant 3D
        
        Args:
            timeframe: Intervalle (1m, 5m, 1h, 1d, 3d, etc.)
            limit: Nombre de bougies (max 200)
            
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        try:
            # Map timeframe to Bitget granularity format
            granularity_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '4h': '4H', '6h': '6H', '12h': '12H',
                '1d': '1D', '3d': '3D', '1w': '1W'
            }
            
            granularity = granularity_map.get(timeframe, timeframe)
            
            # Direct API call to Bitget v2 endpoint
            import requests
            
            # Clean symbol format: BTC/USDT:USDT -> BTCUSDT
            clean_symbol = self.symbol.replace('/', '').split(':')[0]
            
            url = "https://api.bitget.com/api/v2/mix/market/history-candles"
            params = {
                'symbol': clean_symbol,
                'granularity': granularity,
                'limit': str(min(limit, 200)),
                'productType': 'usdt-futures'
            }
            
            print(f"   🔍 Fetching {timeframe} candles from Bitget API...")
            print(f"      URL: {url}")
            print(f"      Params: {params}")
            
            response = requests.get(url, params=params, timeout=10)
            
            print(f"      Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"⚠️ Bitget API error ({response.status_code}): {response.text[:200]}")
                return pd.DataFrame()
            
            data = response.json()
            
            print(f"      Response keys: {list(data.keys()) if data else 'None'}")
            if data and 'data' in data:
                print(f"      Candles received: {len(data['data'])}")
            
            if not data or 'data' not in data or not data['data']:
                print(f"⚠️ No history data returned for {timeframe}")
                print(f"   Full response: {data}")
                return pd.DataFrame()
            
            # Parse response - Bitget returns [timestamp, open, high, low, close, volume, usdtVolume]
            candles = data['data']
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'usdtVolume'])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Drop extra column and sort chronologically
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp')
            
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
