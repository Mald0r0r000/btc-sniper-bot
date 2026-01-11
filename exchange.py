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
