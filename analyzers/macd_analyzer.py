
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from exchange import BitgetConnector

class MACDAnalyzer:
    """
    Analyzes MACD on the 3D timeframe using native Bitget candles.
    """

    def __init__(self):
        self.connector = BitgetConnector()
        self.timeframe = '3d'
        # MACD Parameters
        self.fast = 12
        self.slow = 26
        self.signal = 9

    def analyze(self) -> Dict[str, Any]:
        """
        Fetches 3D candles and calculates MACD.
        Returns:
            Dict containing MACD values, histogram, and trend interpretation.
        """
        try:
            # Fetch 3D candles (need enough for slow MA + signal)
            # 26 + 9 + buffer = ~50 candles
            df = self.connector.fetch_ohlcv(self.timeframe, limit=100)
            
            if df is None or df.empty or len(df) < 50:
                print(f"   ⚠️ Not enough 3D candles for MACD: {len(df) if df is not None else 0}")
                return {
                    'macd': 0, 'signal': 0, 'hist': 0,
                    'trend': 'NEUTRAL', 'strength': 0,
                    'available': False
                }

            # Calculate MACD using pandas-ta
            # Default columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd = df.ta.macd(close='close', fast=self.fast, slow=self.slow, signal=self.signal)
            
            if macd is None or macd.empty:
                return {'available': False}

            # Get latest values (last row)
            # Names are usually: MACD_12_26_9 (Line), MACDh_12_26_9 (Hist), MACDs_12_26_9 (Signal)
            latest = macd.iloc[-1]
            
            # Identify column names dynamically to be safe
            macd_col = [c for c in macd.columns if c.startswith('MACD_')][0]
            hist_col = [c for c in macd.columns if c.startswith('MACDh_')][0]
            sig_col  = [c for c in macd.columns if c.startswith('MACDs_')][0]
            
            macd_val = float(latest[macd_col])
            hist_val = float(latest[hist_col])
            sig_val  = float(latest[sig_col])
            
            # Determine Trend
            # Bullish: MACD > Signal (Hist > 0)
            # Bearish: MACD < Signal (Hist < 0)
            if hist_val > 0:
                trend = 'BULLISH'
            elif hist_val < 0:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
                
            # Determine Strength (Slope of Histogram or Magnitude)
            # Simple heuristic: absolute value of histogram
            strength = abs(hist_val)

            return {
                'macd': macd_val,
                'signal': sig_val,
                'hist': hist_val,
                'trend': trend,
                'strength': strength,
                'available': True
            }

        except Exception as e:
            print(f"   ❌ MACD Analysis Error: {e}")
            return {
                'macd': 0, 'signal': 0, 'hist': 0,
                'trend': 'ERROR', 'strength': 0,
                'available': False
            }
