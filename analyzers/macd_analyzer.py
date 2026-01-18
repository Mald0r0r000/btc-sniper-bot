
import pandas as pd
from typing import Dict, Any
from exchange import BitgetConnector

class MACDAnalyzer:
    """
    Analyzes MACD on the 3D timeframe using native Bitget candles.
    """

    def __init__(self):
        self.connector = BitgetConnector()
        self.timeframe = '3d'
        # MACD Parameters (12, 26, 9)
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

            # Calculate MACD manually using pandas EMA
            # MACD = 12-EMA - 26-EMA
            # Signal = 9-EMA of MACD
            # Histogram = MACD - Signal
            
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
            
            # MACD Line
            macd_line = ema_fast - ema_slow
            
            # Signal Line (9-period EMA of MACD)
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            # Get latest values
            macd_val = float(macd_line.iloc[-1])
            sig_val = float(signal_line.iloc[-1])
            hist_val = float(histogram.iloc[-1])
            
            # Determine Trend
            # Bullish: MACD > Signal (Hist > 0)
            # Bearish: MACD < Signal (Hist < 0)
            if hist_val > 0:
                trend = 'BULLISH'
            elif hist_val < 0:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
                
            # Determine Strength (absolute value of histogram)
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
