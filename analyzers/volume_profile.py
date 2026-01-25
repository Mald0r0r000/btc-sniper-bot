"""
Analyseur Volume Profile
- POC (Point of Control)
- VAH / VAL (Value Area High / Low)
- Shape Detection (P-Shape, b-Shape, D-Shape)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timezone

import config


class VolumeProfileAnalyzer:
    """Analyse le profil de volume pour identifier les zones de valeur"""
    
    def __init__(self, df: pd.DataFrame, session_hours: int = 24):
        """
        Args:
            df: DataFrame OHLCV avec colonnes timestamp, open, high, low, close, volume
            session_hours: Heures à considérer pour la session (24 = journée complète)
        """
        self.df = df.copy()
        self.session_hours = session_hours
        self.bin_size = config.BIN_SIZE
        self.value_area_pct = config.VALUE_AREA_PCT
    
    def analyze(self, current_price: float = None) -> Dict[str, Any]:
        """
        Simplified Volume Profile Analysis
        Focus on Levels (POC/VAH/VAL) and Gaps (LVNs)
        """
        if self.df.empty or len(self.df) < 10:
            return self._empty_result()
        
        df_session = self._get_session_data()
        if len(df_session) < 10:
            df_session = self.df.tail(288)
        
        price_min = df_session['low'].min()
        price_max = df_session['high'].max()
        if price_max <= price_min:
            return self._empty_result()
        
        # Create bins
        bins = np.arange(start=price_min, stop=price_max + self.bin_size, step=self.bin_size)
        profile = pd.Series(0.0, index=bins[:-1], dtype='float64')
        
        # Distribute volume
        for _, row in df_session.iterrows():
            price_bin = int((row['close'] - price_min) / self.bin_size) * self.bin_size + price_min
            if price_bin in profile.index:
                profile[price_bin] += row['volume']
        
        total_vol = profile.sum()
        if total_vol == 0:
            return self._empty_result()
        
        # Core Levels (POC/VAH/VAL)
        poc = profile.idxmax()
        sorted_profile = profile.sort_values(ascending=False)
        accumulated_vol = 0
        value_area_prices = []
        for price, vol in sorted_profile.items():
            accumulated_vol += vol
            value_area_prices.append(price)
            if accumulated_vol >= total_vol * self.value_area_pct:
                break
        vah = max(value_area_prices)
        val = min(value_area_prices)
        
        # R&D: LVN (Low Volume Node) Detection - The "Gaps"
        # We look for bins inside the Value Area with significantly less volume than average
        avg_va_vol = accumulated_vol / len(value_area_prices)
        lvns = []
        # Check every 3 bins to find significant gaps
        va_range_profile = profile.loc[val:vah]
        for price, vol in va_range_profile.items():
            if vol < avg_va_vol * 0.3: # Less than 30% of average VA volume
                lvns.append(round(float(price), 2))
        
        # Contextual logic if current_price is provided
        context = "NEUTRAL"
        if current_price:
            context = self._determine_context(current_price, poc, vah, val, lvns)
            
        return {
            'poc': round(poc, 2),
            'vah': round(vah, 2),
            'val': round(val, 2),
            'lvns': lvns, # List of gap levels
            'context': context,
            'total_volume': round(total_vol, 2),
            'price_range': {'min': round(price_min, 2), 'max': round(price_max, 2)}
        }

    def _determine_context(self, price: float, poc: float, vah: float, val: float, lvns: List[float]) -> str:
        """Determines the market context relative to the profile structure"""
        buffer = (vah - val) * 0.05 if vah > val else 0
        
        # 1. Extreme Levels (Rejections)
        if price > vah + buffer: return "BREAKOUT_HIGH"
        if price < val - buffer: return "BREAKDOWN_LOW"
        
        # 2. Near Gaps (Fast Travel Zones)
        for lvn in lvns:
            if abs(price - lvn) / lvn < 0.001: # Within 0.1%
                return "TRAVERSING_GAP"
        
        # 3. Value Area Rotation
        if price > poc + buffer: return "VA_ROTATION_UP"
        if price < poc - buffer: return "VA_ROTATION_DOWN"
        
        # 4. Stuck at POC
        return "POC_STUCK"

    def _get_session_data(self) -> pd.DataFrame:
        if 'timestamp' not in self.df.columns:
            return self.df.tail(288)
        now = datetime.now(timezone.utc)
        start_of_session = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.df[self.df['timestamp'] >= start_of_session]

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'poc': 0, 'vah': 0, 'val': 0, 'lvns': [],
            'context': 'UNKNOWN', 'total_volume': 0,
            'price_range': {'min': 0, 'max': 0}
        }
