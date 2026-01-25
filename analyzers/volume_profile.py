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
        
        # R&D: Structural Nodes (AMT Approach)
        avg_va_vol = accumulated_vol / len(value_area_prices)
        
        # 1. HVNs (High Volume Nodes) - The Targets/Magnets
        hvns = []
        for price, vol in profile.items():
            if vol > avg_va_vol * 1.5: # 50% above average
                hvns.append(round(float(price), 2))
        
        # 2. LVN Zones (Gap grouping)
        lvn_bins = []
        for price, vol in profile.loc[val:vah].items():
            if vol < avg_va_vol * 0.3:
                lvn_bins.append(float(price))
                
        # Group adjacent bins into zones
        gap_zones = []
        if lvn_bins:
            lvn_bins.sort()
            current_zone = [lvn_bins[0]]
            for i in range(1, len(lvn_bins)):
                if lvn_bins[i] - lvn_bins[i-1] <= self.bin_size * 1.1:
                    current_zone.append(lvn_bins[i])
                else:
                    gap_zones.append({'min': min(current_zone), 'max': max(current_zone)})
                    current_zone = [lvn_bins[i]]
            gap_zones.append({'min': min(current_zone), 'max': max(current_zone)})

        # 3. Regime & Context
        regime = "BALANCE"
        context = "NEUTRAL"
        target_price = None
        
        if current_price:
            regime = "IMBALANCE" if (current_price > vah or current_price < val) else "BALANCE"
            context, target_price = self._determine_amt_context(current_price, poc, vah, val, hvns, gap_zones)
            
        return {
            'poc': round(poc, 2),
            'vah': round(vah, 2),
            'val': round(val, 2),
            'hvns': hvns[:5], # Top 5 targets
            'gap_zones': gap_zones,
            'regime': regime,
            'context': context,
            'target_price': target_price,
            'total_volume': round(total_vol, 2),
            'price_range': {'min': round(price_min, 2), 'max': round(price_max, 2)}
        }

    def _determine_amt_context(self, price: float, poc: float, vah: float, val: float, 
                              hvns: List[float], gap_zones: List[Dict]) -> tuple:
        """Determines context based on Auction Market Theory"""
        buffer = (vah - val) * 0.02 # 2% buffer
        
        # 1. Imbalance States (Breakouts)
        if price > vah + buffer:
            # Finding next HVN target above
            target = min([h for h in hvns if h > price], default=price * 1.01)
            return "IMBALANCE_EXPANSION_UP", round(target, 2)
        if price < val - buffer:
            target = max([h for h in hvns if h < price], default=price * 0.99)
            return "IMBALANCE_EXPANSION_DOWN", round(target, 2)
            
        # 2. Fast Travel (Gaps)
        for zone in gap_zones:
            if zone['min'] <= price <= zone['max']:
                # If in a gap, the target is the next HVN in direction of momentum
                target = poc # Default to POC as major magnet
                return "TRAVERSING_LIQUID_GAP", round(target, 2)
                
        # 3. Balance Rotations
        if price > poc:
            return "VALUE_AREA_ROTATION_UP", round(vah, 2)
        if price < poc:
            return "VALUE_AREA_ROTATION_DOWN", round(val, 2)
            
        return "STUCK_AT_POC", round(poc, 2)

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
