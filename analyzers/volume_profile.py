"""
Analyseur Volume Profile
- POC (Point of Control)
- VAH / VAL (Value Area High / Low)
- Shape Detection (P-Shape, b-Shape, D-Shape)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
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
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète du profil de volume
        
        Returns:
            Dict avec POC, VAH, VAL, shape, et métriques
        """
        if self.df.empty or len(self.df) < 10:
            return self._empty_result()
        
        # Filtrer pour la session (ex: dernières 24h)
        df_session = self._get_session_data()
        
        if len(df_session) < 10:
            df_session = self.df.tail(288)  # Fallback: 288 bougies de 5min = 24h
        
        # Calculer les bornes
        price_min = df_session['low'].min()
        price_max = df_session['high'].max()
        
        if price_max <= price_min:
            return self._empty_result()
        
        # Créer les bins
        bins = np.arange(start=price_min, stop=price_max + self.bin_size, step=self.bin_size)
        profile = pd.Series(0.0, index=bins[:-1], dtype='float64')
        
        # Distribuer le volume dans les bins
        for _, row in df_session.iterrows():
            price_bin = int((row['close'] - price_min) / self.bin_size) * self.bin_size + price_min
            if price_bin in profile.index:
                profile[price_bin] += row['volume']
        
        total_vol = profile.sum()
        
        if total_vol == 0:
            return self._empty_result()
        
        # POC (Point of Control) = bin avec le plus de volume
        poc = profile.idxmax()
        
        # Value Area (70% du volume)
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
        
        # Shape Detection
        shape = self._detect_shape(profile, poc, price_min, price_max, total_vol)
        
        return {
            'poc': round(poc, 2),
            'vah': round(vah, 2),
            'val': round(val, 2),
            'shape': shape,
            'price_range': {
                'min': round(price_min, 2),
                'max': round(price_max, 2),
                'range': round(price_max - price_min, 2)
            },
            'total_volume': round(total_vol, 2),
            'session_bars': len(df_session)
        }
    
    def _get_session_data(self) -> pd.DataFrame:
        """Récupère les données de la session actuelle"""
        if 'timestamp' not in self.df.columns:
            return self.df.tail(288)
        
        now = datetime.now(timezone.utc)
        start_of_session = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        df_session = self.df[self.df['timestamp'] >= start_of_session]
        
        return df_session
    
    def _detect_shape(self, profile: pd.Series, poc: float, 
                      price_min: float, price_max: float, total_vol: float) -> str:
        """
        Détecte la forme du profil de volume
        
        Returns:
            'P-Shape' (haussier), 'b-Shape' (baissier), 'D-Shape' (neutre)
        """
        # Diviser en tiers
        range_size = price_max - price_min
        tiers = range_size / 3
        
        high_zone = price_max - tiers
        low_zone = price_min + tiers
        
        # Volume dans chaque zone
        vol_high = profile.loc[high_zone:].sum()
        vol_low = profile.loc[:low_zone].sum()
        
        # Détection
        if poc > high_zone and vol_high > total_vol * 0.40:
            return "P-Shape"  # Haussier - volume concentré en haut
        elif poc < low_zone and vol_low > total_vol * 0.40:
            return "b-Shape"  # Baissier - volume concentré en bas
        else:
            return "D-Shape"  # Neutre - distribution équilibrée
    
    def _empty_result(self) -> Dict[str, Any]:
        """Résultat vide en cas d'erreur"""
        return {
            'poc': 0,
            'vah': 0,
            'val': 0,
            'shape': 'D-Shape',
            'price_range': {'min': 0, 'max': 0, 'range': 0},
            'total_volume': 0,
            'session_bars': 0
        }
    
    def get_position_relative_to_va(self, current_price: float) -> Dict[str, Any]:
        """
        Détermine la position du prix par rapport à la Value Area
        
        Args:
            current_price: Prix actuel
            
        Returns:
            Dict avec position et distances
        """
        result = self.analyze()
        
        vah = result['vah']
        val = result['val']
        poc = result['poc']
        
        if vah == 0:
            return {'position': 'UNKNOWN', 'near_vah': False, 'near_val': False, 'near_poc': False}
        
        near_threshold = config.NEAR_LEVEL_PCT
        
        near_vah = abs(current_price - vah) / vah < near_threshold
        near_val = abs(current_price - val) / val < near_threshold
        near_poc = abs(current_price - poc) / poc < near_threshold
        
        if current_price > vah:
            position = "ABOVE_VA"
        elif current_price < val:
            position = "BELOW_VA"
        else:
            position = "INSIDE_VA"
        
        return {
            'position': position,
            'near_vah': near_vah,
            'near_val': near_val,
            'near_poc': near_poc,
            'distance_to_vah_pct': round(abs(current_price - vah) / vah * 100, 3),
            'distance_to_val_pct': round(abs(current_price - val) / val * 100, 3),
            'distance_to_poc_pct': round(abs(current_price - poc) / poc * 100, 3)
        }
