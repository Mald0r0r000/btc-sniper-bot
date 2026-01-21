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
        
        # R&D: Volume Skew & Breakout Pressure (D-Shape Logic)
        skew = self._calculate_volume_skew(profile, price_min, price_max, total_vol)
        
        return {
            'poc': round(poc, 2),
            'vah': round(vah, 2),
            'val': round(val, 2),
            'shape': shape,
            'skew': skew,  # > 0 means volume concentrated higher (Bullish Pressure), < 0 means lower (Bearish Pressure)
            'price_range': {
                'min': round(price_min, 2),
                'max': round(price_max, 2),
                'range': round(price_max - price_min, 2)
            },
            'total_volume': round(total_vol, 2),
            'session_bars': len(df_session)
        }
    
    def _calculate_volume_skew(self, profile: pd.Series, price_min: float, price_max: float, total_vol: float) -> float:
        """
        Calcule le skew du volume (Distribution Asymétrique)
        > 0 : Volume concentré dans la moitié haute (Pression haussière)
        < 0 : Volume concentré dans la moitié basse (Pression baissière)
        """
        if total_vol == 0:
            return 0.0
            
        mid_price = (price_min + price_max) / 2
        
        vol_above_mid = profile.loc[mid_price:].sum()
        vol_below_mid = profile.loc[:mid_price].sum()
        
        # Skew normalisé entre -1 et 1
        # Exemple: Si 70% du vol est en haut, skew = 0.4
        skew = (vol_above_mid - vol_below_mid) / total_vol
        return round(skew, 2)

    def detect_breakout_pressure(self, current_price: float, skew: float, cvd_trend: str = 'NEUTRAL') -> Dict[str, Any]:
        """
        Anticipe la direction du breakout d'un D-Shape
        
        Args:
            current_price: Prix actuel
            skew: Volume skew (-1 à 1)
            cvd_trend: Tendance CVD (BULLISH/BEARISH/NEUTRAL)
        """
        pressure_score = 0
        direction = 'NEUTRAL'
        
        # 1. Volume Skew Pressure
        if skew > 0.15:
            pressure_score += 30
            direction_bias = 'UP'
        elif skew < -0.15:
            pressure_score += 30
            direction_bias = 'DOWN'
        else:
            direction_bias = 'NEUTRAL'
            
        # 2. CVD Confirmation (Accumulation/Distribution)
        if cvd_trend == 'BULLISH':
            if direction_bias == 'UP':
                pressure_score += 40  # Confirmation forte
            elif direction_bias == 'NEUTRAL':
                pressure_score += 20
                direction_bias = 'UP'
        elif cvd_trend == 'BEARISH':
            if direction_bias == 'DOWN':
                pressure_score += 40
            elif direction_bias == 'NEUTRAL':
                pressure_score += 20
                direction_bias = 'DOWN'
                
        # 3. Price Proximity Pressure
        result = self.analyze() # Note: Inefficient if called repeatedly, better to pass va bounds
        vah = result['vah']
        val = result['val']
        
        # Pression aux bornes
        near_threshold = 0.005 # 0.5%
        if abs(current_price - vah) / vah < near_threshold and direction_bias == 'UP':
            pressure_score += 30
        elif abs(current_price - val) / val < near_threshold and direction_bias == 'DOWN':
            pressure_score += 30
            
        return {
            'pressure_score': min(100, pressure_score),
            'direction': direction_bias if pressure_score > 40 else 'NEUTRAL',
            'skew': skew,
            'is_breakout_imminent': pressure_score > 70
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
