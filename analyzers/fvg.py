"""
Analyseur FVG (Fair Value Gaps) - Multi-Timeframe
- Détection des gaps haussiers et baissiers
- Intensité de volume (high vol vs normal)
- Tracking de mitigation (remplissage complet)
- Agrégation MTF (5m + 1h + 1d)
"""
import pandas as pd
from typing import Dict, List, Any, Optional

import config


class FVGAnalyzer:
    """Analyse les Fair Value Gaps sur plusieurs timeframes"""
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Args:
            dataframes: Dict de DataFrames par timeframe
                        ex: {'5m': df_5m, '1h': df_1h, '1d': df_1d}
        """
        self.dataframes = dataframes
        self.fvg_width = config.FVG_WIDTH_BARS
        self.vol_avg_length = config.VOL_AVG_LENGTH
        self.vol_multiplier = config.VOL_MULTIPLIER
        self.min_size_pct = config.FVG_MIN_SIZE_PCT / 100
    
    def analyze(self, current_price: float) -> Dict[str, Any]:
        """
        Analyse complète des FVG sur tous les timeframes
        
        Args:
            current_price: Prix actuel pour calculer les distances
            
        Returns:
            Dict avec FVG par timeframe et agrégation
        """
        all_fvg = {}
        all_active_fvg = []
        
        for tf, df in self.dataframes.items():
            if df.empty or len(df) < 20:
                all_fvg[tf] = {'bull': [], 'bear': [], 'total': 0, 'active': 0}
                continue
            
            fvg_list = self._detect_fvg(df, tf)
            
            # Vérifier mitigation
            fvg_list = self._check_mitigation(df, fvg_list)
            
            # Calculer distances au prix actuel
            for fvg in fvg_list:
                fvg['distance_pct'] = self._calculate_distance(fvg, current_price)
            
            # Séparer bull et bear
            bull_fvg = [f for f in fvg_list if f['type'] == 'bull']
            bear_fvg = [f for f in fvg_list if f['type'] == 'bear']
            
            active_fvg = [f for f in fvg_list if not f['mitigated']]
            all_active_fvg.extend(active_fvg)
            
            all_fvg[tf] = {
                'bull': bull_fvg,
                'bear': bear_fvg,
                'total': len(fvg_list),
                'active': len(active_fvg),
                'high_vol_count': len([f for f in fvg_list if f['high_vol']])
            }
        
        # Trouver les FVG les plus proches
        nearest_bull = self._find_nearest_fvg(all_active_fvg, current_price, 'bull')
        nearest_bear = self._find_nearest_fvg(all_active_fvg, current_price, 'bear')
        
        return {
            'by_timeframe': all_fvg,
            'total_active': len(all_active_fvg),
            'nearest_bull': nearest_bull,
            'nearest_bear': nearest_bear,
            'all_active': sorted(all_active_fvg, key=lambda x: abs(x['distance_pct']))[:10]  # Top 10 plus proches
        }
    
    def _detect_fvg(self, df: pd.DataFrame, timeframe: str) -> List[Dict[str, Any]]:
        """
        Détecte les FVG dans un DataFrame
        
        Bull FVG: low[i] > high[i-2] (gap haussier)
        Bear FVG: high[i] < low[i-2] (gap baissier)
        """
        fvg_list = []
        
        # Calculer la moyenne de volume
        if 'volume' in df.columns:
            df = df.copy()
            df['vol_avg'] = df['volume'].rolling(self.vol_avg_length).mean()
        else:
            return []
        
        for i in range(2, len(df)):
            close_price = df['close'].iloc[i]
            
            # Bull FVG
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_top = df['low'].iloc[i]
                gap_bottom = df['high'].iloc[i-2]
                gap_size_pct = (gap_top - gap_bottom) / close_price
                
                # Vérifier taille minimum
                if gap_size_pct < self.min_size_pct:
                    continue
                
                # Volume au moment du move
                vol_at_move = df['volume'].iloc[i-1]
                vol_avg = df['vol_avg'].iloc[i-1]
                is_high_vol = vol_at_move > (vol_avg * self.vol_multiplier) if vol_avg > 0 else False
                
                fvg_list.append({
                    'timeframe': timeframe,
                    'type': 'bull',
                    'bar_index': i,
                    'top': round(gap_top, 2),
                    'bottom': round(gap_bottom, 2),
                    'midpoint': round((gap_top + gap_bottom) / 2, 2),
                    'limit_level': round(gap_bottom, 2),  # Niveau à atteindre pour mitigation
                    'size_pct': round(gap_size_pct * 100, 3),
                    'high_vol': is_high_vol,
                    'mitigated': False
                })
            
            # Bear FVG
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_top = df['low'].iloc[i-2]
                gap_bottom = df['high'].iloc[i]
                gap_size_pct = (gap_top - gap_bottom) / close_price
                
                # Vérifier taille minimum
                if gap_size_pct < self.min_size_pct:
                    continue
                
                # Volume au moment du move
                vol_at_move = df['volume'].iloc[i-1]
                vol_avg = df['vol_avg'].iloc[i-1]
                is_high_vol = vol_at_move > (vol_avg * self.vol_multiplier) if vol_avg > 0 else False
                
                fvg_list.append({
                    'timeframe': timeframe,
                    'type': 'bear',
                    'bar_index': i,
                    'top': round(gap_top, 2),
                    'bottom': round(gap_bottom, 2),
                    'midpoint': round((gap_top + gap_bottom) / 2, 2),
                    'limit_level': round(gap_top, 2),  # Niveau à atteindre pour mitigation
                    'size_pct': round(gap_size_pct * 100, 3),
                    'high_vol': is_high_vol,
                    'mitigated': False
                })
        
        return fvg_list
    
    def _check_mitigation(self, df: pd.DataFrame, fvg_list: List[Dict]) -> List[Dict]:
        """
        Vérifie si les FVG ont été remplis (mitigated)
        
        Bull FVG mitigated: le prix revient toucher le bottom (limit_level)
        Bear FVG mitigated: le prix revient toucher le top (limit_level)
        """
        for fvg in fvg_list:
            start_check = fvg['bar_index'] + 1  # Vérifier APRÈS la création
            
            for j in range(start_check, len(df)):
                if fvg['type'] == 'bull':
                    # Bull FVG rempli si le prix descend jusqu'au limit_level
                    if df['low'].iloc[j] <= fvg['limit_level']:
                        fvg['mitigated'] = True
                        fvg['mitigation_bar'] = j
                        break
                else:
                    # Bear FVG rempli si le prix monte jusqu'au limit_level
                    if df['high'].iloc[j] >= fvg['limit_level']:
                        fvg['mitigated'] = True
                        fvg['mitigation_bar'] = j
                        break
        
        return fvg_list
    
    def _calculate_distance(self, fvg: Dict, current_price: float) -> float:
        """Calcule la distance en % entre le FVG et le prix actuel"""
        # Utiliser le midpoint pour la distance
        midpoint = fvg['midpoint']
        distance_pct = (midpoint - current_price) / current_price * 100
        return round(distance_pct, 3)
    
    def _find_nearest_fvg(self, fvg_list: List[Dict], current_price: float, 
                          fvg_type: str) -> Optional[Dict]:
        """
        Trouve le FVG actif le plus proche d'un type donné
        
        Args:
            fvg_list: Liste des FVG
            current_price: Prix actuel
            fvg_type: 'bull' ou 'bear'
            
        Returns:
            FVG le plus proche ou None
        """
        filtered = [f for f in fvg_list if f['type'] == fvg_type and not f['mitigated']]
        
        if not filtered:
            return None
        
        # Trier par distance absolue
        sorted_fvg = sorted(filtered, key=lambda x: abs(x['distance_pct']))
        
        return sorted_fvg[0] if sorted_fvg else None
    
    def get_fvg_near_price(self, current_price: float, threshold_pct: float = 0.5) -> Dict[str, Any]:
        """
        Trouve les FVG actifs proches du prix actuel
        
        Args:
            current_price: Prix actuel
            threshold_pct: Seuil de distance en %
            
        Returns:
            Dict avec FVG proches (bull et bear)
        """
        result = self.analyze(current_price)
        
        near_bull = None
        near_bear = None
        
        if result['nearest_bull'] and abs(result['nearest_bull']['distance_pct']) < threshold_pct:
            near_bull = result['nearest_bull']
        
        if result['nearest_bear'] and abs(result['nearest_bear']['distance_pct']) < threshold_pct:
            near_bear = result['nearest_bear']
        
        return {
            'has_near_fvg': near_bull is not None or near_bear is not None,
            'near_bull': near_bull,
            'near_bear': near_bear
        }
