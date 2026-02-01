"""
Liquidation Heatmap Analyzer (Estimation)
Estime les zones de liquidation basées sur les Swing Highs/Lows et les leviers standards.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class LiquidationLevel:
    price: float
    volume_est: float
    leverage: int
    type: str # 'LONG_LIQ' (below price) or 'SHORT_LIQ' (above price)
    reference_price: float # The swing high/low

class LiquidationAnalyzer:
    """
    Estime les clusters de liquidation.
    Hypothèse: Les traders entrent sur les Swing Highs (Shorts) et Swing Lows (Longs).
    Leviers: 25x, 50x, 100x.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame avec 'high', 'low', 'close' (Timeframe recommandé: 1h ou 4h)
        """
        self.df = df
        # Leviers à tracker
        self.leverages = [25, 50, 100]
        # Distance de liquidation approx: Entry * (1 +/- 1/Lev)
        # On ajoute un buffer de maintenance margin (ex: 0.5%)
        # Liq Long = Entry * (1 - (1/Lev - 0.005)) 
        # Pour simplifier: Liq = Entry * (1 +/- 0.95/Lev)
        
    def analyze(self, current_price: float) -> Dict[str, Any]:
        """
        Analyse les clusters de liquidation
        """
        if self.df.empty:
            return self._empty_result()
            
        swings = self._find_swing_points(window=5) # 5 candles left/right
        levels = self._calculate_liquidation_levels(swings)
        clusters = self._cluster_levels(levels, current_price)
        
        # Trouver les clusters les plus proches
        nearest_long = self._find_nearest_cluster(clusters['longs'], current_price, is_long=True)
        nearest_short = self._find_nearest_cluster(clusters['shorts'], current_price, is_long=False)
        
        return {
            "nearest_long_liq": nearest_long,
            "nearest_short_liq": nearest_short,
            "total_long_liqs_est": len(clusters['longs']),
            "total_short_liqs_est": len(clusters['shorts']),
            "clusters_json": {
                "longs": [c for c in clusters['longs'][:5]], # Top 5 closest
                "shorts": [c for c in clusters['shorts'][:5]]
            }
        }
        
    def _find_swing_points(self, window: int) -> List[Tuple[float, str]]:
        """Trouve les Swing Highs et Lows locaux"""
        swings = []
        
        # On regarde les n dernières bougies (ex: 100)
        recent_df = self.df.tail(100).copy()
        
        for i in range(window, len(recent_df) - window):
            # Check High
            if recent_df['high'].iloc[i] == recent_df['high'].iloc[i-window:i+window+1].max():
                swings.append((recent_df['high'].iloc[i], 'HIGH'))
            
            # Check Low
            if recent_df['low'].iloc[i] == recent_df['low'].iloc[i-window:i+window+1].min():
                swings.append((recent_df['low'].iloc[i], 'LOW'))
                
        return swings

    def _calculate_liquidation_levels(self, swings: List[Tuple[float, str]]) -> List[LiquidationLevel]:
        levels = []
        
        for price, swing_type in swings:
            for lev in self.leverages:
                # Si Swing HIGH -> Des gens ont SHORT -> Liq au-dessus
                if swing_type == 'HIGH':
                    # Short Liq = Price * (1 + 1/Lev)
                    # Ajustement MM: Price * (1 + (1/Lev * 0.9)) estimation
                    liq_price = price * (1 + (1/lev))
                    levels.append(LiquidationLevel(liq_price, 1000, lev, 'SHORT_LIQ', price)) # Vol arbitraire pour l'instant
                
                # Si Swing LOW -> Des gens ont LONG -> Liq en-dessous
                elif swing_type == 'LOW':
                    # Long Liq = Price * (1 - 1/Lev)
                    liq_price = price * (1 - (1/lev))
                    levels.append(LiquidationLevel(liq_price, 1000, lev, 'LONG_LIQ', price))
                    
        return levels

    def _cluster_levels(self, levels: List[LiquidationLevel], current_price: float) -> Dict[str, List[Dict]]:
        """Regroupe les niveaux proches en clusters"""
        # Séparer Longs (sont en dessous du prix actuel) et Shorts (au dessus)
        long_liqs = sorted([l.price for l in levels if l.type == 'LONG_LIQ' and l.price < current_price], reverse=True)
        short_liqs = sorted([l.price for l in levels if l.type == 'SHORT_LIQ' and l.price > current_price])
        
        return {
            'longs': self._make_clusters(long_liqs),
            'shorts': self._make_clusters(short_liqs)
        }
        
    def _make_clusters(self, prices: List[float], tolerance_pct: float = 0.005) -> List[Dict]:
        """Cluster simple: regroupe prix si distance < 0.5%"""
        if not prices:
            return []
            
        clusters = []
        current_cluster = [prices[0]]
        
        for p in prices[1:]:
            if abs(p - current_cluster[-1]) / current_cluster[-1] < tolerance_pct:
                current_cluster.append(p)
            else:
                # Fin du cluster
                avg = sum(current_cluster) / len(current_cluster)
                clusters.append({
                    "price": round(avg, 2),
                    "intensity": len(current_cluster), # Nombre de niveaux confluents
                    "volume_score": len(current_cluster) * 10 # Score arbitraire
                })
                current_cluster = [p]
                
        # Dernier cluster
        if current_cluster:
            avg = sum(current_cluster) / len(current_cluster)
            clusters.append({
                "price": round(avg, 2),
                "intensity": len(current_cluster),
                "volume_score": len(current_cluster) * 10
            })
            
        # Trier par intensité
        return sorted(clusters, key=lambda x: x['intensity'], reverse=True)

    def _find_nearest_cluster(self, clusters: List[Dict], current_price: float, is_long: bool) -> Dict:
        if not clusters:
            return {"price": 0, "distance_pct": 0, "intensity": 0}
            
        # Trouver le plus proche par distance
        nearest = min(clusters, key=lambda x: abs(x['price'] - current_price))
        dist_pct = (nearest['price'] - current_price) / current_price * 100
        
        return {
            "price": nearest['price'],
            "distance_pct": round(dist_pct, 2),
            "intensity": nearest['intensity']
        }

    def _empty_result(self):
        return {
            "nearest_long_liq": {"price": 0, "distance_pct": 0, "intensity": 0},
            "nearest_short_liq": {"price": 0, "distance_pct": 0, "intensity": 0},
            "total_long_liqs_est": 0,
            "total_short_liqs_est": 0
        }
