"""
Analyseur Quantum Squeeze
- Corrélation Open Interest (OI) vs Volatilité
- Détection des phases de "chargement" (OI en hausse, Volatilité basse)
"""
from typing import Dict, List, Any
import numpy as np

class SqueezeAnalyzer:
    """Analyse la compression du marché via OI et Volatilité"""
    
    def __init__(self, candles: List[Dict], oi_change_pct: float):
        """
        Args:
            candles: Liste de bougies OHLCV
            oi_change_pct: Variation de l'Open Interest en % (ex: 1.5 pour +1.5%)
        """
        self.candles = candles
        self.oi_change_pct = oi_change_pct

    def analyze(self) -> Dict[str, Any]:
        """
        Calcule le score de Squeeze
        
        Returns:
            Dict avec squeeze_score, status, et intensité
        """
        if not self.candles or len(self.candles) < 5:
            return self._empty_result()
            
        # 1. Calculer la volatilité relative (ATR-like)
        # On utilise le range moyen des bougies récentes / prix
        ranges = []
        for c in self.candles[-14:]: # Lookback 14
            h = float(c.get('high', 0))
            l = float(c.get('low', 0))
            cl_prev = float(self.candles[self.candles.index(c)-1].get('close', h)) if self.candles.index(c) > 0 else l
            tr = max(h - l, abs(h - cl_prev), abs(l - cl_prev))
            ranges.append(tr)
            
        avg_range = sum(ranges) / len(ranges)
        current_price = float(self.candles[-1].get('close', 1))
        rel_volatility = (avg_range / current_price) * 100 # Volatilité en %
        
        # 2. Quantum Squeeze Score
        # Plus l'OI monte alors que la volatilité est basse, plus le score est haut
        # Score = OI_Change / Rel_Volatility
        if rel_volatility > 0:
            # On booste le score si la volatilité est exceptionnellement basse (< 0.2%)
            vol_factor = max(0.1, rel_volatility)
            squeeze_score = self.oi_change_pct / vol_factor
        else:
            squeeze_score = 0.0
            
        # 3. Interprétation
        if squeeze_score > 5.0 and self.oi_change_pct > 1.0:
            status = "EXTREME_SQUEEZE"
            intensity = "HIGH" # Prêt à exploser
            emoji = "🍋"
        elif squeeze_score > 2.0:
            status = "ACCUMULATION"
            intensity = "MEDIUM"
            emoji = "⌛"
        else:
            status = "NEUTRAL"
            intensity = "LOW"
            emoji = "⚪"
            
        return {
            'squeeze_score': round(squeeze_score, 2),
            'oi_change_pct': round(self.oi_change_pct, 2),
            'rel_volatility': round(rel_volatility, 3),
            'status': status,
            'intensity': intensity,
            'emoji': emoji,
            'is_squeeze': squeeze_score > 2.0
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'squeeze_score': 0.0,
            'oi_change_pct': 0.0,
            'rel_volatility': 0.0,
            'status': 'NEUTRAL',
            'intensity': 'LOW',
            'emoji': '⚪',
            'is_squeeze': False
        }
