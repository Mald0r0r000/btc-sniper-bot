"""
Analyseur CVD (Cumulative Volume Delta)
- Analyse des trades récents
- Ratio d'agression Taker Buy vs Sell
"""
from typing import Dict, List, Any


class CVDAnalyzer:
    """Analyse le Cumulative Volume Delta basé sur les trades récents"""
    
    def __init__(self, trades: List[Dict]):
        """
        Args:
            trades: Liste des trades avec 'side' et 'amount'
        """
        self.trades = trades
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse le CVD et le ratio d'agression
        
        Returns:
            Dict avec net_cvd, buy_volume, sell_volume, aggression_ratio, status
        """
        if not self.trades:
            return self._empty_result()
        
        buy_vol = 0.0
        sell_vol = 0.0
        
        for trade in self.trades:
            amount = float(trade.get('amount', 0))
            side = trade.get('side', '')
            
            if side == 'buy':
                buy_vol += amount
            else:
                sell_vol += amount
        
        # Net CVD
        net_cvd = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol
        
        # Ratio d'agression (> 1.0 = acheteurs dominent)
        aggression_ratio = buy_vol / sell_vol if sell_vol > 0 else 10.0
        
        # Interprétation
        if aggression_ratio > 1.2:
            status = "AGRESSION_ACHETEUSE"
            emoji = "🟢"
        elif aggression_ratio < 0.8:
            status = "AGRESSION_VENDEUSE"
            emoji = "🔴"
        else:
            status = "NEUTRE"
            emoji = "⚪"
        
        return {
            'net_cvd': round(net_cvd, 4),
            'buy_volume': round(buy_vol, 4),
            'sell_volume': round(sell_vol, 4),
            'total_volume': round(total_vol, 4),
            'aggression_ratio': round(aggression_ratio, 2),
            'status': status,
            'emoji': emoji,
            'is_bullish': aggression_ratio > 1.2,
            'is_bearish': aggression_ratio < 0.8
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Résultat vide en cas d'erreur"""
        return {
            'net_cvd': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'total_volume': 0,
            'aggression_ratio': 1.0,
            'status': 'NEUTRE',
            'emoji': '⚪',
            'is_bullish': False,
            'is_bearish': False
        }
