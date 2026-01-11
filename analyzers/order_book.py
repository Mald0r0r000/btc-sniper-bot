"""
Analyseur du Carnet d'Ordres (Order Book)
- Imbalance bid/ask
- Détection des murs (walls)
"""
from typing import Dict, List, Tuple, Any

import config


class OrderBookAnalyzer:
    """Analyse le carnet d'ordres pour détecter pression et murs"""
    
    def __init__(self, order_book: Dict[str, List], current_price: float):
        """
        Args:
            order_book: Dict avec 'bids' et 'asks'
            current_price: Prix actuel
        """
        self.bids = order_book.get('bids', [])
        self.asks = order_book.get('asks', [])
        self.current_price = current_price
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète du carnet d'ordres
        
        Returns:
            Dict avec toutes les métriques
        """
        if not self.bids or not self.asks:
            return self._empty_result()
        
        # 1. Volumes totaux
        vol_bids = sum(item[1] for item in self.bids)
        vol_asks = sum(item[1] for item in self.asks)
        total_vol = vol_bids + vol_asks
        
        # 2. Ratios (Imbalance)
        bid_ratio = (vol_bids / total_vol) * 100 if total_vol > 0 else 50
        ask_ratio = (vol_asks / total_vol) * 100 if total_vol > 0 else 50
        
        # 3. Détection des murs
        max_bid = max(self.bids, key=lambda x: x[1])
        max_ask = max(self.asks, key=lambda x: x[1])
        
        # Valeur en millions USD
        wall_bid_val_m = (max_bid[1] * max_bid[0]) / 1_000_000
        wall_ask_val_m = (max_ask[1] * max_ask[0]) / 1_000_000
        
        # 4. Proximité des murs au prix actuel
        bid_wall_distance_pct = abs(self.current_price - max_bid[0]) / self.current_price
        ask_wall_distance_pct = abs(self.current_price - max_ask[0]) / self.current_price
        
        # 5. Danger de mur (proche + gros)
        wall_danger_bid = (
            bid_wall_distance_pct < config.WALL_PROXIMITY_PCT and 
            wall_bid_val_m > config.MIN_WALL_VALUE_M
        )
        wall_danger_ask = (
            ask_wall_distance_pct < config.WALL_PROXIMITY_PCT and 
            wall_ask_val_m > config.MIN_WALL_VALUE_M
        )
        
        # 6. Interprétation
        pressure = "NEUTRE"
        if bid_ratio > 60:
            pressure = "HAUSSIERE"
        elif ask_ratio > 60:
            pressure = "BAISSIERE"
        
        return {
            'bid_volume': vol_bids,
            'ask_volume': vol_asks,
            'bid_ratio_pct': round(bid_ratio, 1),
            'ask_ratio_pct': round(ask_ratio, 1),
            'pressure': pressure,
            'wall_bid': {
                'price': max_bid[0],
                'volume': max_bid[1],
                'value_m': round(wall_bid_val_m, 2),
                'distance_pct': round(bid_wall_distance_pct * 100, 3),
                'is_danger': wall_danger_bid
            },
            'wall_ask': {
                'price': max_ask[0],
                'volume': max_ask[1],
                'value_m': round(wall_ask_val_m, 2),
                'distance_pct': round(ask_wall_distance_pct * 100, 3),
                'is_danger': wall_danger_ask
            }
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Résultat vide en cas d'erreur"""
        return {
            'bid_volume': 0,
            'ask_volume': 0,
            'bid_ratio_pct': 50,
            'ask_ratio_pct': 50,
            'pressure': 'NEUTRE',
            'wall_bid': {'price': 0, 'volume': 0, 'value_m': 0, 'distance_pct': 0, 'is_danger': False},
            'wall_ask': {'price': 0, 'volume': 0, 'value_m': 0, 'distance_pct': 0, 'is_danger': False}
        }
    
    def get_visual_bar(self, width: int = 20) -> str:
        """
        Génère une barre visuelle de la pression
        
        Args:
            width: Largeur de la barre
            
        Returns:
            String avec emojis 🟩/🟥
        """
        result = self.analyze()
        bars_green = int(result['bid_ratio_pct'] / 100 * width)
        bars_red = width - bars_green
        return "🟩" * bars_green + "🟥" * bars_red
