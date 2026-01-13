"""
Liquidation Zones Analyzer
Calcule les zones de liquidation pour générer des TP/SL intelligents

Principe:
- Les zones de liquidation sont des "aimants" pour le prix
- Le marché va souvent chercher ces zones pour absorber la liquidité
- TP = Zone de liquidation adverse (où les trades opposés seront liquidés)
- SL = Zone de liquidation à éviter (où nous serions liquidés)
"""
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
import numpy as np


class LiquidationZoneAnalyzer:
    """
    Analyse les zones de liquidation pour TP/SL dynamiques
    
    Basé sur:
    - Pivots hauts/bas récents
    - Calcul des prix de liquidation pour différents leviers
    - Clustering des zones proches
    """
    
    # Leviers les plus communs sur les exchanges
    LEVERAGES = [100, 50, 25, 10]
    
    # Lookback periods pour différents tiers (en bougies)
    TIERS = {
        'scalping': 5,      # Court terme
        'intraday': 20,     # Moyen terme
        'swing': 50         # Long terme
    }
    
    def __init__(self, mm_factors: Dict[int, float] = None):
        """
        Args:
            mm_factors: Maintenance margin factors par levier
        """
        # Facteurs de maintenance margin par levier (estimation Bitget/Binance)
        self.mm_factors = mm_factors or {
            100: 0.50,
            50: 0.60,
            25: 0.80,
            10: 0.90
        }
    
    def analyze(self, candles_5m: List[Dict], 
                candles_1h: List[Dict] = None,
                current_price: float = None) -> Dict[str, Any]:
        """
        Analyse complète des zones de liquidation
        
        Args:
            candles_5m: Bougies 5min pour scalping/intraday
            candles_1h: Bougies 1h pour swing (optionnel)
            current_price: Prix actuel
        
        Returns:
            Dict avec zones, clusters, et recommandations
        """
        if not candles_5m or len(candles_5m) < 20:
            return self._empty_result()
        
        # Extraire les highs et lows
        highs = [float(c.get('high', c.get('h', 0))) for c in candles_5m]
        lows = [float(c.get('low', c.get('l', 0))) for c in candles_5m]
        
        if not current_price:
            current_price = float(candles_5m[-1].get('close', candles_5m[-1].get('c', 0)))
        
        # Trouver tous les pivots
        all_zones = []
        
        # Tier 1: Scalping (court terme)
        pivot_highs = self._find_pivot_highs(highs, self.TIERS['scalping'])
        pivot_lows = self._find_pivot_lows(lows, self.TIERS['scalping'])
        all_zones.extend(self._calculate_liquidation_zones(pivot_highs, is_long=False, tier='scalping'))
        all_zones.extend(self._calculate_liquidation_zones(pivot_lows, is_long=True, tier='scalping'))
        
        # Tier 2: Intraday (moyen terme)
        pivot_highs = self._find_pivot_highs(highs, self.TIERS['intraday'])
        pivot_lows = self._find_pivot_lows(lows, self.TIERS['intraday'])
        all_zones.extend(self._calculate_liquidation_zones(pivot_highs, is_long=False, tier='intraday'))
        all_zones.extend(self._calculate_liquidation_zones(pivot_lows, is_long=True, tier='intraday'))
        
        # Tier 3: Swing (long terme) - si assez de données
        if len(highs) >= 60:
            pivot_highs = self._find_pivot_highs(highs, self.TIERS['swing'])
            pivot_lows = self._find_pivot_lows(lows, self.TIERS['swing'])
            all_zones.extend(self._calculate_liquidation_zones(pivot_highs, is_long=False, tier='swing'))
            all_zones.extend(self._calculate_liquidation_zones(pivot_lows, is_long=True, tier='swing'))
        
        # Séparer les zones au-dessus et en-dessous du prix
        zones_above = [z for z in all_zones if z['price'] > current_price]
        zones_below = [z for z in all_zones if z['price'] < current_price]
        
        # Trier par proximité
        zones_above.sort(key=lambda x: x['price'])
        zones_below.sort(key=lambda x: x['price'], reverse=True)
        
        # Cluster les zones proches
        clusters_above = self._cluster_zones(zones_above, current_price)
        clusters_below = self._cluster_zones(zones_below, current_price)
        
        return {
            'current_price': current_price,
            'zones_above': zones_above[:10],  # Top 10 plus proches
            'zones_below': zones_below[:10],
            'clusters_above': clusters_above,
            'clusters_below': clusters_below,
            'total_zones': len(all_zones),
            'analysis': {
                'nearest_above': clusters_above[0] if clusters_above else None,
                'nearest_below': clusters_below[0] if clusters_below else None,
                'liquidity_bias': self._calculate_liquidity_bias(clusters_above, clusters_below)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_targets_for_direction(self, analysis: Dict, direction: str) -> Dict[str, float]:
        """
        Génère les TP/SL basés sur les zones de liquidation
        
        Args:
            analysis: Résultat de analyze()
            direction: 'LONG' ou 'SHORT'
        
        Returns:
            Dict avec tp1, tp2, sl
        """
        current_price = analysis.get('current_price', 0)
        clusters_above = analysis.get('clusters_above', [])
        clusters_below = analysis.get('clusters_below', [])
        
        targets = {}
        
        if direction == 'LONG':
            # LONG: TP = zones de liq des SHORTS au-dessus (ils seront squeeze)
            # SL = zone de liq des LONGS en-dessous (on veut pas être liquidé avec eux)
            
            if clusters_above:
                # Premier cluster de shorts au-dessus = TP1
                short_clusters = [c for c in clusters_above if not c.get('is_long', True)]
                if short_clusters:
                    targets['tp1'] = short_clusters[0]['center_price']
                    if len(short_clusters) > 1:
                        targets['tp2'] = short_clusters[1]['center_price']
                else:
                    # Fallback: premier cluster au-dessus
                    targets['tp1'] = clusters_above[0]['center_price']
            
            if clusters_below:
                # Zone de liq des longs en-dessous = SL (avec buffer)
                long_clusters = [c for c in clusters_below if c.get('is_long', False)]
                if long_clusters:
                    # SL juste en-dessous de la zone de liq
                    targets['sl'] = long_clusters[0]['center_price'] * 0.998
                else:
                    # Fallback: premier cluster en-dessous
                    targets['sl'] = clusters_below[0]['center_price'] * 0.998
        
        elif direction == 'SHORT':
            # SHORT: TP = zones de liq des LONGS en-dessous (ils seront flush)
            # SL = zone de liq des SHORTS au-dessus
            
            if clusters_below:
                long_clusters = [c for c in clusters_below if c.get('is_long', False)]
                if long_clusters:
                    targets['tp1'] = long_clusters[0]['center_price']
                    if len(long_clusters) > 1:
                        targets['tp2'] = long_clusters[1]['center_price']
                else:
                    targets['tp1'] = clusters_below[0]['center_price']
            
            if clusters_above:
                short_clusters = [c for c in clusters_above if not c.get('is_long', True)]
                if short_clusters:
                    targets['sl'] = short_clusters[0]['center_price'] * 1.002
                else:
                    targets['sl'] = clusters_above[0]['center_price'] * 1.002
        
        # Fallbacks si pas assez de zones
        if 'tp1' not in targets:
            targets['tp1'] = current_price * (1.01 if direction == 'LONG' else 0.99)
        if 'tp2' not in targets:
            targets['tp2'] = current_price * (1.02 if direction == 'LONG' else 0.98)
        if 'sl' not in targets:
            targets['sl'] = current_price * (0.995 if direction == 'LONG' else 1.005)
        
        return {k: round(v, 1) for k, v in targets.items()}
    
    def _find_pivot_highs(self, highs: List[float], lookback: int) -> List[Tuple[int, float]]:
        """Trouve les pivot highs (résistances locales)"""
        pivots = []
        n = len(highs)
        
        for i in range(lookback, n - lookback):
            window = highs[i - lookback:i + lookback + 1]
            if highs[i] == max(window):
                pivots.append((i, highs[i]))
        
        return pivots
    
    def _find_pivot_lows(self, lows: List[float], lookback: int) -> List[Tuple[int, float]]:
        """Trouve les pivot lows (supports locaux)"""
        pivots = []
        n = len(lows)
        
        for i in range(lookback, n - lookback):
            window = lows[i - lookback:i + lookback + 1]
            if lows[i] == min(window):
                pivots.append((i, lows[i]))
        
        return pivots
    
    def _calculate_liquidation_price(self, entry_price: float, leverage: int, is_long: bool) -> float:
        """
        Calcule le prix de liquidation
        
        Formule simplifiée:
        - Long: liq_price = entry - (entry / leverage) * mm_factor
        - Short: liq_price = entry + (entry / leverage) * mm_factor
        """
        mm_factor = self.mm_factors.get(leverage, 0.80)
        movement = (entry_price / leverage) * mm_factor
        
        if is_long:
            return entry_price - movement
        else:
            return entry_price + movement
    
    def _calculate_liquidation_zones(self, pivots: List[Tuple[int, float]], 
                                     is_long: bool, tier: str) -> List[Dict]:
        """Calcule les zones de liquidation pour une liste de pivots"""
        zones = []
        
        for idx, pivot_price in pivots:
            for leverage in self.LEVERAGES:
                liq_price = self._calculate_liquidation_price(pivot_price, leverage, is_long)
                
                zones.append({
                    'price': liq_price,
                    'pivot_price': pivot_price,
                    'leverage': leverage,
                    'is_long': is_long,
                    'tier': tier,
                    'index': idx
                })
        
        return zones
    
    def _cluster_zones(self, zones: List[Dict], current_price: float,
                       cluster_threshold_pct: float = 0.3) -> List[Dict]:
        """
        Regroupe les zones proches en clusters
        
        Args:
            zones: Zones triées par proximité
            current_price: Prix actuel pour calculer le %
            cluster_threshold_pct: Seuil de clustering (0.3% = $270 à 90k)
        """
        if not zones:
            return []
        
        clusters = []
        current_cluster = [zones[0]]
        
        for i in range(1, len(zones)):
            zone = zones[i]
            last_zone = current_cluster[-1]
            
            # Si proche du dernier, ajouter au cluster
            distance_pct = abs(zone['price'] - last_zone['price']) / current_price * 100
            
            if distance_pct < cluster_threshold_pct:
                current_cluster.append(zone)
            else:
                # Sauvegarder le cluster et en commencer un nouveau
                clusters.append(self._summarize_cluster(current_cluster, current_price))
                current_cluster = [zone]
        
        # Ne pas oublier le dernier cluster
        if current_cluster:
            clusters.append(self._summarize_cluster(current_cluster, current_price))
        
        return clusters
    
    def _summarize_cluster(self, zones: List[Dict], current_price: float) -> Dict:
        """Résume un cluster de zones"""
        prices = [z['price'] for z in zones]
        leverages = [z['leverage'] for z in zones]
        
        center_price = np.mean(prices)
        
        return {
            'center_price': round(center_price, 1),
            'min_price': round(min(prices), 1),
            'max_price': round(max(prices), 1),
            'zone_count': len(zones),
            'leverages': list(set(leverages)),
            'is_long': zones[0]['is_long'],  # Tous ont la même direction
            'tiers': list(set(z['tier'] for z in zones)),
            'strength': len(zones),  # Plus de zones = plus fort
            'distance_pct': round(abs(center_price - current_price) / current_price * 100, 2)
        }
    
    def _calculate_liquidity_bias(self, clusters_above: List, clusters_below: List) -> str:
        """Détermine le biais de liquidité"""
        strength_above = sum(c['strength'] for c in clusters_above[:3]) if clusters_above else 0
        strength_below = sum(c['strength'] for c in clusters_below[:3]) if clusters_below else 0
        
        if strength_above > strength_below * 1.5:
            return 'SHORTS_ABOVE'  # Plus de shorts à liquider au-dessus
        elif strength_below > strength_above * 1.5:
            return 'LONGS_BELOW'  # Plus de longs à liquider en-dessous
        else:
            return 'BALANCED'
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'current_price': 0,
            'zones_above': [],
            'zones_below': [],
            'clusters_above': [],
            'clusters_below': [],
            'total_zones': 0,
            'analysis': {
                'nearest_above': None,
                'nearest_below': None,
                'liquidity_bias': 'UNKNOWN'
            }
        }


def test_liquidation_analyzer():
    """Test du module"""
    # Simuler des candles
    import random
    
    base_price = 94000
    candles = []
    
    for i in range(100):
        variation = random.uniform(-200, 200)
        high = base_price + abs(variation) + random.uniform(50, 150)
        low = base_price - abs(variation) - random.uniform(50, 150)
        
        candles.append({
            'high': high,
            'low': low,
            'close': base_price + variation
        })
        base_price += variation * 0.1
    
    analyzer = LiquidationZoneAnalyzer()
    result = analyzer.analyze(candles, current_price=94000)
    
    print(f"Zones totales: {result['total_zones']}")
    print(f"Clusters au-dessus: {len(result['clusters_above'])}")
    print(f"Clusters en-dessous: {len(result['clusters_below'])}")
    print(f"Biais: {result['analysis']['liquidity_bias']}")
    
    # Test des targets
    targets_long = analyzer.get_targets_for_direction(result, 'LONG')
    print(f"\nTargets LONG: {targets_long}")
    
    targets_short = analyzer.get_targets_for_direction(result, 'SHORT')
    print(f"Targets SHORT: {targets_short}")


if __name__ == "__main__":
    test_liquidation_analyzer()
