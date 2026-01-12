"""
Fluid Dynamics Analyzers
Concepts physiques appliqués au trading

1. SelfTradingDetector - Détecte le wash trading et manipulations
2. VenturiAnalyzer - Prédit les breakouts par compression de liquidité
"""
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
import numpy as np


class SelfTradingDetector:
    """
    Détecte les patterns de wash trading / self-trading
    
    Patterns détectés:
    - Volume élevé sans impact sur le prix
    - CVD divergence (prix baisse mais achats cachés)
    - Symétrie buy/sell suspecte
    """
    
    def __init__(self):
        self.volume_impact_threshold = 2.0  # Volume > 2x moyenne
        self.price_impact_threshold = 0.1   # <0.1% de mouvement
        self.cvd_divergence_threshold = 0.3 # Prix bouge >0.3% mais CVD neutre
    
    def analyze(self, trades: List[Dict], 
                current_price: float,
                cvd_data: Dict = None) -> Dict[str, Any]:
        """
        Analyse les trades pour détecter du wash trading
        
        Args:
            trades: Liste des trades récents
            current_price: Prix actuel
            cvd_data: Données CVD si disponibles
        """
        if not trades or len(trades) < 50:
            return self._empty_result()
        
        # Calculer les métriques
        volume_impact = self._calculate_volume_impact(trades)
        cvd_divergence = self._detect_cvd_divergence(trades, cvd_data)
        symmetry = self._calculate_buy_sell_symmetry(trades)
        
        # Calculer la probabilité de wash trading
        probability = self._calculate_probability(
            volume_impact, cvd_divergence, symmetry
        )
        
        # Déterminer le type de manipulation
        manipulation_type = self._determine_type(cvd_divergence, symmetry)
        
        # Signal modifier (pénalité si wash trading détecté)
        signal_modifier = self._calculate_modifier(probability, manipulation_type)
        
        return {
            'detected': probability > 50,
            'probability': round(probability, 1),
            'type': manipulation_type,
            'metrics': {
                'volume_impact_ratio': round(volume_impact['ratio'], 2),
                'cvd_divergence': cvd_divergence,
                'buy_sell_symmetry': round(symmetry, 2)
            },
            'signal_modifier': signal_modifier,
            'interpretation': self._interpret(probability, manipulation_type),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_volume_impact(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calcule le ratio volume/impact sur le prix
        Volume élevé + faible impact = suspect
        """
        if len(trades) < 10:
            return {'ratio': 1.0, 'suspicious': False}
        
        # Diviser en fenêtres
        window_size = len(trades) // 5
        
        volumes = []
        price_changes = []
        
        for i in range(5):
            window = trades[i * window_size:(i + 1) * window_size]
            if not window:
                continue
            
            vol = sum(abs(t.get('amount', t.get('size', 0))) for t in window)
            volumes.append(vol)
            
            if len(window) >= 2:
                p_start = float(window[0].get('price', 0))
                p_end = float(window[-1].get('price', 0))
                change = abs(p_end - p_start) / p_start * 100 if p_start > 0 else 0
                price_changes.append(change)
        
        if not volumes or not price_changes or sum(price_changes) == 0:
            return {'ratio': 1.0, 'suspicious': False}
        
        avg_volume = np.mean(volumes)
        avg_change = np.mean(price_changes)
        
        # Ratio: volume normalisé / changement de prix
        # Plus le ratio est élevé, plus c'est suspect
        ratio = (avg_volume / avg_change) if avg_change > 0.01 else avg_volume * 10
        
        # Normaliser le ratio (1.0 = normal, >2.0 = suspect)
        normalized_ratio = min(10, ratio / 1000)  # Seuil à ajuster selon les données
        
        return {
            'ratio': normalized_ratio,
            'suspicious': normalized_ratio > self.volume_impact_threshold,
            'avg_volume': avg_volume,
            'avg_price_change': avg_change
        }
    
    def _detect_cvd_divergence(self, trades: List[Dict], cvd_data: Dict) -> Dict[str, Any]:
        """
        Détecte divergence entre prix et CVD
        Prix baisse + CVD neutre/positif = accumulation cachée
        """
        if not cvd_data:
            # Calculer CVD depuis les trades
            buy_vol = sum(abs(t.get('amount', 0)) for t in trades 
                         if t.get('side') == 'buy')
            sell_vol = sum(abs(t.get('amount', 0)) for t in trades 
                          if t.get('side') == 'sell')
            net_cvd = buy_vol - sell_vol
            cvd_ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
        else:
            net_cvd = cvd_data.get('net_cvd', 0)
            cvd_ratio = cvd_data.get('aggression_ratio', 1.0)
        
        # Calculer le changement de prix
        if len(trades) >= 2:
            p_start = float(trades[0].get('price', 0))
            p_end = float(trades[-1].get('price', 0))
            price_change_pct = (p_end - p_start) / p_start * 100 if p_start > 0 else 0
        else:
            price_change_pct = 0
        
        # Détecter les divergences
        divergence_type = None
        
        # Prix baisse mais CVD positif = accumulation cachée (bullish)
        if price_change_pct < -self.cvd_divergence_threshold and cvd_ratio > 0.95:
            divergence_type = 'HIDDEN_ACCUMULATION'
        
        # Prix monte mais CVD négatif = distribution cachée (bearish)
        elif price_change_pct > self.cvd_divergence_threshold and cvd_ratio < 1.05:
            divergence_type = 'HIDDEN_DISTRIBUTION'
        
        return {
            'detected': divergence_type is not None,
            'type': divergence_type,
            'price_change_pct': round(price_change_pct, 2),
            'cvd_ratio': round(cvd_ratio, 2)
        }
    
    def _calculate_buy_sell_symmetry(self, trades: List[Dict]) -> float:
        """
        Calcule la symétrie entre achats et ventes
        Symétrie parfaite (ratio ~1.0) = suspect
        """
        buy_vol = sum(abs(t.get('amount', t.get('size', 0))) for t in trades 
                     if t.get('side') == 'buy')
        sell_vol = sum(abs(t.get('amount', t.get('size', 0))) for t in trades 
                      if t.get('side') == 'sell')
        
        if buy_vol == 0 and sell_vol == 0:
            return 0.5
        
        total = buy_vol + sell_vol
        symmetry = 1 - abs(buy_vol - sell_vol) / total if total > 0 else 0.5
        
        return symmetry
    
    def _calculate_probability(self, volume_impact: Dict, 
                               cvd_divergence: Dict, 
                               symmetry: float) -> float:
        """Calcule la probabilité de wash trading"""
        probability = 0
        
        # Volume élevé sans impact (25 points)
        if volume_impact.get('suspicious'):
            probability += 25
        elif volume_impact['ratio'] > 1.5:
            probability += 15
        
        # CVD divergence (35 points)
        if cvd_divergence.get('detected'):
            probability += 35
        
        # Symétrie suspecte (25 points)
        # Symétrie > 0.85 = très symétrique = suspect
        if symmetry > 0.85:
            probability += 25
        elif symmetry > 0.75:
            probability += 15
        
        # Bonus si plusieurs indicateurs (15 points)
        indicators = sum([
            volume_impact.get('suspicious', False),
            cvd_divergence.get('detected', False),
            symmetry > 0.75
        ])
        if indicators >= 2:
            probability += 15
        
        return min(100, probability)
    
    def _determine_type(self, cvd_divergence: Dict, symmetry: float) -> str:
        """Détermine le type de manipulation"""
        if cvd_divergence.get('type') == 'HIDDEN_ACCUMULATION':
            return 'ACCUMULATION'
        elif cvd_divergence.get('type') == 'HIDDEN_DISTRIBUTION':
            return 'DISTRIBUTION'
        elif symmetry > 0.85:
            return 'WASH_TRADING'
        else:
            return 'NEUTRAL'
    
    def _calculate_modifier(self, probability: float, manip_type: str) -> int:
        """Calcule le modificateur de signal"""
        if probability < 30:
            return 0
        
        if manip_type == 'ACCUMULATION':
            # Wash trading pour accumuler = bullish caché
            return 5 if probability > 50 else 2
        elif manip_type == 'DISTRIBUTION':
            # Wash trading pour distribuer = bearish caché
            return -5 if probability > 50 else -2
        elif manip_type == 'WASH_TRADING':
            # Wash trading général = ne pas trader
            return -10 if probability > 70 else -5
        
        return 0
    
    def _interpret(self, probability: float, manip_type: str) -> str:
        """Génère une interprétation humaine"""
        if probability < 30:
            return "Marché sain, pas de manipulation détectée"
        elif probability < 50:
            return f"Légère suspicion de {manip_type.lower()}"
        elif probability < 70:
            return f"Probable {manip_type.lower()} en cours"
        else:
            return f"⚠️ Fort pattern de {manip_type.lower()} - Prudence"
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'detected': False,
            'probability': 0,
            'type': 'NEUTRAL',
            'metrics': {},
            'signal_modifier': 0,
            'interpretation': 'Données insuffisantes'
        }


class VenturiAnalyzer:
    """
    Applique l'effet Venturi au trading
    
    Principe physique:
    - Fluide dans tuyau étroit → accélération
    - Order book fin → breakout imminent
    
    Métriques:
    - Depth Ratio (déséquilibre bid/ask)
    - Spread Compression
    - Imbalance Velocity
    """
    
    def __init__(self):
        self.compression_threshold = 0.5  # Spread < 50% de la moyenne
        self.imbalance_threshold = 2.0    # Ratio > 2x
    
    def analyze(self, order_book: Dict,
                historical_spreads: List[float] = None) -> Dict[str, Any]:
        """
        Analyse l'order book pour détecter l'effet Venturi
        
        Args:
            order_book: Données de l'order book (bids, asks)
            historical_spreads: Historique des spreads pour comparaison
        """
        if not order_book:
            return self._empty_result()
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return self._empty_result()
        
        # Calculer les métriques
        depth_analysis = self._analyze_depth(bids, asks)
        spread_analysis = self._analyze_spread(bids, asks, historical_spreads)
        pressure = self._calculate_pressure(bids, asks)
        
        # Détecter compression (effet Venturi)
        compression = self._detect_compression(depth_analysis, spread_analysis)
        
        # Prédire la direction du breakout
        direction = self._predict_direction(depth_analysis, pressure)
        
        # Calculer la probabilité de breakout
        breakout_prob = self._calculate_breakout_probability(
            compression, depth_analysis, spread_analysis
        )
        
        # Signal modifier
        signal_modifier = self._calculate_modifier(breakout_prob, direction)
        
        return {
            'compression_detected': compression['detected'],
            'compression_score': compression['score'],
            'direction': direction,
            'breakout_probability': round(breakout_prob, 1),
            'metrics': {
                'depth_ratio': round(depth_analysis['ratio'], 2),
                'bid_depth_5': round(depth_analysis['bid_depth'], 2),
                'ask_depth_5': round(depth_analysis['ask_depth'], 2),
                'spread_pct': round(spread_analysis['current_spread_pct'], 4),
                'spread_compression': round(spread_analysis['compression_ratio'], 2),
                'pressure_differential': round(pressure['differential'], 2)
            },
            'signal_modifier': signal_modifier,
            'interpretation': self._interpret(compression, direction, breakout_prob),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_depth(self, bids: List, asks: List) -> Dict[str, Any]:
        """Analyse la profondeur de l'order book"""
        # Profondeur sur les 5 premiers niveaux
        bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
        ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
        
        # Profondeur totale (10 niveaux)
        bid_depth_10 = sum(float(b[1]) for b in bids[:10]) if bids else 0
        ask_depth_10 = sum(float(a[1]) for a in asks[:10]) if asks else 0
        
        # Ratio de déséquilibre
        total = bid_depth + ask_depth
        ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0
        
        # Concentration (profondeur top 5 / top 10)
        bid_concentration = bid_depth / bid_depth_10 if bid_depth_10 > 0 else 0.5
        ask_concentration = ask_depth / ask_depth_10 if ask_depth_10 > 0 else 0.5
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total,
            'ratio': ratio,
            'bid_concentration': bid_concentration,
            'ask_concentration': ask_concentration,
            'imbalanced': ratio > self.imbalance_threshold or ratio < 1/self.imbalance_threshold
        }
    
    def _analyze_spread(self, bids: List, asks: List, 
                        historical: List[float] = None) -> Dict[str, Any]:
        """Analyse le spread et sa compression"""
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        
        current_spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 else 0
        current_spread_pct = (current_spread / mid_price * 100) if mid_price > 0 else 0
        
        # Comparer à l'historique
        if historical and len(historical) > 5:
            avg_spread = np.mean(historical[-20:])
            compression_ratio = current_spread_pct / avg_spread if avg_spread > 0 else 1.0
        else:
            # Estimer basé sur le prix (spread typique ~0.01% pour BTC)
            expected_spread = 0.01
            compression_ratio = current_spread_pct / expected_spread if expected_spread > 0 else 1.0
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'current_spread': current_spread,
            'current_spread_pct': current_spread_pct,
            'compression_ratio': compression_ratio,
            'compressed': compression_ratio < self.compression_threshold
        }
    
    def _calculate_pressure(self, bids: List, asks: List) -> Dict[str, Any]:
        """
        Calcule la pression différentielle (analogie physique)
        
        Pression = Densité × Profondeur
        """
        # Pression bid = volume proche du marché
        bid_pressure = sum(float(b[1]) * (1 / (i + 1)) for i, b in enumerate(bids[:10]))
        
        # Pression ask = volume proche du marché
        ask_pressure = sum(float(a[1]) * (1 / (i + 1)) for i, a in enumerate(asks[:10]))
        
        # Différentiel de pression
        differential = bid_pressure - ask_pressure
        
        return {
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'differential': differential,
            'dominant': 'BID' if differential > 0 else 'ASK'
        }
    
    def _detect_compression(self, depth: Dict, spread: Dict) -> Dict[str, Any]:
        """Détecte si l'effet Venturi est en cours"""
        score = 0
        reasons = []
        
        # Spread compressé
        if spread['compressed']:
            score += 40
            reasons.append("Spread compressé")
        elif spread['compression_ratio'] < 0.7:
            score += 20
            reasons.append("Spread légèrement compressé")
        
        # Déséquilibre de profondeur
        if depth['imbalanced']:
            score += 30
            reasons.append(f"Déséquilibre fort (ratio: {depth['ratio']:.2f})")
        elif depth['ratio'] > 1.5 or depth['ratio'] < 0.67:
            score += 15
            reasons.append("Léger déséquilibre")
        
        # Concentration élevée (murs proches)
        if depth['bid_concentration'] > 0.7 or depth['ask_concentration'] > 0.7:
            score += 20
            reasons.append("Concentration élevée près du marché")
        
        return {
            'detected': score >= 50,
            'score': min(100, score),
            'reasons': reasons
        }
    
    def _predict_direction(self, depth: Dict, pressure: Dict) -> str:
        """Prédit la direction du breakout"""
        # Ratio > 1 = plus de bids = pression haussière
        if depth['ratio'] > 1.3 and pressure['differential'] > 0:
            return 'UP'
        elif depth['ratio'] < 0.77 and pressure['differential'] < 0:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _calculate_breakout_probability(self, compression: Dict, 
                                        depth: Dict, spread: Dict) -> float:
        """Calcule la probabilité de breakout imminent"""
        prob = 0
        
        # Score de compression
        prob += compression['score'] * 0.5
        
        # Déséquilibre fort
        imbalance = abs(depth['ratio'] - 1)
        prob += min(30, imbalance * 30)
        
        # Spread très serré
        if spread['compression_ratio'] < 0.3:
            prob += 20
        
        return min(100, prob)
    
    def _calculate_modifier(self, breakout_prob: float, direction: str) -> int:
        """Calcule le modificateur de signal"""
        if breakout_prob < 40:
            return 0
        
        base = int(breakout_prob / 10)  # 0-10
        
        if direction == 'UP':
            return min(15, base)  # Bonus pour LONG
        elif direction == 'DOWN':
            return max(-15, -base)  # Bonus pour SHORT
        else:
            return 0
    
    def _interpret(self, compression: Dict, direction: str, prob: float) -> str:
        """Génère une interprétation"""
        if not compression['detected']:
            return "Marché fluide, pas de compression"
        
        if prob < 50:
            return f"Légère compression détectée, direction probable: {direction}"
        elif prob < 70:
            return f"⚡ Compression importante - Breakout {direction} probable"
        else:
            return f"🚀 Forte compression - Breakout {direction} imminent!"
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'compression_detected': False,
            'compression_score': 0,
            'direction': 'NEUTRAL',
            'breakout_probability': 0,
            'metrics': {},
            'signal_modifier': 0,
            'interpretation': 'Données insuffisantes'
        }


def test_analyzers():
    """Test des analyseurs"""
    print("=== Test SelfTradingDetector ===")
    detector = SelfTradingDetector()
    
    # Simuler des trades suspects (symétrie élevée)
    trades = []
    for i in range(100):
        side = 'buy' if i % 2 == 0 else 'sell'
        trades.append({
            'price': 90000 + (i % 10),
            'amount': 0.5,
            'side': side
        })
    
    result = detector.analyze(trades, 90005)
    print(f"Detected: {result['detected']}")
    print(f"Probability: {result['probability']}%")
    print(f"Type: {result['type']}")
    print(f"Modifier: {result['signal_modifier']:+d}")
    
    print("\n=== Test VenturiAnalyzer ===")
    venturi = VenturiAnalyzer()
    
    # Simuler un order book déséquilibré
    order_book = {
        'bids': [[90000, 10], [89990, 8], [89980, 5], [89970, 3], [89960, 2]],
        'asks': [[90010, 2], [90020, 1], [90030, 1], [90040, 0.5], [90050, 0.5]]
    }
    
    result = venturi.analyze(order_book)
    print(f"Compression: {result['compression_detected']}")
    print(f"Direction: {result['direction']}")
    print(f"Breakout Prob: {result['breakout_probability']}%")
    print(f"Modifier: {result['signal_modifier']:+d}")


if __name__ == "__main__":
    test_analyzers()
