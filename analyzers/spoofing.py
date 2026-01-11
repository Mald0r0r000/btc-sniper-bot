"""
Spoofing & Manipulation Detection
Détecte les manipulations de marché : ghost walls, layering, wash trading
"""
import time
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np

import config


@dataclass
class OrderBookSnapshot:
    """Snapshot d'un order book à un instant T"""
    timestamp: float
    bids: List[List[float]]  # [[price, volume], ...]
    asks: List[List[float]]
    
    def get_walls(self, min_value_usd: float = 500_000) -> Dict[str, List[Dict]]:
        """Trouve les murs significatifs"""
        bid_walls = []
        ask_walls = []
        
        for bid in self.bids:
            value = bid[0] * bid[1]
            if value >= min_value_usd:
                bid_walls.append({'price': bid[0], 'volume': bid[1], 'value_usd': value})
        
        for ask in self.asks:
            value = ask[0] * ask[1]
            if value >= min_value_usd:
                ask_walls.append({'price': ask[0], 'volume': ask[1], 'value_usd': value})
        
        return {'bids': bid_walls, 'asks': ask_walls}


@dataclass
class TradeSnapshot:
    """Snapshot de trades récents"""
    timestamp: float
    trades: List[Dict]  # [{'price', 'volume', 'side', 'time'}, ...]


class SpoofingDetector:
    """
    Détecteur de spoofing et manipulation de marché
    
    Techniques détectées:
    1. Ghost Walls - Murs qui apparaissent/disparaissent rapidement
    2. Layering - Ordres étagés artificiels
    3. Wash Trading - Auto-trading pour gonfler le volume
    4. Pump Coordination - Volume spike suspect
    """
    
    def __init__(self, history_size: int = 30):
        """
        Args:
            history_size: Nombre de snapshots à conserver (default: 30 = 2.5 min à 5s interval)
        """
        self.history_size = history_size
        self.ob_history: deque = deque(maxlen=history_size)
        self.trade_history: deque = deque(maxlen=history_size)
        self.wall_tracker: Dict[float, Dict] = {}  # {price: {first_seen, last_seen, appearances}}
        
        # Seuils configurables
        self.WALL_MIN_VALUE_USD = 500_000  # 500K minimum pour être un "mur"
        self.GHOST_WALL_MAX_LIFETIME_SEC = 30  # Mur qui disparaît en < 30s = suspect
        self.GHOST_WALL_MIN_APPEARANCES = 2  # Doit apparaître/disparaître au moins 2 fois
        self.LAYERING_PRICE_GAP_PCT = 0.05  # Ordres espacés de ~0.05%
        self.LAYERING_VOLUME_SIMILARITY = 0.15  # Volumes similaires à 15% près
        self.WASH_TRADE_TIME_WINDOW_MS = 500  # Trades opposés en < 500ms
        self.WASH_TRADE_SIZE_TOLERANCE = 0.02  # Même taille à 2% près
    
    def add_orderbook_snapshot(self, bids: List, asks: List):
        """Ajoute un snapshot d'order book à l'historique"""
        snapshot = OrderBookSnapshot(
            timestamp=time.time(),
            bids=bids[:30],  # Top 30
            asks=asks[:30]
        )
        self.ob_history.append(snapshot)
        self._update_wall_tracker(snapshot)
    
    def add_trades_snapshot(self, trades: List[Dict]):
        """Ajoute un snapshot de trades à l'historique"""
        snapshot = TradeSnapshot(
            timestamp=time.time(),
            trades=trades[-100:]  # Derniers 100 trades
        )
        self.trade_history.append(snapshot)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète de manipulation
        
        Returns:
            Dict avec scores de fiabilité et alertes
        """
        ghost_walls = self._detect_ghost_walls()
        layering = self._detect_layering()
        wash_trading = self._detect_wash_trading()
        
        # Score global de manipulation (0-1, 0 = pas de manipulation)
        manipulation_score = self._calculate_manipulation_score(ghost_walls, layering, wash_trading)
        
        # Risk level
        if manipulation_score > 0.7:
            risk_level = "HIGH"
            risk_emoji = "🔴"
        elif manipulation_score > 0.4:
            risk_level = "MEDIUM"
            risk_emoji = "🟡"
        else:
            risk_level = "LOW"
            risk_emoji = "🟢"
        
        return {
            'manipulation_score': round(manipulation_score, 3),
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'ghost_walls': ghost_walls,
            'layering': layering,
            'wash_trading': wash_trading,
            'wall_reliability': self._calculate_wall_reliability(),
            'recommendations': self._generate_recommendations(ghost_walls, layering, wash_trading),
            'snapshots_analyzed': len(self.ob_history)
        }
    
    def _update_wall_tracker(self, snapshot: OrderBookSnapshot):
        """Met à jour le tracking des murs"""
        current_time = snapshot.timestamp
        current_walls = snapshot.get_walls(self.WALL_MIN_VALUE_USD)
        
        # Marquer tous les murs existants comme non-vus dans ce snapshot
        for price in self.wall_tracker:
            self.wall_tracker[price]['seen_this_snapshot'] = False
        
        # Tracker les murs actuels
        for wall_type in ['bids', 'asks']:
            for wall in current_walls[wall_type]:
                price = round(wall['price'], 1)  # Arrondir pour grouper
                
                if price in self.wall_tracker:
                    # Mur existant
                    self.wall_tracker[price]['last_seen'] = current_time
                    self.wall_tracker[price]['appearances'] += 1
                    self.wall_tracker[price]['seen_this_snapshot'] = True
                    self.wall_tracker[price]['volumes'].append(wall['volume'])
                else:
                    # Nouveau mur
                    self.wall_tracker[price] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'appearances': 1,
                        'disappearances': 0,
                        'type': wall_type,
                        'seen_this_snapshot': True,
                        'volumes': [wall['volume']],
                        'is_ghost': False
                    }
        
        # Détecter les disparitions
        for price, data in self.wall_tracker.items():
            if not data['seen_this_snapshot'] and data['last_seen'] == current_time:
                pass  # Attendu - on vient de le mettre à False
            elif not data['seen_this_snapshot']:
                # Le mur a disparu
                lifetime = current_time - data['first_seen']
                if lifetime < self.GHOST_WALL_MAX_LIFETIME_SEC:
                    data['disappearances'] += 1
                    if data['disappearances'] >= self.GHOST_WALL_MIN_APPEARANCES:
                        data['is_ghost'] = True
        
        # Nettoyer les vieux murs (> 5 min sans apparition)
        stale_prices = [p for p, d in self.wall_tracker.items() 
                       if current_time - d['last_seen'] > 300]
        for price in stale_prices:
            del self.wall_tracker[price]
    
    def _detect_ghost_walls(self) -> Dict[str, Any]:
        """Détecte les murs fantômes (apparaissent/disparaissent rapidement)"""
        ghost_bids = []
        ghost_asks = []
        
        for price, data in self.wall_tracker.items():
            if data['is_ghost']:
                ghost_info = {
                    'price': price,
                    'appearances': data['appearances'],
                    'disappearances': data['disappearances'],
                    'avg_volume': np.mean(data['volumes']) if data['volumes'] else 0
                }
                
                if data['type'] == 'bids':
                    ghost_bids.append(ghost_info)
                else:
                    ghost_asks.append(ghost_info)
        
        total_ghosts = len(ghost_bids) + len(ghost_asks)
        
        return {
            'detected': total_ghosts > 0,
            'count': total_ghosts,
            'ghost_bids': ghost_bids,
            'ghost_asks': ghost_asks,
            'severity': 'HIGH' if total_ghosts > 3 else 'MEDIUM' if total_ghosts > 0 else 'NONE'
        }
    
    def _detect_layering(self) -> Dict[str, Any]:
        """Détecte le layering (ordres étagés régulièrement)"""
        if not self.ob_history:
            return {'detected': False, 'patterns': []}
        
        latest = self.ob_history[-1]
        patterns = []
        
        for side, orders in [('bids', latest.bids), ('asks', latest.asks)]:
            if len(orders) < 5:
                continue
            
            # Analyser les gaps entre prix
            prices = [o[0] for o in orders[:20]]
            volumes = [o[1] for o in orders[:20]]
            
            gaps = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
            
            if not gaps:
                continue
            
            # Vérifier régularité des gaps
            gap_mean = np.mean([abs(g) for g in gaps])
            gap_std = np.std([abs(g) for g in gaps])
            gap_cv = gap_std / gap_mean if gap_mean > 0 else 1  # Coefficient de variation
            
            # Vérifier similarité des volumes
            vol_mean = np.mean(volumes)
            vol_std = np.std(volumes)
            vol_cv = vol_std / vol_mean if vol_mean > 0 else 1
            
            # Pattern suspect si gaps ET volumes très réguliers
            is_suspicious = gap_cv < 0.3 and vol_cv < self.LAYERING_VOLUME_SIMILARITY
            
            if is_suspicious:
                patterns.append({
                    'side': side,
                    'gap_regularity': round(1 - gap_cv, 2),
                    'volume_regularity': round(1 - vol_cv, 2),
                    'affected_levels': len(prices)
                })
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns,
            'severity': 'HIGH' if len(patterns) > 1 else 'MEDIUM' if patterns else 'NONE'
        }
    
    def _detect_wash_trading(self) -> Dict[str, Any]:
        """Détecte le wash trading (auto-trading artificiel)"""
        if not self.trade_history:
            return {'detected': False, 'probability': 0, 'suspicious_trades': 0}
        
        latest = self.trade_history[-1]
        trades = latest.trades
        
        suspicious_count = 0
        total_analyzed = 0
        
        for i, trade in enumerate(trades[:-1]):
            for j in range(i+1, min(i+10, len(trades))):
                other = trades[j]
                
                # Vérifier si trades opposés
                if trade.get('side') == other.get('side'):
                    continue
                
                # Vérifier timing
                t1 = trade.get('timestamp', 0)
                t2 = other.get('timestamp', 0)
                time_diff = abs(t2 - t1)
                
                if time_diff > self.WASH_TRADE_TIME_WINDOW_MS:
                    continue
                
                # Vérifier taille similaire
                size1 = trade.get('amount', 0)
                size2 = other.get('amount', 0)
                
                if size1 == 0 or size2 == 0:
                    continue
                
                size_diff = abs(size1 - size2) / max(size1, size2)
                
                if size_diff < self.WASH_TRADE_SIZE_TOLERANCE:
                    suspicious_count += 1
                
                total_analyzed += 1
        
        probability = suspicious_count / max(total_analyzed, 1)
        
        return {
            'detected': suspicious_count > 5,
            'probability': round(probability, 3),
            'suspicious_trades': suspicious_count,
            'total_analyzed': total_analyzed,
            'severity': 'HIGH' if probability > 0.2 else 'MEDIUM' if probability > 0.05 else 'NONE'
        }
    
    def _calculate_manipulation_score(self, ghost: Dict, layering: Dict, wash: Dict) -> float:
        """Calcule un score global de manipulation (0-1)"""
        score = 0.0
        
        # Ghost walls contribution (0-0.4)
        if ghost['severity'] == 'HIGH':
            score += 0.4
        elif ghost['severity'] == 'MEDIUM':
            score += 0.2
        
        # Layering contribution (0-0.3)
        if layering['severity'] == 'HIGH':
            score += 0.3
        elif layering['severity'] == 'MEDIUM':
            score += 0.15
        
        # Wash trading contribution (0-0.3)
        score += min(wash['probability'], 0.3)
        
        return min(score, 1.0)
    
    def _calculate_wall_reliability(self) -> Dict[str, float]:
        """Calcule la fiabilité des murs actuels"""
        bid_reliability = 1.0
        ask_reliability = 1.0
        
        # Réduire la fiabilité s'il y a des ghost walls
        for price, data in self.wall_tracker.items():
            if data['is_ghost']:
                if data['type'] == 'bids':
                    bid_reliability -= 0.15
                else:
                    ask_reliability -= 0.15
        
        return {
            'bid_walls': max(round(bid_reliability, 2), 0),
            'ask_walls': max(round(ask_reliability, 2), 0)
        }
    
    def _generate_recommendations(self, ghost: Dict, layering: Dict, wash: Dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        if ghost['detected']:
            if ghost['ghost_asks']:
                recommendations.append(f"⚠️ {len(ghost['ghost_asks'])} mur(s) vendeur suspect(s) - Ne pas se fier aux résistances")
            if ghost['ghost_bids']:
                recommendations.append(f"⚠️ {len(ghost['ghost_bids'])} mur(s) acheteur suspect(s) - Ne pas se fier aux supports")
        
        if layering['detected']:
            for pattern in layering['patterns']:
                recommendations.append(f"⚠️ Layering détecté côté {pattern['side']} - Ordres probablement artificiels")
        
        if wash['detected']:
            recommendations.append(f"⚠️ Wash trading probable ({wash['probability']*100:.1f}%) - Volume potentiellement gonflé")
        
        if not recommendations:
            recommendations.append("✅ Aucune manipulation significative détectée")
        
        return recommendations


class RealTimeSpoofingMonitor:
    """
    Moniteur en temps réel qui accumule les données pour la détection
    Utilisé pour le monitoring continu
    """
    
    def __init__(self):
        self.detector = SpoofingDetector(history_size=60)  # 5 min d'historique
        self.last_analysis = None
        self.analysis_interval = 10  # Analyser toutes les 10s
        self.last_analysis_time = 0
    
    def update(self, order_book: Dict, trades: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Met à jour avec de nouvelles données et analyse si nécessaire
        
        Args:
            order_book: {'bids': [...], 'asks': [...]}
            trades: Liste des trades récents
            
        Returns:
            Résultat d'analyse si intervalle atteint, sinon None
        """
        self.detector.add_orderbook_snapshot(
            order_book.get('bids', []),
            order_book.get('asks', [])
        )
        self.detector.add_trades_snapshot(trades)
        
        current_time = time.time()
        
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.last_analysis = self.detector.analyze()
            self.last_analysis_time = current_time
            return self.last_analysis
        
        return None
    
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Retourne la dernière analyse effectuée"""
        return self.last_analysis
