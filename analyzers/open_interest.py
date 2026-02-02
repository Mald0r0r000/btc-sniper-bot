"""
Open Interest Analyzer
Analyse avancée de l'Open Interest avec détection de changements significatifs
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import deque
import json
import os


class OpenInterestAnalyzer:
    """
    Analyseur d'Open Interest avancé
    
    Features:
    - Multi-exchange aggregation
    - Delta OI (changement)
    - OI vs Price divergence
    - Concentration par exchange
    - Historical tracking
    """
    
    # Fichier pour stocker l'historique
    HISTORY_FILE = "oi_history.json"
    MAX_HISTORY = 96  # 24h si on analyse toutes les 15min
    
    def __init__(self):
        try:
            from data_store import GistDataStore
            self.gist_store = GistDataStore()
        except ImportError:
            self.gist_store = None
            
        self.disable_gist_save = False # Safety flag
        self.history = self._load_history()
    
    def _load_history(self) -> deque:
        """Charge l'historique précédent"""
        # 1. Try Gist first (Source of Truth for persistence)
        # Check if we have a token (configured) before trying to load
        if self.gist_store and self.gist_store.github_token:
            gist_data = self.gist_store.load_oi_history()
            
            if gist_data is None:
                print("🚨 CRITICAL: Failed to load OI History from Gist (Error). Disabling Gist Save to prevent data loss.")
                self.disable_gist_save = True
            else:
                # gist_data is List (empty or not), which means load was successful (or file missing safe)
                self.disable_gist_save = False
                
                if gist_data:
                    # Validation du Gist data (même logique de reset)
                    if gist_data[-1].get('total_oi', 0) > 1000000:
                       print("⚠️ OI History (Gist) corrupted/unnormalized. Resetting.")
                       return deque(maxlen=self.MAX_HISTORY)
                       
                    return deque(gist_data, maxlen=self.MAX_HISTORY)
                else:
                    return deque(maxlen=self.MAX_HISTORY)
        elif self.gist_store:
            # Gist store exists but no token - Silent fallback to local
            self.disable_gist_save = True # Implicitly disabled since no token

        # 2. Fallback to local file
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                
                # Validation: detected massive drop (normalization fix case)
                if data and data[-1].get('total_oi', 0) > 1000000:
                    print("⚠️ OI History (Local) corrupted/unnormalized. Resetting history.")
                    return deque(maxlen=self.MAX_HISTORY)
                    
                return deque(data, maxlen=self.MAX_HISTORY)
        except Exception:
            pass
        return deque(maxlen=self.MAX_HISTORY)
    
    def _save_history(self):
        """Sauvegarde l'historique"""
        history_list = list(self.history)
        
        # 1. Save locally
        try:
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(history_list, f)
        except Exception:
            pass
            
        # 2. Save to Gist
        if self.gist_store:
            if self.disable_gist_save:
                print("⚠️ Skipping Gist Save (Safety Mode Active)")
            else:
                self.gist_store.save_oi_history(history_list)
    
    def analyze(self, current_price: float, 
                open_interests: Dict[str, float],
                save_history: bool = True) -> Dict[str, Any]:
        """
        Analyse complète de l'Open Interest
        
        Args:
            current_price: Prix BTC actuel
            open_interests: {exchange: oi_btc}
            save_history: Sauvegarder pour tracking
            
        Returns:
            Dict avec analyse OI complète
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Agrégation
        total_oi = sum(open_interests.values())
        total_oi_usd = total_oi * current_price
        
        # Concentration par exchange
        concentration = {}
        for ex, oi in open_interests.items():
            if total_oi > 0:
                concentration[ex] = round(oi / total_oi * 100, 2)
            else:
                concentration[ex] = 0
        
        # Leader (exchange avec le plus d'OI)
        leader = max(open_interests.items(), key=lambda x: x[1]) if open_interests else (None, 0)
        
        # Delta OI (comparé à la dernière entrée)
        delta_analysis = self._calculate_delta(total_oi, current_price)
        
        # OI vs Price divergence
        divergence = self._detect_oi_price_divergence()
        
        # Signal basé sur OI
        signal = self._generate_oi_signal(delta_analysis, divergence, total_oi)
        
        # Sauvegarder dans l'historique
        if save_history:
            self.history.append({
                'timestamp': timestamp,
                'total_oi': total_oi,
                'price': current_price,
                'by_exchange': open_interests
            })
            self._save_history()
        
        return {
            'total_oi_btc': round(total_oi, 2),
            'total_oi_usd_b': round(total_oi_usd / 1e9, 3),
            'by_exchange': {ex: round(oi, 2) for ex, oi in open_interests.items()},
            'concentration_pct': concentration,
            'leader': {
                'exchange': leader[0],
                'oi_btc': round(leader[1], 2),
                'share_pct': concentration.get(leader[0], 0) if leader[0] else 0
            },
            'delta': delta_analysis,
            'divergence': divergence,
            'signal': signal,
            'historical_data_points': len(self.history),
            'timestamp': timestamp
        }
    
    def _calculate_delta(self, current_oi: float, current_price: float) -> Dict[str, Any]:
        """Calcule le changement d'OI"""
        if len(self.history) == 0:
            return {
                'available': False,
                'message': 'Pas d\'historique - Première exécution'
            }
        
        # Comparer avec différentes périodes
        last = self.history[-1]
        last_oi = last.get('total_oi', 0)
        last_price = last.get('price', 0)
        
        # Delta OI
        delta_oi = current_oi - last_oi
        delta_oi_pct = (delta_oi / last_oi * 100) if last_oi > 0 else 0
        
        # Delta Price
        delta_price = current_price - last_price
        delta_price_pct = (delta_price / last_price * 100) if last_price > 0 else 0
        
        # Changement sur 1h (4 points à 15min d'intervalle)
        if len(self.history) >= 4:
            h1_ago = self.history[-4]
            delta_1h_oi = current_oi - h1_ago.get('total_oi', current_oi)
            delta_1h_pct = (delta_1h_oi / h1_ago.get('total_oi', 1) * 100)
        else:
            delta_1h_oi = delta_oi
            delta_1h_pct = delta_oi_pct
        
        # Changement sur 4h (16 points)
        if len(self.history) >= 16:
            h4_ago = self.history[-16]
            delta_4h_oi = current_oi - h4_ago.get('total_oi', current_oi)
            delta_4h_pct = (delta_4h_oi / h4_ago.get('total_oi', 1) * 100)
        else:
            delta_4h_oi = delta_1h_oi
            delta_4h_pct = delta_1h_pct
            
        # Changement sur 24h (96 points - max history)
        if len(self.history) >= 96:
            h24_ago = self.history[0] # Oldest point
            delta_24h_oi = current_oi - h24_ago.get('total_oi', current_oi)
            delta_24h_pct = (delta_24h_oi / h24_ago.get('total_oi', 1) * 100)
        elif len(self.history) > 0:
            # Fallback to oldest available if < 24h
            oldest = self.history[0]
            delta_24h_oi = current_oi - oldest.get('total_oi', current_oi)
            delta_24h_pct = (delta_24h_oi / oldest.get('total_oi', 1) * 100)
        else:
            delta_24h_oi = delta_4h_oi
            delta_24h_pct = delta_4h_pct
        
        # Interpretation
        if delta_oi_pct > 2:
            interpretation = "OI en forte hausse - Nouvelles positions ouvertes"
            emoji = "📈"
        elif delta_oi_pct > 0.5:
            interpretation = "OI en légère hausse"
            emoji = "↗️"
        elif delta_oi_pct < -2:
            interpretation = "OI en forte baisse - Liquidations/Clôtures"
            emoji = "📉"
        elif delta_oi_pct < -0.5:
            interpretation = "OI en légère baisse"
            emoji = "↘️"
        else:
            interpretation = "OI stable"
            emoji = "➡️"
        
        return {
            'available': True,
            'last_update': {
                'delta_oi_btc': round(delta_oi, 2),
                'delta_oi_pct': round(delta_oi_pct, 3),
                'delta_price_pct': round(delta_price_pct, 3)
            },
            '1h': {
                'delta_oi_btc': round(delta_1h_oi, 2),
                'delta_oi_pct': round(delta_1h_pct, 3)
            },
            '4h': {
                'delta_oi_btc': round(delta_4h_oi, 2),
                'delta_oi_pct': round(delta_4h_pct, 3)
            },
            '24h': {
                'delta_oi_btc': round(delta_24h_oi, 2),
                'delta_oi_pct': round(delta_24h_pct, 3)
            },
            'interpretation': interpretation,
            'emoji': emoji
        }
    
    def _detect_oi_price_divergence(self) -> Dict[str, Any]:
        """
        Détecte les divergences OI vs Prix
        
        - Prix monte + OI monte = Trend sain (bullish)
        - Prix monte + OI baisse = Weak rally (bearish divergence)
        - Prix baisse + OI monte = Strong selling (bearish)
        - Prix baisse + OI baisse = Capitulation (peut être bullish)
        """
        if len(self.history) < 4:
            return {
                'available': False,
                'message': 'Historique insuffisant'
            }
        
        # Comparer sur la dernière heure
        recent = self.history[-1]
        old = self.history[-4]
        
        oi_change = recent.get('total_oi', 0) - old.get('total_oi', 0)
        price_change = recent.get('price', 0) - old.get('price', 0)
        
        oi_up = oi_change > 0
        price_up = price_change > 0
        
        if price_up and oi_up:
            divergence_type = "HEALTHY_RALLY"
            signal = "BULLISH"
            interpretation = "Prix et OI montent - Trend sain"
            emoji = "🟢"
        elif price_up and not oi_up:
            divergence_type = "WEAK_RALLY"
            signal = "BEARISH_DIVERGENCE"
            interpretation = "Prix monte mais OI baisse - Rally faible"
            emoji = "🟠"
        elif not price_up and oi_up:
            divergence_type = "STRONG_SELLING"
            signal = "BEARISH"
            interpretation = "Prix baisse mais OI monte - Pression vendeuse"
            emoji = "🔴"
        else:  # Prix baisse et OI baisse
            divergence_type = "CAPITULATION"
            signal = "POTENTIALLY_BULLISH"
            interpretation = "Prix et OI baissent - Capitulation possible"
            emoji = "🟡"
        
        return {
            'available': True,
            'type': divergence_type,
            'signal': signal,
            'interpretation': interpretation,
            'emoji': emoji,
            'oi_change_1h': round(oi_change, 2),
            'price_change_1h': round(price_change, 2)
        }
    
    def _generate_oi_signal(self, delta: Dict, divergence: Dict, total_oi: float) -> Dict[str, Any]:
        """Génère un signal basé sur l'analyse OI"""
        score = 50  # Neutre
        factors = []
        
        # Delta OI factor
        if delta.get('available'):
            delta_1h = delta.get('1h', {}).get('delta_oi_pct', 0)
            
            if delta_1h > 3:
                score += 10
                factors.append(f"+10 OI en forte hausse ({delta_1h:.1f}%)")
            elif delta_1h < -3:
                score -= 10
                factors.append(f"-10 OI en forte baisse ({delta_1h:.1f}%)")
        
        # Divergence factor
        if divergence.get('available'):
            div_signal = divergence.get('signal', 'NEUTRAL')
            
            if div_signal == 'BULLISH':
                score += 15
                factors.append("+15 Healthy rally (OI + Prix up)")
            elif div_signal == 'BEARISH_DIVERGENCE':
                score -= 15
                factors.append("-15 Weak rally divergence")
            elif div_signal == 'BEARISH':
                score -= 10
                factors.append("-10 Strong selling pressure")
            elif div_signal == 'POTENTIALLY_BULLISH':
                score += 5
                factors.append("+5 Capitulation possible")
        
        # Normaliser
        score = max(0, min(100, score))
        
        if score >= 65:
            sentiment = "BULLISH"
            emoji = "🟢"
        elif score >= 55:
            sentiment = "SLIGHTLY_BULLISH"
            emoji = "🟢"
        elif score >= 45:
            sentiment = "NEUTRAL"
            emoji = "⚪"
        elif score >= 35:
            sentiment = "SLIGHTLY_BEARISH"
            emoji = "🔴"
        else:
            sentiment = "BEARISH"
            emoji = "🔴"
        
        return {
            'score': score,
            'sentiment': sentiment,
            'emoji': emoji,
            'factors': factors
        }


def test_oi_analyzer():
    """Test de l'analyseur OI"""
    analyzer = OpenInterestAnalyzer()
    
    # Simuler des données
    result = analyzer.analyze(
        current_price=90000,
        open_interests={
            'binance': 45000,
            'okx': 18000,
            'bybit': 15000,
            'bitget': 12000
        }
    )
    
    print("=== OI Analysis ===")
    print(f"Total OI: {result['total_oi_btc']:,.0f} BTC (${result['total_oi_usd_b']:.2f}B)")
    print(f"Leader: {result['leader']['exchange']} ({result['leader']['share_pct']}%)")
    
    delta = result.get('delta', {})
    if delta.get('available'):
        print(f"\nDelta: {delta['emoji']} {delta['interpretation']}")
        print(f"   1h: {delta['1h']['delta_oi_pct']:+.2f}%")
    
    div = result.get('divergence', {})
    if div.get('available'):
        print(f"\nDivergence: {div['emoji']} {div['interpretation']}")
    
    sig = result.get('signal', {})
    print(f"\nSignal: {sig['emoji']} {sig['sentiment']} ({sig['score']}/100)")


if __name__ == "__main__":
    test_oi_analyzer()
