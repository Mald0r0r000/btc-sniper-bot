"""
Consistency Checker
Vérifie la cohérence des signaux dans le temps pour améliorer la qualité
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta


class ConsistencyChecker:
    """
    Analyse la cohérence temporelle des signaux
    
    Un signal confirmé par plusieurs analyses successives est plus fiable
    qu'un signal qui flip-flop entre LONG et SHORT.
    """
    
    def __init__(self, lookback: int = 5):
        """
        Args:
            lookback: Nombre de signaux précédents à analyser
        """
        self.lookback = lookback
    
    def check_consistency(self, current_signal: Dict[str, Any], 
                         history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse la cohérence du signal actuel avec l'historique
        
        Args:
            current_signal: Signal actuel (direction, confidence, type)
            history: Liste des signaux précédents (plus récent en dernier)
            
        Returns:
            Dict avec:
            - score: -20 à +20 (bonus/malus à appliquer)
            - consistency_level: 0-5 (nombre de confirmations)
            - status: CONFIRMED, FLIP, NEW, NEUTRAL
            - details: Explication
        """
        if not history:
            return {
                "score": 0,
                "consistency_level": 0,
                "status": "NEW",
                "details": "Premier signal - pas d'historique"
            }
        
        current_direction = current_signal.get("direction", "NEUTRAL")
        current_confidence = current_signal.get("confidence", 50)
        
        # Prendre les N derniers signaux
        recent_signals = history[-self.lookback:] if len(history) >= self.lookback else history
        
        # Analyser les directions
        directions = [s.get("signal", {}).get("direction", "NEUTRAL") for s in recent_signals]
        confidences = [s.get("signal", {}).get("confidence", 50) for s in recent_signals]
        
        # Compter les confirmations (même direction que le signal actuel)
        confirmations = sum(1 for d in directions if d == current_direction)
        
        # Compter les flip-flops (changements de direction)
        flips = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1] and directions[i] != "NEUTRAL" and directions[i-1] != "NEUTRAL":
                flips += 1
        
        # Calculer le trend de confiance
        if len(confidences) >= 2:
            confidence_trend = current_confidence - confidences[-1]
        else:
            confidence_trend = 0
        
        # Calculer le score
        score = 0
        status = "NEUTRAL"
        details = []
        
        # Bonus pour confirmations
        if confirmations >= 3 and current_direction != "NEUTRAL":
            score += 15
            status = "STRONG_CONFIRM"
            details.append(f"+15: Signal confirmé {confirmations}x")
        elif confirmations >= 2 and current_direction != "NEUTRAL":
            score += 10
            status = "CONFIRMED"
            details.append(f"+10: Signal confirmé {confirmations}x")
        elif confirmations == 1 and current_direction != "NEUTRAL":
            score += 5
            status = "EMERGING"
            details.append("+5: Signal en émergence")
        
        # Pénalité pour flip-flops
        if flips >= 2:
            score -= 15
            status = "FLIP_FLOP"
            details.append(f"-15: {flips} flip-flops récents")
        elif flips == 1:
            score -= 5
            details.append("-5: 1 flip récent")
        
        # Bonus si confiance croissante
        if confidence_trend >= 5:
            score += 5
            details.append(f"+5: Confiance croissante (+{confidence_trend:.0f})")
        elif confidence_trend <= -5:
            score -= 5
            details.append(f"-5: Confiance décroissante ({confidence_trend:.0f})")
        
        # Pénalité passage à NEUTRAL après direction claire
        if current_direction == "NEUTRAL" and len(directions) > 0:
            last_dir = directions[-1]
            if last_dir in ["LONG", "SHORT"]:
                score -= 5
                details.append("-5: Momentum perdu (→ NEUTRAL)")
        
        # Limiter le score
        score = max(-20, min(20, score))
        
        return {
            "score": score,
            "consistency_level": confirmations,
            "flips_count": flips,
            "confidence_trend": round(confidence_trend, 1),
            "status": status,
            "details": details,
            "analyzed_signals": len(recent_signals)
        }
    
    def get_persistence_score(self, history: List[Dict[str, Any]], 
                              direction: str) -> float:
        """
        Calcule un score de persistance pour une direction donnée
        
        Utile pour savoir si le marché a été consistently LONG ou SHORT récemment.
        """
        if not history:
            return 0.0
        
        recent = history[-self.lookback:]
        matching = sum(1 for s in recent 
                      if s.get("signal", {}).get("direction") == direction)
        
        return matching / len(recent) * 100
    
    def detect_regime_change(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Détecte un changement de régime (ex: passage bear → bull)
        
        Un changement de régime est détecté quand:
        - Les 3+ derniers signaux ont une direction différente des 3+ précédents
        - La confiance du nouveau régime est croissante
        """
        if len(history) < 6:
            return None
        
        recent_3 = history[-3:]
        previous_3 = history[-6:-3]
        
        recent_dirs = [s.get("signal", {}).get("direction") for s in recent_3]
        prev_dirs = [s.get("signal", {}).get("direction") for s in previous_3]
        
        # Vérifier si régime uniforme
        recent_regime = recent_dirs[0] if len(set(recent_dirs)) == 1 else None
        prev_regime = prev_dirs[0] if len(set(prev_dirs)) == 1 else None
        
        if recent_regime and prev_regime and recent_regime != prev_regime:
            # Changement de régime détecté
            recent_conf = sum(s.get("signal", {}).get("confidence", 50) for s in recent_3) / 3
            prev_conf = sum(s.get("signal", {}).get("confidence", 50) for s in previous_3) / 3
            
            return {
                "detected": True,
                "from": prev_regime,
                "to": recent_regime,
                "confidence_change": round(recent_conf - prev_conf, 1),
                "description": f"Changement de régime: {prev_regime} → {recent_regime}"
            }
        
        return None


def test_consistency():
    """Test du consistency checker"""
    checker = ConsistencyChecker()
    
    # Simuler un historique
    history = [
        {"signal": {"direction": "LONG", "confidence": 55}},
        {"signal": {"direction": "LONG", "confidence": 58}},
        {"signal": {"direction": "LONG", "confidence": 60}},
    ]
    
    current = {"direction": "LONG", "confidence": 63}
    
    result = checker.check_consistency(current, history)
    
    print("=== Consistency Check ===")
    print(f"Status: {result['status']}")
    print(f"Score: {result['score']:+d}")
    print(f"Consistency Level: {result['consistency_level']}")
    print(f"Details: {result['details']}")
    
    # Test flip-flop
    print("\n=== Test Flip-Flop ===")
    flip_history = [
        {"signal": {"direction": "LONG", "confidence": 55}},
        {"signal": {"direction": "SHORT", "confidence": 52}},
        {"signal": {"direction": "LONG", "confidence": 53}},
        {"signal": {"direction": "SHORT", "confidence": 51}},
    ]
    
    current_flip = {"direction": "LONG", "confidence": 54}
    result_flip = checker.check_consistency(current_flip, flip_history)
    
    print(f"Status: {result_flip['status']}")
    print(f"Score: {result_flip['score']:+d}")
    print(f"Flips: {result_flip['flips_count']}")


if __name__ == "__main__":
    test_consistency()
