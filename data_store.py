"""
Data Store - Stockage des signaux via GitHub Gist
Permet de conserver l'historique des signaux pour analyse et ML
"""
import os
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


class GistDataStore:
    """
    Stocke les données de signaux dans un GitHub Gist
    
    Avantages:
    - Persistant entre les runs GitHub Actions
    - API simple (REST)
    - Gratuit et illimité
    - Historique versionné
    """
    
    GIST_FILENAME = "btc_signals_history.json"
    
    def __init__(self, gist_id: str = None, github_token: str = None):
        """
        Args:
            gist_id: ID du Gist (créé automatiquement si None)
            github_token: Token GitHub avec scope 'gist'
        """
        self.gist_id = gist_id or os.getenv('GIST_ID')
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.api_base = "https://api.github.com"
        
        if not self.github_token:
            print("⚠️ GITHUB_TOKEN non configuré - stockage désactivé")
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def create_gist(self, description: str = "BTC Sniper Bot - Signal History") -> Optional[str]:
        """Crée un nouveau Gist et retourne son ID"""
        if not self.github_token:
            return None
        
        try:
            response = requests.post(
                f"{self.api_base}/gists",
                headers=self._headers(),
                json={
                    "description": description,
                    "public": False,  # Gist privé
                    "files": {
                        self.GIST_FILENAME: {
                            "content": json.dumps({
                                "created": datetime.now(timezone.utc).isoformat(),
                                "signals": []
                            }, indent=2)
                        }
                    }
                },
                timeout=15
            )
            
            if response.ok:
                gist_data = response.json()
                self.gist_id = gist_data["id"]
                print(f"✅ Gist créé: https://gist.github.com/{self.gist_id}")
                return self.gist_id
            else:
                print(f"❌ Erreur création Gist: {response.status_code}")
                print(response.text[:200])
                return None
                
        except Exception as e:
            print(f"❌ Erreur création Gist: {e}")
            return None
    
    def read_signals(self) -> List[Dict[str, Any]]:
        """Lit tous les signaux stockés"""
        if not self.github_token or not self.gist_id:
            return []
        
        try:
            response = requests.get(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                timeout=15
            )
            
            if response.ok:
                gist_data = response.json()
                content = gist_data["files"][self.GIST_FILENAME]["content"]
                data = json.loads(content)
                return data.get("signals", [])
            else:
                print(f"⚠️ Erreur lecture Gist: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"⚠️ Erreur lecture Gist: {e}")
            return []
    
    def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Ajoute un signal à l'historique
        
        Args:
            signal_data: Données du signal à sauvegarder
        """
        if not self.github_token:
            print("⚠️ Stockage désactivé (pas de token)")
            return False
        
        # Créer le Gist si nécessaire
        if not self.gist_id:
            print("📝 Création du Gist de stockage...")
            if not self.create_gist():
                return False
        
        try:
            # Lire les signaux existants
            signals = self.read_signals()
            
            # Ajouter le nouveau signal avec timestamp
            signal_data["stored_at"] = datetime.now(timezone.utc).isoformat()
            signals.append(signal_data)
            
            # Garder les 1000 derniers signaux (éviter fichier trop gros)
            if len(signals) > 1000:
                signals = signals[-1000:]
            
            # Mettre à jour le Gist
            response = requests.patch(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                json={
                    "files": {
                        self.GIST_FILENAME: {
                            "content": json.dumps({
                                "last_updated": datetime.now(timezone.utc).isoformat(),
                                "total_signals": len(signals),
                                "signals": signals
                            }, indent=2)
                        }
                    }
                },
                timeout=15
            )
            
            if response.ok:
                print(f"💾 Signal sauvegardé ({len(signals)} signals stockés)")
                return True
            else:
                print(f"❌ Erreur sauvegarde: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Récupère les N derniers signaux"""
        signals = self.read_signals()
        return signals[-count:] if signals else []
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """Calcule des statistiques sur les signaux stockés"""
        signals = self.read_signals()
        
        if not signals:
            return {"total": 0}
        
        # Compter par type
        by_type = {}
        by_direction = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        confidences = []
        
        for sig in signals:
            sig_type = sig.get("signal", {}).get("type", "UNKNOWN")
            direction = sig.get("signal", {}).get("direction", "NEUTRAL")
            confidence = sig.get("signal", {}).get("confidence", 0)
            
            by_type[sig_type] = by_type.get(sig_type, 0) + 1
            by_direction[direction] = by_direction.get(direction, 0) + 1
            confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total": len(signals),
            "by_type": by_type,
            "by_direction": by_direction,
            "avg_confidence": round(avg_confidence, 1),
            "high_confidence_count": len([c for c in confidences if c >= 60])
        }


def test_gist_store():
    """Test du stockage Gist"""
    store = GistDataStore()
    
    if not store.github_token:
        print("❌ GITHUB_TOKEN requis pour le test")
        return
    
    # Test création/lecture
    test_signal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": 90000,
        "signal": {
            "type": "TEST",
            "direction": "LONG",
            "confidence": 65
        }
    }
    
    if store.save_signal(test_signal):
        print("✅ Test réussi!")
        stats = store.get_signal_stats()
        print(f"📊 Stats: {stats}")
    else:
        print("❌ Test échoué")


if __name__ == "__main__":
    test_gist_store()
