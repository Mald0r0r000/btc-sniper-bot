"""
Data Store - Stockage des signaux via GitHub Gist
Permet de conserver l'historique des signaux pour analyse et ML
"""
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os

# Google Sheets dependencies (lazy import to avoid crash if not installed)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_DEPS_AVAILABLE = True
except ImportError:
    GOOGLE_DEPS_AVAILABLE = False


class GoogleSheetDataStore:
    """
    Stocke les signaux "Blackbox" dans Google Sheets pour analyse immédiate.
    Implémente le schéma "Ultimate" aplati.
    """
    
    # Ultimate Schema Headers
    HEADERS = [
        # --- Core ---
        "Timestamp", "Signal_Type", "Direction", "Confidence", "Price",
        "TP1", "TP2", "SL", "RR_Ratio",
        
        # --- Composite Scores ---
        "Score_Tech", "Score_Struct", "Score_Sent", "Score_OnChain", "Score_Macro", "Score_Deriv",
        
        # --- Deep Alpha ---
        "Quantum_State", "VP_Context", "Risk_Env", "Fear_Greed",
        
        # --- Fluid Dynamics ---
        "Fluid_Venturi_Score", "Fluid_Compression_Detected", "Fluid_Direction", "Fluid_Breakout_Prob",
        "Fluid_ST_Detected", "Fluid_ST_Prob", # ADDED: Self-Trading
        
        # --- Hyperliquid Whales ---
        "HL_Whale_Sentiment", "HL_Long_Ratio", "HL_Whale_Count", "HL_Curated_Count", "HL_Leaderboard_Count", # ADDED: Counts
        "HL_Weighted_Long", "HL_Weighted_Short",
        
        # --- Order Book ---
        "OB_Bid_Ratio", "OB_Pressure", "OB_Imbalance", "OB_Spread_Bps", # ADDED: Spread
        
        # --- CVD Multi-Timeframe ---
        "CVD_Score_Composite", "CVD_Trend", "CVD_Aggression", "CVD_Confluence",
        "CVD_5m_Net", "CVD_5m_Score", "CVD_5m_Aggression",
        "CVD_1h_Net", "CVD_1h_Score",
        "CVD_4h_Net", "CVD_4h_Score",
        "CVD_1d_Net", "CVD_1d_Score", # ADDED: 1D CVD
        
        # --- Technicals ---
        "Tech_KDJ_J", "Tech_KDJ_Signal", # ADDED: Signal
        "Tech_ADX", "Tech_ADX_Trend", 
        "Tech_DI_Plus", "Tech_DI_Minus",
        # --- Institutional (GEX & Liq) ---
        "Inst_GEX_Net_USD_M", "Inst_GEX_Regime", # GEX
        "Inst_Liq_Long_Price", "Inst_Liq_Long_Dist", "Inst_Liq_Long_Int", # Liquidation Long
        "Inst_Liq_Short_Price", "Inst_Liq_Short_Dist", "Inst_Liq_Short_Int", # Liquidation Short
        
        "MACD_3D_Trend", "MACD_3D_Slope", 
        "MACD_1D_Trend", "MACD_1D_Slope",
        "MTF_MACD_Composite", "MTF_Divergence_Type",
        
        # --- Structure & Volume Profile ---
        "Struct_FVG_Dist",
        "VP_POC", "VP_VAH", "VP_VAL", "VP_Regime", # ADDED: VP Levels
        
        # --- OI ---
        "OI_Total", "OI_Delta_1h", "OI_Delta_24h", # ADDED: 24h
        
        # --- Macro Raw ---
        "Macro_DXY", "Macro_SPX", "Macro_M2" # ADDED: Macro Raw
    ]

    def __init__(self, sheet_id: str = None, credentials_json: str = None):
        self.sheet_id = sheet_id or os.getenv('GOOGLE_SHEET_ID')
        # Credentials can be a file path or JSON string in env var
        self.credentials_json = credentials_json or os.getenv('GOOGLE_CREDENTIALS_JSON')
        self.scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        self.client = None
        self.sheet = None
        
        if not GOOGLE_DEPS_AVAILABLE:
            print("⚠️ Google Deps missing (gspread). Google Sheet storage disabled.")
            return

        if self.credentials_json:
            self._authenticate()

    def _authenticate(self):
        try:
            # Check if it's a file path or JSON content
            if os.path.exists(self.credentials_json):
                print(f"🔑 Auth using Service Account File: {self.credentials_json}")
                creds = Credentials.from_service_account_file(self.credentials_json, scopes=self.scope)
            else:
                # Assume it's a JSON string
                print(f"🔑 Auth using JSON String from Env (Length: {len(self.credentials_json)} chars)")
                # Print partial key ID for debugging (safe)
                try:
                    info = json.loads(self.credentials_json)
                    print(f"   Service Account Email: {info.get('client_email', 'UNKNOWN')}")
                    creds = Credentials.from_service_account_info(info, scopes=self.scope)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Decode Error for Credentials: {e}")
                    return
            
            self.client = gspread.authorize(creds)
            print("✅ Google Client Authenticated Successfully")
            
        except Exception as e:
            print(f"❌ Google Auth Error: {e}")

    def _get_sheet(self):
        if not self.client or not self.sheet_id:
            return None
        
        try:
            # Open by Key
            sheet_file = self.client.open_by_key(self.sheet_id)
            # Select first worksheet
            return sheet_file.sheet1
        except Exception as e:
            print(f"❌ Error opening sheet {self.sheet_id}: {repr(e)}")
            # Print traceback to see exactly where gspread failed (HTTP error, etc)
            import traceback
            traceback.print_exc()
            return None

    def _flatten_signal(self, signal_record: Dict[str, Any]) -> List[Any]:
        """Convertit le signal JSON complexe en une liste de valeurs ordonnée"""
        s = signal_record # Raccourci
        
        # Extraction sécurisée avec valeurs par défaut
        sig = s.get("sig", {})
        ds = s.get("ds", {})
        tgt = s.get("tgt", {})
        ctx = s.get("ctx", {})
        fd = s.get("fd", {})
        fd_v = fd.get("v", {})
        hl = s.get("hl", {})
        ob = s.get("ob", {})
        cvd = s.get("cvd", {})
        cvd_mtf = cvd.get("mtf", {})
        tech = s.get("tech", {})
        mtf = s.get("mtf", {})
        oi = s.get("oi", {})
        
        # Calculate RR if possible
        tp1 = tgt.get("tp1", 0)
        sl = tgt.get("sl", 0)
        px = s.get("px", 0)
        rr = 0
        if px > 0 and sl > 0 and tp1 > 0:
            risk = abs(px - sl)
            reward = abs(tp1 - px)
            if risk > 0:
                rr = round(reward / risk, 2)

        return [
            # --- Core ---
            s.get("ts"), sig.get("t"), sig.get("d"), sig.get("c"), s.get("px"),
            tgt.get("tp1"), tgt.get("tp2"), tgt.get("sl"), rr,
            
            # --- Scores ---
            ds.get("technical"), ds.get("structure"), ds.get("sentiment"), 
            ds.get("onchain"), ds.get("macro"), ds.get("derivatives"),
            
            # --- Context ---
            ctx.get("qs"), ctx.get("vpc"), ctx.get("re"), ctx.get("fg"),
            
            # --- Fluid ---
            fd_v.get("cs"), fd_v.get("cd"), fd_v.get("dir"), fd_v.get("bp"),
            fd.get("st", {}).get("det"), fd.get("st", {}).get("pb"), # ADDED: ST
            
            # --- Hyperliquid ---
            hl.get("ws"), hl.get("lr"), hl.get("wc"), hl.get("cc"), hl.get("lc"), # ADDED: Counts
            hl.get("wl"), hl.get("wsh"),
            
            # --- OB ---
            ob.get("br"), ob.get("pr"), ob.get("im"), ob.get("sp"), # ADDED: Spread
            
            # --- CVD ---
            cvd.get("cs"), cvd.get("tr"), cvd.get("ag"), cvd.get("cf"),
            cvd_mtf.get("5m", {}).get("nc"), cvd_mtf.get("5m", {}).get("sc"), cvd_mtf.get("5m", {}).get("ar"),
            cvd_mtf.get("1h", {}).get("nc"), cvd_mtf.get("1h", {}).get("sc"),
            cvd_mtf.get("4h", {}).get("nc"), cvd_mtf.get("4h", {}).get("sc"),
            cvd_mtf.get("1d", {}).get("nc"), cvd_mtf.get("1d", {}).get("sc"), # ADDED: 1D
            
            # --- Tech ---
            tech.get("kj"), tech.get("ks"), # ADDED: Signal
            tech.get("adx"), tech.get("atd"), 
            tech.get("dip"), tech.get("dim"),
            
            # --- Institutional ---
            s.get("gex", {}).get("net_gex_usd_m"), s.get("gex", {}).get("regime"),
            s.get("liq", {}).get("nearest_long_liq", {}).get("price"), s.get("liq", {}).get("nearest_long_liq", {}).get("distance_pct"), s.get("liq", {}).get("nearest_long_liq", {}).get("intensity"),
            s.get("liq", {}).get("nearest_short_liq", {}).get("price"), s.get("liq", {}).get("nearest_short_liq", {}).get("distance_pct"), s.get("liq", {}).get("nearest_short_liq", {}).get("intensity"),
            
            # MACD Precision
            mtf.get("tf", {}).get("3d", {}).get("t"), mtf.get("tf", {}).get("3d", {}).get("sl"), 
            mtf.get("tf", {}).get("1d", {}).get("t"), mtf.get("tf", {}).get("1d", {}).get("sl"), 
            
            mtf.get("cs"), (mtf.get("dv") or {}).get("t", "NONE"),
            
            # --- Structure ---
            s.get("str", {}).get("fvg_d"),
            s.get("vp", {}).get("poc"), s.get("vp", {}).get("vah"), s.get("vp", {}).get("val"), s.get("vp", {}).get("reg"), # ADDED: VP
            
            # --- OI ---
            oi.get("t"), oi.get("d1h"), oi.get("d24h"), # ADDED: 24h
            
            # --- Macro Raw ---
            s.get("macro", {}).get("dxy"), s.get("macro", {}).get("spx"), s.get("macro", {}).get("m2", {}).get("v") # ADDED: Macro Raw
        ]

    def save_signal(self, signal_record: Dict[str, Any]) -> bool:
        """Publie le signal aplati sur Google Sheet"""
        if not self.client:
            return False
            
        try:
            print(f"🔍 Accessing Sheet ID: {self.sheet_id[:5]}...{self.sheet_id[-5:]}")
            sheet = self._get_sheet()
            if not sheet:
                print("❌ Failed to get sheet object.")
                return False
                
            # Vérifier headers (si feuille vide)
            if sheet.row_count == 0 or not sheet.row_values(1):
                print("📝 Initializing Sheet Headers (Sheet was empty)...")
                sheet.append_row(self.HEADERS)
            
            # Aplatir et ajouter
            row_data = self._flatten_signal(signal_record)
            
            # Convertir en types compatibles JSON (str pour dates, float pour nombres)
            clean_row = []
            for item in row_data:
                if item is None:
                    clean_row.append("")
                else:
                    clean_row.append(str(item)) # Convert everything to string for safety initially
            
            print(f"📤 Appending row with {len(clean_row)} columns...")
            sheet.append_row(clean_row)
            print(f"📊 Signal SUCCESSFULLY recorded in Google Sheet (Row {len(sheet.col_values(1))})")
            return True
            
        except Exception as e:
            print(f"❌ Google Sheet Save Error: {e}")
            import traceback
            traceback.print_exc()
            return False



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
    
    def read_signals(self) -> Optional[List[Dict[str, Any]]]:
        """
        Lit tous les signaux stockés
        Returns:
            List[Dict]: Liste des signaux si succès
            []: Si fichier vide ou Gist nouveau
            None: Si ERREUR de lecture (pour éviter d'écraser)
        """
        if not self.github_token or not self.gist_id:
            return []
        
        try:
            response = requests.get(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                timeout=30  # Increased timeout
            )
            
            if response.ok:
                gist_data = response.json()
                files = gist_data.get("files", {})
                
                if self.GIST_FILENAME not in files:
                    print(f"⚠️ Fichier {self.GIST_FILENAME} absent du Gist")
                    return []
                    
                content = files[self.GIST_FILENAME]["content"]
                if not content:
                    return []
                    
                data = json.loads(content)
                return data.get("signals", [])
            else:
                print(f"⚠️ Erreur lecture Gist: {response.status_code} - {response.text[:100]}")
                return None # Return None to indicate failure
                
        except Exception as e:
            print(f"⚠️ Erreur lecture Gist: {e}")
            return None # Return None to indicate failure
    
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
            
            # SECURITE CRITIQUE: Si la lecture a échoué (None), ON NE SAUVEGARDE PAS
            # Cela évite d'écraser tout l'historique avec une liste vide
            if signals is None:
                print("❌ ABORT: Impossible de lire l'historique. Sauvegarde annulée pour éviter perte de données.")
                return False
                
            # Ajouter le nouveau signal avec timestamp
            signal_data["stored_at"] = datetime.now(timezone.utc).isoformat()
            signals.append(signal_data)
            
            # Garder les 1000 derniers signaux (éviter fichier trop gros)
            if len(signals) > 1000:
                signals = signals[-1000:]
            
            # Mettre à jour le Gist (JSON compact pour économiser l'espace)
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
                            }, separators=(',', ':'))  # Compact JSON
                        }
                    }
                },
                timeout=30
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
    
    # ========== ADAPTIVE WEIGHTS PERSISTENCE ==========
    
    WEIGHTS_FILENAME = "adaptive_weights.json"
    
    def save_adaptive_weights(self, weights_data: Dict[str, Any]) -> bool:
        """
        Sauvegarde les poids adaptatifs dans le Gist
        Permet la persistance entre les runs GitHub Actions
        """
        if not self.github_token or not self.gist_id:
            return False
        
        try:
            # Lire le gist actuel pour obtenir tous les fichiers
            response = requests.get(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                timeout=15
            )
            
            if not response.ok:
                print(f"⚠️ Erreur lecture Gist: {response.status_code}")
                return False
            
            # Mettre à jour avec le nouveau fichier de poids
            update_response = requests.patch(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                json={
                    "files": {
                        self.WEIGHTS_FILENAME: {
                            "content": json.dumps(weights_data, indent=2)
                        }
                    }
                },
                timeout=15
            )
            
            if update_response.ok:
                print(f"   ✅ Poids adaptatifs sauvegardés dans le Gist")
                return True
            else:
                print(f"⚠️ Erreur sauvegarde poids: {update_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur sauvegarde poids adaptatifs: {e}")
            return False
    
    def load_adaptive_weights(self) -> Optional[Dict[str, Any]]:
        """
        Charge les poids adaptatifs depuis le Gist
        Utilisé au démarrage pour récupérer les poids persistés
        """
        if not self.github_token or not self.gist_id:
            return None
        
        try:
            response = requests.get(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                timeout=15
            )
            
            if response.ok:
                gist_data = response.json()
                files = gist_data.get("files", {})
                
                if self.WEIGHTS_FILENAME in files:
                    content = files[self.WEIGHTS_FILENAME]["content"]
                    weights_data = json.loads(content)
                    print(f"   🧠 Poids adaptatifs chargés depuis le Gist")
                    return weights_data
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Erreur chargement poids adaptatifs: {e}")
            return None
    
    def get_performance_stats(self) -> Optional[Dict[str, Any]]:
        """
        Récupère les statistiques de performance (winrate) depuis le Gist
        Utilisé pour afficher le winrate dans les notifications
        """
        weights_data = self.load_adaptive_weights()
        if weights_data and 'performance' in weights_data:
            return weights_data['performance']
        return None

    # ========== OI HISTORY PERSISTENCE ==========
    
    OI_HISTORY_FILENAME = "oi_history.json"
    
    def save_oi_history(self, history_data: List[Dict[str, Any]]) -> bool:
        """
        Sauvegarde l'historique OI dans le Gist
        """
        if not self.github_token or not self.gist_id:
            return False
        
        try:
            # Compact JSON for efficiency
            content = json.dumps(history_data, separators=(',', ':'))
            
            response = requests.patch(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                json={
                    "files": {
                        self.OI_HISTORY_FILENAME: {
                            "content": content
                        }
                    }
                },
                timeout=15
            )
            
            if response.ok:
                print(f"   💾 Historique OI sauvegardé ({len(history_data)} points)")
                return True
            else:
                print(f"⚠️ Erreur sauvegarde OI: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur sauvegarde OI: {e}")
            return False

    def load_oi_history(self) -> Optional[List[Dict[str, Any]]]:
        """
        Charge l'historique OI depuis le Gist
        """
        if not self.github_token or not self.gist_id:
            return None
        
        try:
            response = requests.get(
                f"{self.api_base}/gists/{self.gist_id}",
                headers=self._headers(),
                timeout=15
            )
            
            if response.ok:
                gist_data = response.json()
                files = gist_data.get("files", {})
                
                if self.OI_HISTORY_FILENAME in files:
                    content = files[self.OI_HISTORY_FILENAME]["content"]
                    history_data = json.loads(content)
                    print(f"   📈 Historique OI chargé depuis Gist ({len(history_data)} points)")
                    return history_data
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Erreur chargement OI: {e}")
            return None


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
