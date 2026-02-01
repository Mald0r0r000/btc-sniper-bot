"""
Deribit Options Analyzer
Récupère les vraies données d'options BTC depuis l'API Deribit (gratuit)
Max Pain, Put/Call Ratio, IV, Greeks, Open Interest par strike
"""
import asyncio
import httpx
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# === GEX CONFIGURATION ===
class GEXConfig:
    """
    Configuration pour le calcul GEX optimisé
    Basé sur les best practices de gex-tool_v3
    """
    def __init__(
        self,
        dte_min: int = 0,
        dte_max: int = 65,
        weight_quarterly: float = 3.0,
        weight_monthly: float = 2.0,
        weight_weekly: float = 1.0,
        fridays_only: bool = True,
        min_oi: float = 1.0
    ):
        self.dte_min = dte_min
        self.dte_max = dte_max
        self.weight_quarterly = weight_quarterly
        self.weight_monthly = weight_monthly
        self.weight_weekly = weight_weekly
        self.fridays_only = fridays_only
        self.min_oi = min_oi


# === HELPER FUNCTIONS ===
def calculate_dte(expiry_timestamp: int) -> float:
    """
    Calcule Days To Expiry en jours
    
    Args:
        expiry_timestamp: Timestamp d'expiration en milliseconds
    
    Returns:
        Nombre de jours jusqu'à l'expiration
    """
    now = datetime.now(timezone.utc).timestamp() * 1000
    dte_days = (expiry_timestamp - now) / (1000 * 86400)
    return dte_days


def is_last_friday_of_month(date: datetime) -> bool:
    """
    Détecte si une date est le dernier vendredi du mois
    Utilisé pour identifier les monthly expirations
    
    Args:
        date: Date à vérifier
    
    Returns:
        True si c'est le dernier vendredi
    """
    from datetime import timedelta
    
    # Calculer le dernier jour du mois
    if date.month == 12:
        next_month = date.replace(year=date.year + 1, month=1, day=1)
    else:
        next_month = date.replace(month=date.month + 1, day=1)
    
    last_day = next_month - timedelta(days=1)
    
    # Trouver le dernier vendredi
    days_to_friday = (last_day.weekday() - 4) % 7
    last_friday = last_day - timedelta(days=days_to_friday)
    
    return date.date() == last_friday.date()


def detect_expiration_type(instrument_name: str, expiry_timestamp: int) -> Tuple[bool, bool, int]:
    """
    Détecte le type d'expiration (quarterly, monthly, weekly)
    
    Args:
        instrument_name: Nom de l'instrument (ex: BTC-27DEC24-100000-C)
        expiry_timestamp: Timestamp d'expiration
    
    Returns:
        (is_quarterly, is_monthly, weekday)
    """
    try:
        # Parser la date d'expiration
        parts = instrument_name.split("-")
        if len(parts) < 2:
            return False, False, -1
        
        date_str = parts[1]
        expiry_date = datetime.strptime(date_str, "%d%b%y")
        
        weekday = expiry_date.weekday()  # 0=Lundi, 4=Vendredi
        month = expiry_date.month
        
        # Vérifier si c'est un vendredi
        is_friday = (weekday == 4)
        
        # Vérifier si c'est le dernier vendredi du mois
        is_monthly = is_friday and is_last_friday_of_month(expiry_date)
        
        # Quarterly = dernier vendredi des mois Q (Mars, Juin, Sept, Déc)
        is_quarterly = is_monthly and (month in [3, 6, 9, 12])
        
        return is_quarterly, is_monthly, weekday
        
    except Exception as e:
        logger.debug(f"Erreur parsing expiration type pour {instrument_name}: {e}")
        return False, False, -1


class DeribitOptionsAnalyzer:
    """
    Analyseur d'options BTC via API Deribit (gratuit, pas de clé requise)
    
    Données récupérées:
    - Max Pain (strike où le plus d'options expirent sans valeur)
    - Put/Call Ratio (sentiment des traders)
    - Implied Volatility (IV) moyenne
    - Open Interest par strike
    - Greeks globaux (Delta, Gamma)
    """
    
    # API Deribit (production, données réelles)
    API_BASE = "https://www.deribit.com/api/v2/public"
    
    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http
    
    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
    
    async def analyze(self, current_price: float = None, gex_config: Optional[GEXConfig] = None) -> Dict[str, Any]:
        """
        Analyse complète du marché des options BTC
        
        Args:
            current_price: Prix BTC actuel (optionnel, sera fetché si non fourni)
            gex_config: Configuration pour le calcul GEX (optionnel)
            
        Returns:
            Dict avec max pain, P/C ratio, IV, OI par strike, etc.
        """
        try:
            client = await self._get_client()
            
            # 1. Récupérer le prix index si pas fourni
            if current_price is None:
                current_price = await self._get_btc_index_price(client)
            
            # 2. Récupérer les instruments d'options
            instruments = await self._get_option_instruments(client)
            
            if not instruments:
                return {"error": "No option instruments found"}
            
            # 3. Récupérer les données de book pour chaque instrument (top 300 par OI pour meilleure couverture GEX)
            options_data = await self._fetch_options_data(client, instruments[:300])
            
            # 4. Calculer les métriques
            max_pain = self._calculate_max_pain(options_data, current_price)
            pc_ratio = self._calculate_put_call_ratio(options_data)
            iv_analysis = self._analyze_iv(options_data)
            oi_by_strike = self._aggregate_oi_by_strike(options_data)
            nearest_expiry = self._get_nearest_expiry_analysis(options_data, current_price)
            
            # 5. Calculer GEX (Gamma Exposure)
            gex_profile = self._calculate_gex_profile(options_data, current_price, config=gex_config)
            
            # 6. Score global
            options_score = self._calculate_options_score(
                max_pain, pc_ratio, iv_analysis, gex_profile, current_price
            )
            
            return {
                "current_price": current_price,
                "max_pain": max_pain,
                "put_call_ratio": pc_ratio,
                "iv_analysis": iv_analysis,
                "gex_profile": gex_profile,  # ADDED: GEX
                "open_interest": oi_by_strike,
                "nearest_expiry": nearest_expiry,
                "score": options_score,
                "instruments_analyzed": len(options_data),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur Deribit options: {e}")
            return {"error": str(e)}
    
    async def _get_btc_index_price(self, client: httpx.AsyncClient) -> float:
        """Récupère le prix index BTC depuis Deribit"""
        try:
            response = await client.get(
                f"{self.API_BASE}/get_index_price",
                params={"index_name": "btc_usd"}
            )
            data = response.json()
            return data.get("result", {}).get("index_price", 0)
        except Exception as e:
            logger.error(f"Erreur get index price: {e}")
            return 0
    
    async def _get_option_instruments(self, client: httpx.AsyncClient) -> List[Dict]:
        """Récupère la liste des instruments d'options BTC actifs"""
        try:
            response = await client.get(
                f"{self.API_BASE}/get_instruments",
                params={
                    "currency": "BTC",
                    "kind": "option",
                    "expired": "false"
                }
            )
            data = response.json()
            instruments = data.get("result", [])
            
            # Trier par open interest décroissant
            instruments.sort(key=lambda x: x.get("open_interest", 0), reverse=True)
            
            return instruments
            
        except Exception as e:
            logger.error(f"Erreur get instruments: {e}")
            return []
    
    async def _fetch_options_data(self, client: httpx.AsyncClient, 
                                   instruments: List[Dict]) -> List[Dict]:
        """Récupère les données détaillées pour chaque option"""
        options_data = []
        
        # Batching pour éviter rate limit
        batch_size = 10
        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i+batch_size]
            
            tasks = []
            for inst in batch:
                tasks.append(self._fetch_single_option(client, inst))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and "error" not in result:
                    options_data.append(result)
            
            # Petit délai entre batches
            if i + batch_size < len(instruments):
                await asyncio.sleep(0.1)
        
        return options_data
    
    async def _fetch_single_option(self, client: httpx.AsyncClient, 
                                    instrument: Dict) -> Dict:
        """Récupère les données d'une option spécifique"""
        try:
            instrument_name = instrument.get("instrument_name")
            
            response = await client.get(
                f"{self.API_BASE}/ticker",
                params={"instrument_name": instrument_name}
            )
            data = response.json()
            ticker = data.get("result", {})
            
            # Parser le nom de l'instrument: BTC-27DEC24-100000-C
            parts = instrument_name.split("-")
            expiry_str = parts[1] if len(parts) > 1 else ""
            strike = float(parts[2]) if len(parts) > 2 else 0
            option_type = parts[3] if len(parts) > 3 else ""
            
            return {
                "instrument_name": instrument_name,
                "strike": strike,
                "option_type": "call" if option_type == "C" else "put",
                "expiry": expiry_str,
                "expiry_timestamp": instrument.get("expiration_timestamp", 0),
                "open_interest": ticker.get("open_interest", 0),
                "volume_24h": ticker.get("stats", {}).get("volume", 0),
                "mark_price": ticker.get("mark_price", 0),
                "mark_iv": ticker.get("mark_iv", 0),  # Implied Volatility
                "bid_iv": ticker.get("bid_iv", 0),
                "ask_iv": ticker.get("ask_iv", 0),
                "underlying_price": ticker.get("underlying_price", 0),
                "greeks": ticker.get("greeks", {}),
                "best_bid": ticker.get("best_bid_price", 0),
                "best_ask": ticker.get("best_ask_price", 0)
            }
            
        except Exception as e:
            return {"error": str(e), "instrument": instrument.get("instrument_name", "")}
    
    def _calculate_max_pain(self, options: List[Dict], current_price: float) -> Dict:
        """
        Calcule le Max Pain (prix où le plus d'options expirent sans valeur)
        
        C'est le prix où la somme des pertes pour les acheteurs d'options est maximale
        = prix où les market makers profitent le plus
        """
        # Grouper par strike
        strikes_data = defaultdict(lambda: {"call_oi": 0, "put_oi": 0})
        
        for opt in options:
            strike = opt.get("strike", 0)
            oi = opt.get("open_interest", 0)
            
            if opt.get("option_type") == "call":
                strikes_data[strike]["call_oi"] += oi
            else:
                strikes_data[strike]["put_oi"] += oi
        
        if not strikes_data:
            return {"max_pain_price": current_price, "confidence": "low"}
        
        # Calculer la douleur totale pour chaque strike potentiel
        all_strikes = sorted(strikes_data.keys())
        pain_by_strike = {}
        
        for test_price in all_strikes:
            total_pain = 0
            
            for strike, data in strikes_data.items():
                # Pain pour les calls: si prix < strike, les calls expirent OTM
                # Pain pour les puts: si prix > strike, les puts expirent OTM
                
                # Calls: paient si prix > strike
                if test_price > strike:
                    # Les calls sont ITM, pas de pain pour les call buyers
                    pass
                else:
                    # Les calls sont OTM, pain = call_oi * (strike - test_price) -> simplifié à call_oi
                    total_pain += data["call_oi"] * (strike - test_price)
                
                # Puts: paient si prix < strike
                if test_price < strike:
                    # Les puts sont ITM
                    pass
                else:
                    # Les puts sont OTM
                    total_pain += data["put_oi"] * (test_price - strike)
            
            pain_by_strike[test_price] = total_pain
        
        # Le max pain est le strike avec le MINIMUM de payout (max de pain pour buyers)
        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get) if pain_by_strike else current_price
        
        # Distance au max pain
        distance_usd = current_price - max_pain_strike
        distance_pct = (distance_usd / current_price * 100) if current_price > 0 else 0
        
        # Signal
        if abs(distance_pct) < 1:
            gravity = "AT_MAX_PAIN"
            signal = "NEUTRAL"
        elif distance_pct > 0:
            gravity = "ABOVE_MAX_PAIN"
            signal = "BEARISH"  # Prix tend à revenir vers max pain
        else:
            gravity = "BELOW_MAX_PAIN"
            signal = "BULLISH"
        
        return {
            "max_pain_price": max_pain_strike,
            "current_price": current_price,
            "distance_usd": round(distance_usd, 2),
            "distance_pct": round(distance_pct, 2),
            "gravity": gravity,
            "signal": signal,
            "top_strikes": sorted(
                [(s, d["call_oi"] + d["put_oi"]) for s, d in strikes_data.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
    
    def _calculate_put_call_ratio(self, options: List[Dict]) -> Dict:
        """
        Calcule le Put/Call Ratio basé sur l'Open Interest
        
        PCR > 1 = Plus de puts = Sentiment bearish
        PCR < 0.7 = Plus de calls = Sentiment bullish
        PCR 0.7-1 = Neutre
        """
        total_call_oi = 0
        total_put_oi = 0
        total_call_volume = 0
        total_put_volume = 0
        
        for opt in options:
            oi = opt.get("open_interest", 0)
            vol = opt.get("volume_24h", 0)
            
            if opt.get("option_type") == "call":
                total_call_oi += oi
                total_call_volume += vol
            else:
                total_put_oi += oi
                total_put_volume += vol
        
        # Ratio OI
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1
        
        # Ratio Volume
        pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 1
        
        # Interprétation
        if pcr_oi > 1.2:
            signal = "BEARISH"
            interpretation = "Beaucoup de puts = protection/spéculation baissière"
            emoji = "🔴"
        elif pcr_oi < 0.7:
            signal = "BULLISH"
            interpretation = "Beaucoup de calls = spéculation haussière"
            emoji = "🟢"
        else:
            signal = "NEUTRAL"
            interpretation = "Ratio équilibré"
            emoji = "⚪"
        
        return {
            "pcr_oi": round(pcr_oi, 3),
            "pcr_volume": round(pcr_volume, 3),
            "total_call_oi": round(total_call_oi, 2),
            "total_put_oi": round(total_put_oi, 2),
            "signal": signal,
            "emoji": emoji,
            "interpretation": interpretation
        }
    
    def _analyze_iv(self, options: List[Dict]) -> Dict:
        """
        Analyse l'Implied Volatility (IV)
        
        IV élevée = Marché anticipe de gros mouvements
        IV basse = Marché calme
        """
        ivs = [opt.get("mark_iv", 0) for opt in options if opt.get("mark_iv", 0) > 0]
        
        if not ivs:
            return {"error": "No IV data"}
        
        avg_iv = np.mean(ivs)
        median_iv = np.median(ivs)
        min_iv = min(ivs)
        max_iv = max(ivs)
        
        # IV Rank approximatif (normalement sur 52 semaines)
        # Ici on estime: 30-60 = normal, <30 = bas, >60 = élevé
        if avg_iv > 70:
            iv_environment = "VERY_HIGH"
            emoji = "🔴"
            interpretation = "Options très chères - Gros mouvement anticipé"
        elif avg_iv > 50:
            iv_environment = "HIGH"
            emoji = "🟠"
            interpretation = "Volatilité élevée - Prudence"
        elif avg_iv > 35:
            iv_environment = "NORMAL"
            emoji = "🟢"
            interpretation = "Volatilité normale"
        else:
            iv_environment = "LOW"
            emoji = "🟢"
            interpretation = "Volatilité basse - Calme avant tempête?"
        
        return {
            "average_iv": round(avg_iv, 1),
            "median_iv": round(median_iv, 1),
            "min_iv": round(min_iv, 1),
            "max_iv": round(max_iv, 1),
            "iv_environment": iv_environment,
            "emoji": emoji,
            "interpretation": interpretation
        }
    
    def _aggregate_oi_by_strike(self, options: List[Dict]) -> Dict:
        """Agrège l'Open Interest par strike"""
        by_strike = defaultdict(lambda: {"calls": 0, "puts": 0, "total": 0})
        
        for opt in options:
            strike = opt.get("strike", 0)
            oi = opt.get("open_interest", 0)
            
            if opt.get("option_type") == "call":
                by_strike[strike]["calls"] += oi
            else:
                by_strike[strike]["puts"] += oi
            by_strike[strike]["total"] += oi
        
        # Top 10 par OI total
        sorted_strikes = sorted(
            [(s, d) for s, d in by_strike.items()],
            key=lambda x: x[1]["total"],
            reverse=True
        )[:10]
        
        return {
            "top_strikes": [
                {
                    "strike": s,
                    "call_oi": round(d["calls"], 2),
                    "put_oi": round(d["puts"], 2),
                    "total_oi": round(d["total"], 2)
                }
                for s, d in sorted_strikes
            ],
            "total_open_interest": sum(d["total"] for d in by_strike.values())
        }
    
    def _get_nearest_expiry_analysis(self, options: List[Dict], 
                                      current_price: float) -> Dict:
        """Analyse l'expiration la plus proche"""
        # Grouper par expiry
        by_expiry = defaultdict(list)
        for opt in options:
            by_expiry[opt.get("expiry", "")].append(opt)
        
        if not by_expiry:
            return {"error": "No expiry data"}
        
        # Trouver la plus proche
        now = datetime.now(timezone.utc).timestamp() * 1000
        nearest_expiry = None
        nearest_ts = float('inf')
        
        for expiry, opts in by_expiry.items():
            ts = opts[0].get("expiry_timestamp", 0)
            if ts > now and ts < nearest_ts:
                nearest_ts = ts
                nearest_expiry = expiry
        
        if not nearest_expiry:
            return {"error": "No future expiry found"}
        
        expiry_opts = by_expiry[nearest_expiry]
        
        # Calculer les métriques pour cette expiry
        call_oi = sum(o.get("open_interest", 0) for o in expiry_opts if o.get("option_type") == "call")
        put_oi = sum(o.get("open_interest", 0) for o in expiry_opts if o.get("option_type") == "put")
        
        # Temps restant
        time_to_expiry_hours = (nearest_ts - now) / (1000 * 3600)
        
        return {
            "expiry": nearest_expiry,
            "hours_to_expiry": round(time_to_expiry_hours, 1),
            "call_oi": round(call_oi, 2),
            "put_oi": round(put_oi, 2),
            "pcr": round(put_oi / call_oi, 3) if call_oi > 0 else 0,
            "options_count": len(expiry_opts)
        }
    
    def _calculate_gex_profile(
        self, 
        options: List[Dict], 
        current_price: float,
        config: Optional[GEXConfig] = None
    ) -> Dict:
        """
        Calcule l'exposition Gamma (GEX) des Dealers avec optimisations.
        
        Améliorations vs version précédente:
        - Filtrage par horizon temporel (DTE)
        - Pondération par type d'expiration (quarterly 3x, monthly 2x, weekly 1x)
        - Filtre "fridays only" pour réduire le bruit
        - Interpolation linéaire pour Zero Gamma précis
        
        Args:
            options: Liste des options
            current_price: Prix spot actuel
            config: Configuration GEX (défaut: 65j horizon)
        
        Formula: Gamma * OI * Spot^2 * 0.01 (Dollar Gamma per 1% move)
        """
        if config is None:
            config = GEXConfig()  # Défauts: 65j, pondération 3x/2x/1x
        
        net_gamma = 0
        total_gamma = 0
        gex_by_strike = defaultdict(float)
        
        # Tracking pour warnings
        missed_quarterly_dtes = []
        filtered_count = 0
        total_count = len(options)
        
        # 1. Calculate GEX per strike avec filtres
        for opt in options:
            instrument_name = opt.get("instrument_name", "")
            expiry_ts = opt.get("expiry_timestamp", 0)
            gamma = opt.get("greeks", {}).get("gamma", 0)
            oi = opt.get("open_interest", 0)
            strike = opt.get("strike", 0)
            
            # === FILTRE 1: Open Interest minimum ===
            if oi < config.min_oi:
                filtered_count += 1
                continue
            
            # === FILTRE 2: DTE (Days To Expiry) ===
            dte_days = calculate_dte(expiry_ts)
            
            # Tracking quarterly manquées
            is_quarterly, is_monthly, weekday = detect_expiration_type(instrument_name, expiry_ts)
            
            if dte_days > config.dte_max:
                if is_quarterly:
                    missed_quarterly_dtes.append(dte_days)
                filtered_count += 1
                continue
            
            if dte_days < config.dte_min:
                filtered_count += 1
                continue
            
            # === FILTRE 3: Fridays Only (optionnel) ===
            if config.fridays_only and weekday != 4:
                filtered_count += 1
                continue
            
            # === FILTRE 4: Gamma valide ===
            if gamma == 0 or gamma is None:
                filtered_count += 1
                continue
            
            # === CALCUL GEX avec PONDÉRATION ===
            # Déterminer le poids selon le type d'expiration
            if is_quarterly:
                weight = config.weight_quarterly  # 3.0x par défaut
            elif is_monthly:
                weight = config.weight_monthly    # 2.0x par défaut
            else:
                weight = config.weight_weekly     # 1.0x par défaut
            
            # GEX en USD pour un mouvement de 1%
            # Gamma est coin-margined (en BTC), donc conversion :
            # GEX USD = Gamma * OI * Spot^2 / 100
            gex_value = (gamma * oi * (current_price ** 2)) / 100
            
            # Appliquer la pondération
            gex_value *= weight
            
            # Convention: 
            # - Clients achètent CALLS → Dealers SHORT CALLS → Gamma positive
            # - Clients achètent PUTS (hedge) → Dealers SHORT PUTS → Gamma négative
            if opt.get("option_type") == "call":
                gex_by_strike[strike] += gex_value
                net_gamma += gex_value
            else:
                gex_by_strike[strike] -= gex_value
                net_gamma -= gex_value
                
            total_gamma += abs(gex_value)

        net_gex_usd_m = net_gamma / 1_000_000
        
        # 2. Identify Walls and Zero Gamma
        if not gex_by_strike:
            return {
                "net_gex_usd_m": 0, 
                "regime": "NO_DATA",
                "warnings": [f"⚠️ Aucune option après filtrage ({filtered_count}/{total_count} filtrées)"],
                "stats": {
                    "total_options": total_count,
                    "filtered": filtered_count,
                    "analyzed": 0
                },
                "config": {
                    "dte_max": config.dte_max,
                    "fridays_only": config.fridays_only,
                    "weights": f"{config.weight_quarterly}x/{config.weight_monthly}x/{config.weight_weekly}x"
                }
            }

        sorted_strikes = sorted(gex_by_strike.items())
        
        # Call Wall: The strike with the most POSITIVE Gamma (Dealer Long Calls)
        call_wall = max(gex_by_strike.items(), key=lambda x: x[1])[0] 
        
        # Put Wall: The strike with the most NEGATIVE Gamma (Dealer Short Puts)
        put_wall = min(gex_by_strike.items(), key=lambda x: x[1])[0]
        
        # === AMÉLIORATION: Zero Gamma avec interpolation linéaire ===
        # Focus sur une fenêtre autour du spot (±15%)
        subset = {k: v for k, v in gex_by_strike.items() 
                  if current_price * 0.85 < k < current_price * 1.15}
        
        if not subset:
            # Fallback: fenêtre plus large
            subset = {k: v for k, v in gex_by_strike.items() 
                      if current_price * 0.5 < k < current_price * 2.0}
        
        # Séparer GEX négatif et positif
        neg_strikes = {k: v for k, v in subset.items() if v < 0}
        pos_strikes = {k: v for k, v in subset.items() if v > 0}
        
        zero_gamma = current_price  # Défaut
        
        if neg_strikes and pos_strikes:
            # Trouver le strike négatif le plus élevé
            max_neg_strike = max(neg_strikes.keys())
            neg_value = neg_strikes[max_neg_strike]
            
            # Trouver le strike positif le plus bas AU-DESSUS du neg
            candidates_pos = {k: v for k, v in pos_strikes.items() if k > max_neg_strike}
            
            if candidates_pos:
                min_pos_strike = min(candidates_pos.keys())
                pos_value = candidates_pos[min_pos_strike]
                
                # INTERPOLATION LINÉAIRE entre les deux strikes
                # Ratio basé sur les valeurs absolues
                total_abs = abs(neg_value) + abs(pos_value)
                if total_abs > 0:
                    ratio = abs(neg_value) / total_abs
                    zero_gamma = max_neg_strike + (min_pos_strike - max_neg_strike) * ratio
            else:
                # Pas de candidat positif au-dessus, prendre le strike le plus proche de 0
                zero_gamma = min(subset.items(), key=lambda x: abs(x[1]))[0]
        else:
            # Fallback: strike avec GEX le plus proche de zéro
            if subset:
                zero_gamma = min(subset.items(), key=lambda x: abs(x[1]))[0]
        
        # 3. Regime Interpretation
        if net_gex_usd_m > 5:
            regime = "POSITIVE_GAMMA"
            desc = "Dealers Long Gamma → Volatility Suppressed"
            emoji = "🛡️"
        elif net_gex_usd_m < -5:
            regime = "NEGATIVE_GAMMA"
            desc = "Dealers Short Gamma → Volatility Amplified"
            emoji = "☢️"
        else:
            regime = "NEUTRAL_GAMMA"
            desc = "Low Gamma Exposure"
            emoji = "⚪"
        
        # 4. Warnings pour quarterly manquées
        warnings = []
        if missed_quarterly_dtes:
            next_missed_q = min(missed_quarterly_dtes)
            if next_missed_q < (config.dte_max * 1.5):
                warnings.append(
                    f"⚠️ QUARTERLY PROCHE IGNORÉE : Dans {int(next_missed_q)} jours "
                    f"(horizon: {config.dte_max}j). Augmentez à {int(next_missed_q + 10)}j."
                )
        
        # Stats de filtrage
        if filtered_count > 0:
            filter_pct = (filtered_count / total_count * 100)
            logger.debug(f"GEX: {filtered_count}/{total_count} options filtrées ({filter_pct:.1f}%)")

        return {
            "net_gex_usd_m": round(net_gex_usd_m, 2),
            "total_gamma_usd_m": round(total_gamma / 1_000_000, 2),
            "regime": regime,
            "description": desc,
            "emoji": emoji,
            "call_wall": call_wall,  # Strike
            "put_wall": put_wall,    # Strike
            "zero_gamma": round(zero_gamma, 2),
            "key_levels": sorted(
                [(s, v/1_000_000) for s, v in gex_by_strike.items()],
                key=lambda x: abs(x[1]), reverse=True
            )[:5],
            "config": {
                "dte_max": config.dte_max,
                "fridays_only": config.fridays_only,
                "weights": f"{config.weight_quarterly}x/{config.weight_monthly}x/{config.weight_weekly}x"
            },
            "stats": {
                "total_options": total_count,
                "filtered": filtered_count,
                "analyzed": total_count - filtered_count
            },
            "warnings": warnings
        }
    def _calculate_options_score(self, max_pain: Dict, pc_ratio: Dict, 
                                  iv_analysis: Dict, gex: Dict, current_price: float) -> Dict:
        """Calcule un score global options"""
        score = 50
        factors = []
        
        # 1. Max Pain (Gravity)
        if max_pain.get("signal") == "BULLISH":
            score += 10
            factors.append("+10 Bullish Gravity (Under Max Pain)")
        elif max_pain.get("signal") == "BEARISH":
            score -= 10
            factors.append("-10 Bearish Gravity (Above Max Pain)")
            
        # 2. PCR
        pcr = pc_ratio.get("pcr_oi", 1)
        if pcr < 0.7: score += 10
        elif pcr > 1.2: score -= 10
        
        # 3. IV
        iv = iv_analysis.get("iv_environment")
        if iv == "LOW": score += 5
        elif iv == "VERY_HIGH": score -= 5
        
        # 4. GEX (Nouveau)
        # Positive GEX = Bullish pour le "Range" (Support tient)
        # Negative GEX = Bearish si prix baisse, Bullish si prix monte (Accélération)
        # C'est contextuel. On va dire:
        # Positive GEX -> +Stabilizing (Bon pour swing setups)
        # Negative GEX -> +Risk (Bon pour breakout, mauvais pour range)
        
        gex_val = gex.get("net_gex_usd_m", 0)
        if gex_val > 0:
            score += 5
            factors.append(f"+5 Positive Gamma (${gex_val}M) - Sticky Market")
        else:
            # Negative gamma increase risk/volatility
            factors.append(f"⚠️ Negative Gamma (${gex_val}M) - Volatility Warning")



# ============================================================
# Wrapper synchrone pour compatibilité
# ============================================================

class OptionsAnalyzer:
    """Wrapper synchrone pour DeribitOptionsAnalyzer"""
    
    def __init__(self):
        pass
    
    def analyze(self, current_price: float = None) -> Dict[str, Any]:
        """Version synchrone"""
        try:
            return asyncio.run(self._run_analysis(current_price))
        except RuntimeError:
            # Fallback si déjà dans un event loop
            return self._sync_fallback(current_price)
        except Exception as e:
            logger.error(f"Erreur options analyzer: {e}")
            return {"error": str(e)}
    
    async def _run_analysis(self, current_price: float) -> Dict[str, Any]:
        analyzer = DeribitOptionsAnalyzer()
        try:
            return await analyzer.analyze(current_price)
        finally:
            await analyzer.close()
    
    def _sync_fallback(self, current_price: float) -> Dict[str, Any]:
        """Fallback simplifié"""
        import requests
        
        result = {
            "max_pain": {"max_pain_price": 0, "signal": "UNKNOWN"},
            "put_call_ratio": {"pcr_oi": 1.0, "signal": "NEUTRAL"},
            "iv_analysis": {"average_iv": 50, "iv_environment": "NORMAL"},
            "gex_profile": {"net_gex_usd_m": 0, "regime": "NEUTRAL"}, # ADDED
            "score": {"value": 50, "sentiment": "NEUTRAL", "emoji": "⚪"}
        }
        
        try:
            # Essayer de récupérer au moins le prix index
            resp = requests.get(
                "https://www.deribit.com/api/v2/public/get_index_price",
                params={"index_name": "btc_usd"},
                timeout=10
            )
            if resp.ok:
                result["current_price"] = resp.json().get("result", {}).get("index_price", 0)
        except Exception:
            pass
        
        return result
