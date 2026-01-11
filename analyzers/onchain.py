"""
On-Chain Analytics V2
Whale Tracking + Blockchain Metrics entièrement gratuits
Basé sur blockchain.info, mempool.space, et autres APIs gratuites
"""
import asyncio
import httpx
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import deque
import time

import config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FreeWhaleTracker:
    """Tracker de baleines 100% gratuit et asynchrone optimisé."""

    def __init__(self) -> None:
        # ------------------------------------------------------------------ #
        # 1. Config & seuils
        # ------------------------------------------------------------------ #
        self.blockchain_api = "https://blockchain.info"
        
        # Granularité requin/baleine/mega
        self.levels = {
            "shark": 10,        # 10-30 BTC  = 🦈
            "whale": 30,        # 30-100 BTC = 🐋
            "mega_whale": 100   # 100+ BTC   = 🐋🐋🐋
        }
        self.whale_threshold = self.levels["shark"]

        # ------------------------------------------------------------------ #
        # 2. État runtime
        # ------------------------------------------------------------------ #
        self.address_labels: Dict[str, str] = self._load_known_addresses()
        self.alert_cache: deque = deque(maxlen=100)
        self.seen_tx: set = set()
        self.last_block_hash: Optional[str] = None

        # ------------------------------------------------------------------ #
        # 3. Client HTTP persistant
        # ------------------------------------------------------------------ #
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy init du client HTTP"""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    async def close(self) -> None:
        """Ferme proprement le client HTTP."""
        if self._http:
            await self._http.aclose()

    def _load_known_addresses(self) -> Dict[str, str]:
        """Charge les adresses connues des principaux exchanges."""
        return {
            # Binance
            "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "Binance Cold Wallet",
            "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6": "Binance Hot Wallet",
            "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h": "Binance Cold 2",
            "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s": "Binance 3",
            # Bitfinex
            "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r": "Bitfinex Cold",
            "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g": "Bitfinex Hot",
            # Coinbase
            "3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64": "Coinbase Cold",
            "3Nxwenay9Z8Lc9JBiywExpnEFiLp6Afp8v": "Coinbase Hot",
            "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh": "Coinbase Commerce",
            # Huobi
            "1LAnF8h3qMGx3TSwNUHVneBZUEpwE4gu3D": "Huobi Cold",
            # Kraken
            "2NHguzT8fxwJ3eqAGzSfUm8kT96L1PcVZG": "Kraken Cold",
            "bc1qxu2jwr0yfq6r6f5p2l6u6g6dpkwmx4ex4yfqxe": "Kraken 2",
            # OKEx / OKX
            "3KZ526NxCVXbKwwP66RgM3pte6zW4gY1tD": "OKEx Cold",
            # Bybit
            "bc1qjasf9z3h7w3jspkhtgatgpyvvzgpa2wwd2lr0p": "Bybit",
            # Bitget
            "bc1qvdnpntlxe5zy2n9rk4fvkxppyslq9fw6gq9kcp": "Bitget"
        }

    def _categorize_transaction(self, btc_amount: float) -> str:
        """Renvoie un label emoji selon la taille de la transaction."""
        if btc_amount >= self.levels["mega_whale"]:
            return "🐋🐋🐋 MÉGA BALEINE"
        if btc_amount >= self.levels["whale"]:
            return "🐋 BALEINE"
        if btc_amount >= self.levels["shark"]:
            return "🦈 REQUIN"
        return "🐟 Poisson"

    async def check_movements(self) -> Dict[str, Any]:
        """Vérifie les mouvements de baleines et renvoie un rapport."""
        try:
            large_txs = await self._fetch_large_transactions()
            exchange_flows = await self._analyze_exchange_flows(large_txs)
            suspicious_patterns = self._detect_suspicious_patterns(large_txs)
            score = self._calculate_whale_score(large_txs, exchange_flows, suspicious_patterns)
            
            # Interprétation
            if score > 0.7:
                activity_level = "HIGH"
                emoji = "🐋"
            elif score > 0.4:
                activity_level = "MODERATE"
                emoji = "🦈"
            else:
                activity_level = "LOW"
                emoji = "🐟"
            
            return {
                "whale_activity": bool(large_txs),
                "activity_level": activity_level,
                "emoji": emoji,
                "score": round(score, 3),
                "large_transactions": large_txs[:10],
                "exchange_flows": exchange_flows,
                "patterns": suspicious_patterns,
                "stats": {
                    "total_tx_found": len(large_txs),
                    "mega_whales": len([t for t in large_txs if t["value_btc"] >= self.levels["mega_whale"]]),
                    "whales": len([t for t in large_txs if self.levels["whale"] <= t["value_btc"] < self.levels["mega_whale"]]),
                    "sharks": len([t for t in large_txs if self.levels["shark"] <= t["value_btc"] < self.levels["whale"]])
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Erreur whale tracker: {e}")
            return {"whale_activity": False, "score": 0.0, "error": str(e)}

    async def _fetch_large_transactions(self) -> List[Dict[str, Any]]:
        """Récupère les grosses transactions depuis blockchain.info et mempool.space."""
        try:
            client = await self._get_client()
            
            # Appels en parallèle
            tasks = [
                client.get(f"{self.blockchain_api}/latestblock", follow_redirects=True),
                client.get("https://mempool.space/api/mempool/recent", timeout=10)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            block_resp, mempool_resp = responses

            large_txs: List[Dict[str, Any]] = []

            # Traitement bloc
            if not isinstance(block_resp, Exception) and block_resp.status_code == 200:
                lb = block_resp.json()
                bh = lb.get("hash")
                if bh and bh != self.last_block_hash:
                    self.last_block_hash = bh
                    try:
                        raw_resp = await client.get(
                            f"{self.blockchain_api}/rawblock/{bh}", 
                            follow_redirects=True,
                            timeout=20
                        )
                        blk = raw_resp.json()
                        for tx in blk.get("tx", [])[:100]:
                            txh = tx.get("hash")
                            if not txh or txh in self.seen_tx:
                                continue
                            total = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
                            if total >= self.whale_threshold:
                                self.seen_tx.add(txh)
                                ins = self._analyze_tx_inputs(tx)
                                outs = self._analyze_tx_outputs(tx)
                                large_txs.append({
                                    "hash": txh,
                                    "time": tx.get("time", 0),
                                    "value_btc": round(total, 4),
                                    "tier": self._categorize_transaction(total),
                                    "inputs": ins,
                                    "outputs": outs,
                                    "is_exchange": self._is_exchange_related(ins, outs),
                                    "source": "blockchain"
                                })
                    except Exception as e:
                        logger.warning(f"Erreur fetch raw block: {e}")
            
            # Fallback mempool
            if not isinstance(mempool_resp, Exception) and mempool_resp.status_code == 200:
                for txd in mempool_resp.json()[:50]:
                    if txd.get("value", 0) >= self.whale_threshold * 1e8:
                        txid = txd.get("txid")
                        if txid and txid not in self.seen_tx:
                            total = txd.get("value", 0) / 1e8
                            self.seen_tx.add(txid)
                            large_txs.append({
                                "hash": txid,
                                "time": int(datetime.now().timestamp()),
                                "value_btc": round(total, 4),
                                "tier": self._categorize_transaction(total),
                                "inputs": [],
                                "outputs": [],
                                "is_exchange": False,
                                "source": "mempool"
                            })
            
            # Tri et top
            large_txs.sort(key=lambda x: x["value_btc"], reverse=True)
            return large_txs[:20]
            
        except Exception as e:
            logger.error(f"Erreur fetch transactions: {e}")
            return []

    def _analyze_tx_inputs(self, tx: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs: List[Dict[str, Any]] = []
        for inp in tx.get("inputs", []):
            prev = inp.get("prev_out", {})
            addr = prev.get("addr", "Unknown")
            val = prev.get("value", 0) / 1e8
            inputs.append({
                "address": addr[:12] + "..." if len(addr) > 15 else addr,
                "value_btc": round(val, 4),
                "label": self.address_labels.get(addr, "Unknown"),
                "is_exchange": addr in self.address_labels
            })
        return inputs[:5]  # Limiter à 5

    def _analyze_tx_outputs(self, tx: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for out in tx.get("out", []):
            addr = out.get("addr", "Unknown")
            val = out.get("value", 0) / 1e8
            outputs.append({
                "address": addr[:12] + "..." if len(addr) > 15 else addr,
                "value_btc": round(val, 4),
                "label": self.address_labels.get(addr, "Unknown"),
                "is_exchange": addr in self.address_labels
            })
        return outputs[:5]  # Limiter à 5

    def _is_exchange_related(self, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]]) -> bool:
        return any(i.get("is_exchange") for i in inputs) or any(o.get("is_exchange") for o in outputs)

    async def _analyze_exchange_flows(self, txs: List[Dict[str, Any]]) -> Dict[str, Any]:
        inflow = outflow = 0.0
        ex_txs: List[Dict[str, Any]] = []
        
        for tx in txs:
            if not tx.get("is_exchange"):
                continue
            
            inp_ex = sum(1 for i in tx.get("inputs", []) if i.get("is_exchange"))
            out_ex = sum(1 for o in tx.get("outputs", []) if o.get("is_exchange"))
            
            # Si l'output va vers un exchange et l'input ne vient pas d'un exchange = INFLOW
            # Si l'input vient d'un exchange et l'output ne va pas vers un exchange = OUTFLOW
            if out_ex > inp_ex:
                direction = "INFLOW"
                inflow += tx["value_btc"]
            else:
                direction = "OUTFLOW"
                outflow += tx["value_btc"]
            
            ex_txs.append({
                "hash": tx["hash"][:16] + "...",
                "value_btc": tx["value_btc"],
                "direction": direction,
                "time": tx["time"]
            })
        
        net = inflow - outflow
        
        # Signal basé sur le net flow
        if net < -100:  # Outflow > 100 BTC
            signal = "ACCUMULATION"
            emoji = "🟢"
            interpretation = "Sorties massives des exchanges = Accumulation"
        elif net > 100:  # Inflow > 100 BTC
            signal = "DISTRIBUTION"
            emoji = "🔴"
            interpretation = "Entrées massives sur exchanges = Distribution potentielle"
        else:
            signal = "NEUTRAL"
            emoji = "⚪"
            interpretation = "Flux équilibrés"
        
        return {
            "inflow_btc": round(inflow, 2),
            "outflow_btc": round(outflow, 2),
            "net_flow_btc": round(net, 2),
            "signal": signal,
            "emoji": emoji,
            "interpretation": interpretation,
            "exchange_transactions": ex_txs[:5]
        }

    def _detect_suspicious_patterns(self, txs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        
        # 1. Splitting - Transactions de même taille
        amounts: Dict[float, List[Dict[str, Any]]] = {}
        for tx in txs:
            amt = round(tx["value_btc"], 0)
            amounts.setdefault(amt, []).append(tx)
        
        for amt, group in amounts.items():
            if len(group) >= 3:
                patterns.append({
                    "type": "SPLITTING",
                    "emoji": "✂️",
                    "count": len(group),
                    "message": f"Fractionnement détecté: {len(group)} tx de ~{amt:.0f} BTC"
                })
        
        # 2. Burst activity - Beaucoup de TX en peu de temps
        times: Dict[int, List[Dict[str, Any]]] = {}
        for tx in txs:
            bucket = tx["time"] // 300  # Buckets de 5 min
            times.setdefault(bucket, []).append(tx)
        
        for bucket, group in times.items():
            if len(group) >= 5:
                total = sum(t["value_btc"] for t in group)
                patterns.append({
                    "type": "BURST_ACTIVITY",
                    "emoji": "⚡",
                    "count": len(group),
                    "message": f"Burst: {len(group)} tx en 5min ({total:.0f} BTC)"
                })
        
        return patterns

    def _calculate_whale_score(self, txs: List[Dict[str, Any]], 
                               flows: Dict[str, Any], 
                               patterns: List[Dict[str, Any]]) -> float:
        score = 0.0
        
        # 1. Volume de transactions
        n = len(txs)
        if n > 10:
            score += 0.3
        elif n > 5:
            score += 0.2
        elif n > 2:
            score += 0.1
        
        # 2. Exchange flows
        net = abs(flows.get("net_flow_btc", 0))
        if net > 1000:
            score += 0.3
        elif net > 500:
            score += 0.2
        elif net > 100:
            score += 0.1
        
        # 3. Patterns suspects
        if len(patterns) >= 2:
            score += 0.3
        elif len(patterns) == 1:
            score += 0.2
        
        # 4. Bonus accumulation
        if flows.get("signal") == "ACCUMULATION":
            score += 0.1
        
        return min(score, 1.0)


class FreeBlockchainMetrics:
    """Collecte des métriques blockchain via APIs gratuites"""

    def __init__(self):
        self.apis = {
            "blockchain_info": "https://api.blockchain.info/stats",
            "mempool_space": "https://mempool.space/api"
        }
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()

    async def get_network_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques réseau principales"""
        try:
            client = await self._get_client()
            
            # Appels en parallèle
            tasks = [
                self._fetch_blockchain_info_stats(client),
                self._analyze_mempool(client),
                self._get_mining_stats(client)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            metrics = {}
            for result in results:
                if isinstance(result, dict) and "error" not in result:
                    metrics.update(result)
            
            # Score de santé du réseau
            health_score = self._calculate_network_health(metrics)
            metrics["network_health"] = health_score
            
            return metrics

        except Exception as e:
            logger.error(f"Erreur metrics blockchain: {e}")
            return {"error": str(e)}

    async def _fetch_blockchain_info_stats(self, client: httpx.AsyncClient) -> Dict[str, Any]:
        """Récupère les stats depuis blockchain.info"""
        try:
            response = await client.get(self.apis["blockchain_info"])
            data = response.json()

            hash_rate_eh = data.get("hash_rate", 0) / 1e18  # Convertir en EH/s

            return {
                "difficulty": data.get("difficulty", 0),
                "hash_rate_eh": round(hash_rate_eh, 2),
                "total_fees_btc": data.get("total_fees_btc", 0),
                "n_tx_24h": data.get("n_tx", 0),
                "n_blocks_total": data.get("n_blocks_total", 0),
                "minutes_between_blocks": data.get("minutes_between_blocks", 0),
                "market_price_usd": data.get("market_price_usd", 0),
                "miners_revenue_btc": data.get("miners_revenue_btc", 0)
            }

        except Exception as e:
            logger.error(f"Erreur blockchain.info stats: {e}")
            return {"error": str(e)}

    async def _analyze_mempool(self, client: httpx.AsyncClient) -> Dict[str, Any]:
        """Analyse le mempool pour détecter la congestion"""
        try:
            # Stats mempool
            mempool_resp = await client.get(f"{self.apis['mempool_space']}/mempool")
            mempool_data = mempool_resp.json()

            # Fees recommandés
            fees_resp = await client.get(f"{self.apis['mempool_space']}/v1/fees/recommended")
            fees_data = fees_resp.json()

            unconfirmed_count = mempool_data.get("count", 0)
            mempool_vsize = mempool_data.get("vsize", 0)

            # Calculer la congestion
            fastest_fee = fees_data.get("fastestFee", 0)

            if fastest_fee > 100:
                congestion = "VERY_HIGH"
                congestion_emoji = "🔴"
            elif fastest_fee > 50:
                congestion = "HIGH"
                congestion_emoji = "🟠"
            elif fastest_fee > 20:
                congestion = "MEDIUM"
                congestion_emoji = "🟡"
            else:
                congestion = "LOW"
                congestion_emoji = "🟢"

            return {
                "mempool_tx_count": unconfirmed_count,
                "mempool_vsize_mb": round(mempool_vsize / 1e6, 2),
                "congestion": congestion,
                "congestion_emoji": congestion_emoji,
                "fastest_fee_sat": fastest_fee,
                "half_hour_fee_sat": fees_data.get("halfHourFee", 0),
                "hour_fee_sat": fees_data.get("hourFee", 0),
                "economy_fee_sat": fees_data.get("economyFee", 0)
            }

        except Exception as e:
            logger.error(f"Erreur mempool: {e}")
            return {"congestion": "UNKNOWN", "error": str(e)}

    async def _get_mining_stats(self, client: httpx.AsyncClient) -> Dict[str, Any]:
        """Récupère les statistiques du dernier bloc"""
        try:
            response = await client.get("https://blockchain.info/latestblock")
            latest_block = response.json()

            block_time = datetime.fromtimestamp(latest_block.get("time", 0), tz=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - block_time).total_seconds()

            return {
                "latest_block_height": latest_block.get("height", 0),
                "latest_block_time": latest_block.get("time", 0),
                "block_age_seconds": int(age_seconds),
                "latest_block_hash": latest_block.get("hash", "")[:16] + "..."
            }

        except Exception as e:
            logger.error(f"Erreur mining stats: {e}")
            return {"error": str(e)}

    def _calculate_network_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule un score de santé du réseau"""
        score = 50  # Base neutre

        # 1. Temps entre blocs (idéal = 10 minutes)
        block_time = metrics.get("minutes_between_blocks", 10)
        if 9 <= block_time <= 11:
            score += 10
        elif block_time > 15 or block_time < 5:
            score -= 15

        # 2. Congestion du mempool
        congestion = metrics.get("congestion", "MEDIUM")
        if congestion == "LOW":
            score += 10
        elif congestion == "HIGH":
            score -= 10
        elif congestion == "VERY_HIGH":
            score -= 20

        # 3. Hash rate (présence = réseau actif)
        hash_rate = metrics.get("hash_rate_eh", 0)
        if hash_rate > 600:
            score += 10

        # 4. Fees (fees bas = bonne santé)
        fastest_fee = metrics.get("fastest_fee_sat", 50)
        if fastest_fee < 20:
            score += 10
        elif fastest_fee > 100:
            score -= 10

        # Normaliser
        score = max(0, min(100, score))

        if score >= 70:
            status = "HEALTHY"
            emoji = "🟢"
        elif score >= 50:
            status = "NORMAL"
            emoji = "🟡"
        else:
            status = "DEGRADED"
            emoji = "🔴"

        return {
            "score": score,
            "status": status,
            "emoji": emoji
        }


class OnChainAnalyzerV2:
    """
    Analyseur On-Chain V2 combinant Whale Tracking et Blockchain Metrics
    Entièrement gratuit, basé sur APIs publiques
    """

    def __init__(self):
        self.whale_tracker = FreeWhaleTracker()
        self.blockchain_metrics = FreeBlockchainMetrics()

    async def analyze(self, current_price: float) -> Dict[str, Any]:
        """
        Analyse complète on-chain
        
        Args:
            current_price: Prix BTC actuel
            
        Returns:
            Dict avec whale activity, network metrics, et score global
        """
        try:
            # Exécuter les analyses en parallèle
            results = await asyncio.gather(
                self.whale_tracker.check_movements(),
                self.blockchain_metrics.get_network_metrics(),
                return_exceptions=True
            )

            whale_data = results[0] if not isinstance(results[0], Exception) else {}
            network_data = results[1] if not isinstance(results[1], Exception) else {}

            # Calculer le score on-chain global
            onchain_score = self._calculate_onchain_score(whale_data, network_data)

            return {
                "whale": whale_data,
                "network": network_data,
                "score": onchain_score,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Erreur analyse on-chain: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Ferme les clients HTTP"""
        await self.whale_tracker.close()
        await self.blockchain_metrics.close()

    def _calculate_onchain_score(self, whale: Dict, network: Dict) -> Dict[str, Any]:
        """Calcule un score on-chain global (0-100)"""
        score = 50  # Base neutre
        factors = []

        # 1. Whale activity
        whale_score = whale.get("score", 0)
        if whale_score > 0.6:
            score += 10
            factors.append(f"+10 Forte activité whale ({whale_score:.2f})")
        elif whale_score < 0.2:
            score -= 5
            factors.append("-5 Faible activité whale")

        # 2. Exchange flows
        flows = whale.get("exchange_flows", {})
        flow_signal = flows.get("signal", "NEUTRAL")
        if flow_signal == "ACCUMULATION":
            score += 15
            factors.append("+15 Accumulation détectée (outflow exchanges)")
        elif flow_signal == "DISTRIBUTION":
            score -= 15
            factors.append("-15 Distribution détectée (inflow exchanges)")

        # 3. Network health
        health = network.get("network_health", {})
        health_score_raw = health.get("score", 50)
        health_adj = (health_score_raw - 50) / 5  # -10 à +10
        score += health_adj
        if abs(health_adj) > 5:
            factors.append(f"{health_adj:+.0f} Santé réseau ({health.get('status', 'UNKNOWN')})")

        # 4. Congestion (fees bas = bullish car activité normale)
        congestion = network.get("congestion", "MEDIUM")
        if congestion == "LOW":
            score += 5
            factors.append("+5 Congestion basse (activité normale)")
        elif congestion in ["HIGH", "VERY_HIGH"]:
            score += 5  # Activité haute peut être bullish aussi
            factors.append("+5 Forte activité on-chain")

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
            "value": round(score, 1),
            "sentiment": sentiment,
            "emoji": emoji,
            "factors": factors
        }


# ============================================================
# Wrapper synchrone pour compatibilité avec main_v2.py
# ============================================================

class OnChainAnalyzer:
    """
    Wrapper synchrone de OnChainAnalyzerV2
    Permet l'intégration avec le code synchrone existant
    """

    def __init__(self, *args, **kwargs):
        pass  # Lazy init

    def analyze(self, current_price: float) -> Dict[str, Any]:
        """Version synchrone de analyze"""
        try:
            # Utiliser asyncio.run() dans un nouveau contexte
            return asyncio.run(self._run_analysis(current_price))
        except RuntimeError:
            # Fallback si déjà dans un event loop
            return self._sync_fallback(current_price)
        except Exception as e:
            logger.error(f"Erreur wrapper sync on-chain: {e}")
            return {"error": str(e), "score": {"value": 50, "sentiment": "NEUTRAL", "emoji": "⚪"}}

    async def _run_analysis(self, current_price: float) -> Dict[str, Any]:
        """Exécute l'analyse async"""
        analyzer = OnChainAnalyzerV2()
        try:
            result = await analyzer.analyze(current_price)
            return result
        finally:
            await analyzer.close()

    def _sync_fallback(self, current_price: float) -> Dict[str, Any]:
        """Fallback synchrone simplifié si event loop existe"""
        import requests
        
        result = {
            "whale": {},
            "network": {},
            "score": {"value": 50, "sentiment": "NEUTRAL", "emoji": "⚪", "factors": []},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Mempool fees
            fees_resp = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10)
            if fees_resp.ok:
                fees = fees_resp.json()
                fastest = fees.get("fastestFee", 0)
                
                if fastest > 100:
                    congestion = "VERY_HIGH"
                elif fastest > 50:
                    congestion = "HIGH"
                elif fastest > 20:
                    congestion = "MEDIUM"
                else:
                    congestion = "LOW"
                
                result["network"] = {
                    "congestion": congestion,
                    "fastest_fee_sat": fastest
                }
            
            # Blockchain stats
            stats_resp = requests.get("https://api.blockchain.info/stats", timeout=10)
            if stats_resp.ok:
                stats = stats_resp.json()
                result["network"]["hash_rate_eh"] = round(stats.get("hash_rate", 0) / 1e18, 2)
                result["network"]["market_price_usd"] = stats.get("market_price_usd", 0)
                
        except Exception as e:
            logger.warning(f"Fallback sync partiel: {e}")
        
        return result


# Garder la compatibilité avec l'ancien StablecoinAnalyzer
class StablecoinAnalyzer:
    """Analyse des stablecoins (déplacé de l'ancienne implémentation)"""
    
    DEFI_LLAMA_API = "https://stablecoins.llama.fi"
    
    def __init__(self):
        self.session = None
    
    def analyze(self) -> Dict[str, Any]:
        """Analyse la supply de stablecoins"""
        try:
            import requests
            response = requests.get(
                f"{self.DEFI_LLAMA_API}/stablecoincharts/all",
                timeout=10
            )
            
            if not response.ok:
                return {'error': 'Failed to fetch stablecoin data'}
            
            data = response.json()
            
            if data:
                latest = data[-1] if isinstance(data, list) else data
                total_supply_b = latest.get('totalCirculating', {}).get('peggedUSD', 0) / 1e9
                
                if len(data) >= 2:
                    previous = data[-2]
                    prev_supply = previous.get('totalCirculating', {}).get('peggedUSD', 0) / 1e9
                    change_24h = total_supply_b - prev_supply
                    change_pct = (change_24h / prev_supply * 100) if prev_supply > 0 else 0
                else:
                    change_24h = 0
                    change_pct = 0
                
                if change_pct > 0.5:
                    signal = "BULLISH"
                    interpretation = "Supply stablecoin en hausse = Dry powder augmente"
                elif change_pct < -0.5:
                    signal = "BEARISH"
                    interpretation = "Supply stablecoin en baisse = Retrait de liquidité"
                else:
                    signal = "NEUTRAL"
                    interpretation = "Supply stablecoin stable"
                
                return {
                    'total_supply_b': round(total_supply_b, 2),
                    'change_24h_b': round(change_24h, 3),
                    'change_24h_pct': round(change_pct, 3),
                    'signal': signal,
                    'interpretation': interpretation
                }
            
            return {'error': 'No data available'}
            
        except Exception as e:
            return {'error': str(e)}
