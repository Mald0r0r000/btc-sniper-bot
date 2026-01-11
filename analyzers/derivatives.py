"""
Derivatives Intelligence
Analyse avancée des produits dérivés : Options, Futures, Liquidations
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import ccxt

import config


class DerivativesAnalyzer:
    """
    Analyse les données de dérivés pour intelligence de marché
    
    Features:
    - Futures basis (premium/discount)
    - Term structure analysis
    - Funding rate divergence
    - Liquidation heatmap améliorée
    - Options data (si disponible via API)
    """
    
    def __init__(self, exchange_data: Dict[str, Any] = None):
        """
        Args:
            exchange_data: Données brutes des exchanges (optionnel)
        """
        self.exchange_data = exchange_data or {}
        
    def analyze(self, current_price: float, funding_rates: Dict[str, float], 
                open_interests: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyse complète des dérivés
        
        Args:
            current_price: Prix spot actuel
            funding_rates: {exchange: rate} 
            open_interests: {exchange: oi_amount}
            
        Returns:
            Dict avec analyse complète
        """
        futures_analysis = self._analyze_futures_basis(current_price, funding_rates)
        funding_analysis = self._analyze_funding_dynamics(funding_rates)
        liquidation_map = self._generate_liquidation_heatmap(current_price, open_interests)
        options_analysis = self._analyze_options_market(current_price)
        
        # Sentiment dérivé global
        derivative_sentiment = self._calculate_derivative_sentiment(
            futures_analysis, funding_analysis, liquidation_map
        )
        
        return {
            'futures': futures_analysis,
            'funding': funding_analysis,
            'liquidations': liquidation_map,
            'options': options_analysis,
            'sentiment': derivative_sentiment,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_futures_basis(self, spot_price: float, funding_rates: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyse le basis (premium/discount des futures vs spot)
        
        Le basis est approximé via le funding rate:
        - Funding positif élevé = Futures en premium (contango)
        - Funding négatif = Futures en discount (backwardation)
        """
        if not funding_rates:
            return {'error': 'No funding data'}
        
        avg_funding = np.mean(list(funding_rates.values()))
        
        # Annualiser le funding pour estimer le basis
        # Funding = 3x par jour, donc basis annualisé ≈ funding * 3 * 365
        annualized_basis_pct = avg_funding * 3 * 365 * 100
        
        # Determiner la structure
        if annualized_basis_pct > 15:
            structure = "STEEP_CONTANGO"
            signal = "BEARISH"
            description = "Futures très chers vs spot - Euphorie excessive"
        elif annualized_basis_pct > 5:
            structure = "CONTANGO"
            signal = "NEUTRAL_BULLISH"
            description = "Structure normale de marché haussier"
        elif annualized_basis_pct > -2:
            structure = "FLAT"
            signal = "NEUTRAL"
            description = "Pas de premium significatif"
        elif annualized_basis_pct > -10:
            structure = "BACKWARDATION"
            signal = "BULLISH"
            description = "Futures en discount - Opportunité d'achat"
        else:
            structure = "DEEP_BACKWARDATION"
            signal = "VERY_BULLISH"
            description = "Capitulation - Forte opportunité"
        
        return {
            'average_funding_pct': round(avg_funding * 100, 4),
            'annualized_basis_pct': round(annualized_basis_pct, 2),
            'term_structure': structure,
            'signal': signal,
            'description': description,
            'by_exchange': {ex: round(rate * 100, 4) for ex, rate in funding_rates.items()}
        }
    
    def _analyze_funding_dynamics(self, funding_rates: Dict[str, float]) -> Dict[str, Any]:
        """Analyse la dynamique du funding entre exchanges"""
        if not funding_rates or len(funding_rates) < 2:
            return {'divergence': False}
        
        rates = list(funding_rates.values())
        exchanges = list(funding_rates.keys())
        
        max_rate = max(rates)
        min_rate = min(rates)
        divergence = (max_rate - min_rate) * 100  # En pourcentage
        
        max_ex = exchanges[rates.index(max_rate)]
        min_ex = exchanges[rates.index(min_rate)]
        
        # Significatif si divergence > 0.02%
        is_significant = divergence > 0.02
        
        # Opportunité d'arbitrage de funding
        arbitrage_opportunity = None
        if is_significant:
            arbitrage_opportunity = {
                'long_on': min_ex,  # Long où funding est bas (on reçoit)
                'short_on': max_ex,  # Short où funding est haut (on reçoit)
                'expected_yield_8h': round(divergence / 2, 4),
                'expected_yield_daily': round(divergence / 2 * 3, 4)
            }
        
        return {
            'divergence_pct': round(divergence, 4),
            'is_significant': is_significant,
            'highest': {'exchange': max_ex, 'rate_pct': round(max_rate * 100, 4)},
            'lowest': {'exchange': min_ex, 'rate_pct': round(min_rate * 100, 4)},
            'arbitrage': arbitrage_opportunity,
            'interpretation': f"Longs paient plus sur {max_ex}" if is_significant else "Funding équilibré"
        }
    
    def _generate_liquidation_heatmap(self, current_price: float, 
                                      open_interests: Dict[str, float]) -> Dict[str, Any]:
        """
        Génère une heatmap des zones de liquidation
        
        Amélioration du calcul basique avec:
        - Distribution estimée des positions par prix d'entrée
        - Concentration des liquidations par zone
        - Intensité basée sur l'OI
        """
        total_oi = sum(open_interests.values())
        
        if total_oi == 0:
            return {'error': 'No OI data'}
        
        # Niveaux de levier standards
        leverages = [125, 100, 75, 50, 25, 20, 10, 5]
        
        # Générer les niveaux de liquidation
        long_liquidations = []
        short_liquidations = []
        
        for lev in leverages:
            move_pct = 1 / lev
            
            # Long liquidations (en dessous du prix)
            long_liq_price = current_price * (1 - move_pct)
            
            # Short liquidations (au dessus du prix)
            short_liq_price = current_price * (1 + move_pct)
            
            # Estimer l'intensité basée sur l'OI et le levier
            # Plus le levier est élevé, plus il y a de positions (en général)
            intensity = self._estimate_position_intensity(lev, total_oi)
            
            long_liquidations.append({
                'leverage': lev,
                'price': round(long_liq_price, 2),
                'distance_pct': round(move_pct * 100, 2),
                'estimated_volume_btc': round(intensity, 2),
                'estimated_value_m': round(intensity * current_price / 1_000_000, 2)
            })
            
            short_liquidations.append({
                'leverage': lev,
                'price': round(short_liq_price, 2),
                'distance_pct': round(move_pct * 100, 2),
                'estimated_volume_btc': round(intensity, 2),
                'estimated_value_m': round(intensity * current_price / 1_000_000, 2)
            })
        
        # Trouver les clusters les plus denses
        nearest_long_liq = long_liquidations[0]  # x125
        nearest_short_liq = short_liquidations[0]  # x125
        
        # Déterminer l'aimant principal
        dist_to_long = abs(current_price - nearest_long_liq['price'])
        dist_to_short = abs(current_price - nearest_short_liq['price'])
        
        if dist_to_short < dist_to_long:
            magnet = {
                'direction': 'UP',
                'target': nearest_short_liq['price'],
                'type': 'SHORT_SQUEEZE',
                'distance_usd': round(dist_to_short, 2),
                'distance_pct': round(dist_to_short / current_price * 100, 3)
            }
        else:
            magnet = {
                'direction': 'DOWN',
                'target': nearest_long_liq['price'],
                'type': 'LONG_FLUSH',
                'distance_usd': round(dist_to_long, 2),
                'distance_pct': round(dist_to_long / current_price * 100, 3)
            }
        
        # Zones de danger (< 1% du prix)
        danger_zones = []
        for liq in long_liquidations + short_liquidations:
            if liq['distance_pct'] < 1.0:
                danger_zones.append(liq)
        
        return {
            'long_liquidations': long_liquidations,
            'short_liquidations': short_liquidations,
            'magnet': magnet,
            'danger_zones': danger_zones,
            'total_oi_btc': round(total_oi, 2),
            'cascade_risk': len(danger_zones) > 2
        }
    
    def _estimate_position_intensity(self, leverage: int, total_oi: float) -> float:
        """
        Estime l'intensité des positions à un niveau de levier donné
        
        Distribution empirique:
        - x100-125: Très peu de positions (dégénérés)
        - x50-75: Modéré
        - x10-25: Beaucoup
        - x5-10: Le plus commun
        """
        # Distribution approximative
        distribution = {
            125: 0.02,
            100: 0.05,
            75: 0.08,
            50: 0.15,
            25: 0.25,
            20: 0.15,
            10: 0.20,
            5: 0.10
        }
        
        return total_oi * distribution.get(leverage, 0.05)
    
    def _analyze_options_market(self, current_price: float) -> Dict[str, Any]:
        """
        Analyse du marché des options
        
        Note: Données simulées car nécessite API Deribit/OKX options
        En production, intégrer avec Deribit API pour données réelles
        """
        # Estimation du Max Pain basée sur des heuristiques
        # Le max pain tend à être proche des niveaux psychologiques
        round_levels = [
            int(current_price / 5000) * 5000,  # Arrondi à 5k
            int(current_price / 1000) * 1000,  # Arrondi à 1k
        ]
        
        # Estimation simple : max pain légèrement en dessous du prix actuel
        estimated_max_pain = round_levels[0]
        if current_price - estimated_max_pain > 2500:
            estimated_max_pain += 5000
        
        # Distance au max pain
        distance_to_max_pain = current_price - estimated_max_pain
        distance_pct = (distance_to_max_pain / current_price) * 100
        
        # Put/Call ratio estimé (corrélé au sentiment)
        # > 1 = bearish, < 1 = bullish, ~0.7-0.8 = neutre haussier
        estimated_pcr = 0.75  # Placeholder
        
        # IV estimation basée sur le funding
        # Funding élevé = IV élevée généralement
        estimated_iv = 45  # Placeholder - normalement entre 30-80%
        
        return {
            'estimated_max_pain': estimated_max_pain,
            'distance_to_max_pain': round(distance_to_max_pain, 2),
            'distance_pct': round(distance_pct, 2),
            'max_pain_gravity': 'PULLING_DOWN' if distance_to_max_pain > 0 else 'PULLING_UP',
            'estimated_put_call_ratio': estimated_pcr,
            'pcr_signal': 'BULLISH' if estimated_pcr < 0.8 else 'BEARISH' if estimated_pcr > 1.2 else 'NEUTRAL',
            'estimated_iv_pct': estimated_iv,
            'iv_environment': 'NORMAL',
            'note': 'Données estimées - Intégrer API Deribit pour données réelles'
        }
    
    def _calculate_derivative_sentiment(self, futures: Dict, funding: Dict, 
                                        liquidations: Dict) -> Dict[str, Any]:
        """Calcule le sentiment global basé sur les dérivés"""
        score = 50  # Neutre par défaut
        factors = []
        
        # Futures basis contribution
        if futures.get('signal') == 'VERY_BULLISH':
            score += 20
            factors.append("+20 Backwardation profonde")
        elif futures.get('signal') == 'BULLISH':
            score += 10
            factors.append("+10 Backwardation légère")
        elif futures.get('signal') == 'BEARISH':
            score -= 15
            factors.append("-15 Contango excessif")
        
        # Funding divergence
        if funding.get('is_significant'):
            if funding.get('lowest', {}).get('rate_pct', 0) < 0:
                score += 10
                factors.append("+10 Funding négatif (short squeeze potential)")
        
        # Liquidation magnet
        magnet = liquidations.get('magnet', {})
        if magnet.get('direction') == 'UP' and magnet.get('distance_pct', 100) < 0.5:
            score += 15
            factors.append("+15 Short squeeze imminent")
        elif magnet.get('direction') == 'DOWN' and magnet.get('distance_pct', 100) < 0.5:
            score -= 15
            factors.append("-15 Long flush imminent")
        
        # Cascade risk
        if liquidations.get('cascade_risk'):
            factors.append("⚠️ Risque de cascade de liquidations")
        
        # Normaliser
        score = max(0, min(100, score))
        
        if score >= 70:
            sentiment = "BULLISH"
            emoji = "🟢"
        elif score >= 55:
            sentiment = "SLIGHTLY_BULLISH"
            emoji = "🟢"
        elif score >= 45:
            sentiment = "NEUTRAL"
            emoji = "⚪"
        elif score >= 30:
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
