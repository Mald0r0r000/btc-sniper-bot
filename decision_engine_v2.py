"""
Decision Engine V2 - Institutional Grade
Système de scoring pondéré multi-dimensionnel avec intelligence combinée

Architecture:
- 8 dimensions d'analyse
- Scoring pondéré configurable
- Détection de manipulation intégrée (réducteur de confiance)
- Signaux avec confiance calibrée
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import config


class SignalType(Enum):
    """Types de signaux générés"""
    # Signaux haute confiance
    QUANTUM_BUY = "QUANTUM_BUY"
    QUANTUM_SELL = "QUANTUM_SELL"
    
    # Signaux techniques
    SHORT_SNIPER = "SHORT_SNIPER"
    LONG_SNIPER = "LONG_SNIPER"
    LONG_BREAKOUT = "LONG_BREAKOUT"
    SHORT_BREAKOUT = "SHORT_BREAKOUT"
    
    # Signaux de range
    FADE_HIGH = "FADE_HIGH"
    FADE_LOW = "FADE_LOW"
    
    # Signaux dérivés
    SHORT_SQUEEZE = "SHORT_SQUEEZE"
    LONG_FLUSH = "LONG_FLUSH"
    DIAMOND_SETUP = "DIAMOND_SETUP"
    
    # Signaux macro
    MACRO_ALIGNED_LONG = "MACRO_ALIGNED_LONG"
    MACRO_ALIGNED_SHORT = "MACRO_ALIGNED_SHORT"
    
    # Signaux contrarian
    CONTRARIAN_BUY = "CONTRARIAN_BUY"
    CONTRARIAN_SELL = "CONTRARIAN_SELL"
    
    # Pas de signal
    NO_SIGNAL = "NO_SIGNAL"


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class CompositeSignal:
    """Signal composite avec toutes les métadonnées"""
    type: SignalType
    direction: SignalDirection
    confidence: float  # 0-100
    raw_score: float  # Score avant ajustements
    adjusted_score: float  # Score après pénalités
    emoji: str
    description: str
    reasons: List[str]
    targets: Dict[str, float]
    warnings: List[str]
    dimension_scores: Dict[str, float]
    manipulation_penalty: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'direction': self.direction.value,
            'confidence': round(self.confidence, 1),
            'raw_score': round(self.raw_score, 1),
            'adjusted_score': round(self.adjusted_score, 1),
            'emoji': self.emoji,
            'description': self.description,
            'reasons': self.reasons,
            'targets': self.targets,
            'warnings': self.warnings,
            'dimension_scores': {k: round(v, 2) for k, v in self.dimension_scores.items()},
            'manipulation_penalty': round(self.manipulation_penalty, 2)
        }


class DecisionEngineV2:
    """
    Moteur de décision v2 - Scoring multi-dimensionnel pondéré
    
    Dimensions:
    1. Technical (Order Book, CVD, Volume Profile) - 25%
    2. Structure (FVG, Entropy, Pivots) - 15%
    3. Multi-Exchange (Arbitrage, Funding divergence) - 15%
    4. Manipulation (Spoofing detection) - Réducteur
    5. Derivatives (Options, Futures basis) - 15%
    6. On-Chain (Whale activity, Exchange flow) - 10%
    7. Sentiment (Fear/Greed, Social) - 10%
    8. Macro (DXY, SPX correlation) - 10%
    """
    
    # Configuration des poids (total = 100)
    WEIGHT_CONFIG = {
        'technical': 25,
        'structure': 15,
        'multi_exchange': 15,
        'derivatives': 15,
        'onchain': 10,
        'sentiment': 10,
        'macro': 10
    }
    
    # Seuils de confiance
    CONFIDENCE_THRESHOLDS = {
        'STRONG': 80,
        'MODERATE': 60,
        'WEAK': 40,
        'NO_TRADE': 0
    }
    
    def __init__(
        self,
        current_price: float,
        # Données core
        order_book_data: Dict = None,
        cvd_data: Dict = None,
        volume_profile_data: Dict = None,
        # Données structure
        fvg_data: Dict = None,
        entropy_data: Dict = None,
        # Données multi-exchange
        multi_exchange_data: Dict = None,
        # Données manipulation
        spoofing_data: Dict = None,
        # Données dérivés
        derivatives_data: Dict = None,
        # Données on-chain
        onchain_data: Dict = None,
        # Données sentiment
        sentiment_data: Dict = None,
        # Données macro
        macro_data: Dict = None,
        # Open Interest
        open_interest: Dict = None
    ):
        self.price = current_price
        
        # Core data
        self.ob = order_book_data or {}
        self.cvd = cvd_data or {}
        self.vp = volume_profile_data or {}
        
        # Structure data
        self.fvg = fvg_data or {}
        self.entropy = entropy_data or {}
        
        # Advanced data
        self.multi_ex = multi_exchange_data or {}
        self.spoofing = spoofing_data or {}
        self.derivatives = derivatives_data or {}
        self.onchain = onchain_data or {}
        self.sentiment = sentiment_data or {}
        self.macro = macro_data or {}
        self.oi = open_interest or {}
    
    def generate_composite_signal(self) -> Dict[str, Any]:
        """
        Génère un signal composite basé sur toutes les dimensions
        
        Returns:
            Dict avec signal principal, contexte et métriques
        """
        # 1. Calculer les scores par dimension
        dimension_scores = self._calculate_dimension_scores()
        
        # 2. Calculer le score composite
        raw_score, weighted_scores = self._calculate_composite_score(dimension_scores)
        
        # 3. Appliquer les pénalités de manipulation
        manipulation_penalty = self._calculate_manipulation_penalty()
        adjusted_score = max(0, raw_score - manipulation_penalty)
        
        # 4. Déterminer la direction
        direction = self._determine_direction(dimension_scores)
        
        # 5. Sélectionner le type de signal
        signal_type = self._select_signal_type(dimension_scores, direction)
        
        # 6. Générer les détails du signal
        signal = self._build_composite_signal(
            signal_type, direction, raw_score, adjusted_score,
            dimension_scores, manipulation_penalty
        )
        
        # 7. Contexte de marché
        market_context = self._build_market_context(dimension_scores)
        
        return {
            'primary_signal': signal.to_dict(),
            'dimension_scores': {k: round(v, 2) for k, v in dimension_scores.items()},
            'weighted_scores': weighted_scores,
            'composite_score': round(adjusted_score, 1),
            'manipulation_penalty': round(manipulation_penalty, 2),
            'market_context': market_context,
            'signal_strength': self._get_signal_strength(adjusted_score),
            'tradeable': adjusted_score >= self.CONFIDENCE_THRESHOLDS['WEAK']
        }
    
    def _calculate_dimension_scores(self) -> Dict[str, float]:
        """Calcule les scores pour chaque dimension (0-100)"""
        scores = {}
        
        # 1. Technical Score (Order Book + CVD + Volume Profile)
        scores['technical'] = self._score_technical()
        
        # 2. Structure Score (FVG + Entropy)
        scores['structure'] = self._score_structure()
        
        # 3. Multi-Exchange Score
        scores['multi_exchange'] = self._score_multi_exchange()
        
        # 4. Derivatives Score
        scores['derivatives'] = self._score_derivatives()
        
        # 5. On-Chain Score
        scores['onchain'] = self._score_onchain()
        
        # 6. Sentiment Score
        scores['sentiment'] = self._score_sentiment()
        
        # 7. Macro Score
        scores['macro'] = self._score_macro()
        
        return scores
    
    def _score_technical(self) -> float:
        """Score technique (50 = neutre)"""
        score = 50.0
        
        # Order Book imbalance
        bid_ratio = self.ob.get('bid_ratio_pct', 50)
        if bid_ratio > 60:
            score += (bid_ratio - 50) * 0.4  # Max +20
        elif bid_ratio < 40:
            score -= (50 - bid_ratio) * 0.4  # Max -20
        
        # CVD
        agg_ratio = self.cvd.get('aggression_ratio', 1.0)
        if agg_ratio > 1.2:
            score += 15
        elif agg_ratio < 0.8:
            score -= 15
        
        # Volume Profile position
        vp_shape = self.vp.get('shape', 'D-Shape')
        if vp_shape == 'P-Shape':  # Bullish
            score += 10
        elif vp_shape == 'b-Shape':  # Bearish
            score -= 10
        
        return max(0, min(100, score))
    
    def _score_structure(self) -> float:
        """Score structure (FVG + Entropy)"""
        score = 50.0
        
        # Entropy / Quantum State
        compression = self.entropy.get('compression', {})
        if compression.get('is_low_entropy'):
            score += 10  # Potentiel d'explosion
        
        signals = self.entropy.get('signals', {})
        if signals.get('quantum_buy'):
            score += 25
        elif signals.get('quantum_sell'):
            score -= 25
        
        # FVG actifs proches
        nearest_bull = self.fvg.get('nearest_bull')
        nearest_bear = self.fvg.get('nearest_bear')
        
        if nearest_bull and not nearest_bull.get('mitigated'):
            distance = abs(nearest_bull.get('distance_pct', 100))
            if distance < 0.3:  # Très proche
                score += 10
        
        if nearest_bear and not nearest_bear.get('mitigated'):
            distance = abs(nearest_bear.get('distance_pct', 100))
            if distance < 0.3:
                score -= 10
        
        return max(0, min(100, score))
    
    def _score_multi_exchange(self) -> float:
        """Score multi-exchange"""
        score = 50.0
        
        if not self.multi_ex:
            return score
        
        # Funding divergence
        funding = self.multi_ex.get('funding_analysis', {})
        if funding.get('signal') == 'SHORTS_PAYING':
            score += 15  # Short squeeze potential
        elif funding.get('signal') == 'LONGS_EXPENSIVE':
            score -= 10
        
        # Arbitrage opportunity
        arb = self.multi_ex.get('arbitrage', {})
        if arb.get('opportunity'):
            spread_pct = arb.get('spread_pct', 0)
            if spread_pct > 0.01:
                score += 5  # Légèrement bullish si arb vers le haut
        
        # Global order book imbalance
        global_ob = self.multi_ex.get('global_orderbook', {})
        imbalance = global_ob.get('imbalance_pct', 50)
        if imbalance > 55:
            score += (imbalance - 50) * 0.3
        elif imbalance < 45:
            score -= (50 - imbalance) * 0.3
        
        return max(0, min(100, score))
    
    def _score_derivatives(self) -> float:
        """Score dérivés"""
        score = 50.0
        
        if not self.derivatives:
            return score
        
        sentiment = self.derivatives.get('sentiment', {})
        deriv_score = sentiment.get('score', 50)
        
        # Mapper sur notre échelle
        score = deriv_score
        
        return max(0, min(100, score))
    
    def _score_onchain(self) -> float:
        """Score on-chain"""
        score = 50.0
        
        if not self.onchain:
            return score
        
        onchain_score = self.onchain.get('score', {})
        if onchain_score:
            score = onchain_score.get('value', 50)
        
        return max(0, min(100, score))
    
    def _score_sentiment(self) -> float:
        """Score sentiment"""
        score = 50.0
        
        if not self.sentiment:
            return score
        
        fg = self.sentiment.get('fear_greed', {})
        fg_value = fg.get('value', 50)
        
        # Utiliser le signal contrarian pour le score
        # Extreme fear = bullish, Extreme greed = bearish
        if fg_value < 25:
            score = 70  # Contrarian bullish
        elif fg_value < 40:
            score = 60
        elif fg_value > 75:
            score = 30  # Contrarian bearish
        elif fg_value > 60:
            score = 40
        else:
            score = 50
        
        return score
    
    def _score_macro(self) -> float:
        """Score macro"""
        score = 50.0
        
        if not self.macro:
            return score
        
        risk_env = self.macro.get('risk_environment', {})
        risk_score = risk_env.get('risk_score', 50)
        
        # Risk-on = bullish for BTC
        score = risk_score
        
        return max(0, min(100, score))
    
    def _calculate_composite_score(self, dimension_scores: Dict[str, float]) -> Tuple[float, Dict]:
        """Calcule le score composite pondéré"""
        weighted_scores = {}
        total_weight = sum(self.WEIGHT_CONFIG.values())
        
        composite = 0.0
        for dim, weight in self.WEIGHT_CONFIG.items():
            dim_score = dimension_scores.get(dim, 50)
            weighted = (dim_score * weight) / total_weight
            weighted_scores[dim] = round(weighted, 2)
            composite += weighted
        
        return composite, weighted_scores
    
    def _calculate_manipulation_penalty(self) -> float:
        """Calcule la pénalité de manipulation"""
        if not self.spoofing:
            return 0.0
        
        manipulation_score = self.spoofing.get('manipulation_score', 0)
        
        # Pénalité proportionnelle (max 30 points)
        penalty = manipulation_score * 30
        
        return penalty
    
    def _determine_direction(self, scores: Dict[str, float]) -> SignalDirection:
        """Détermine la direction globale"""
        avg_score = np.mean(list(scores.values()))
        
        if avg_score >= 55:
            return SignalDirection.LONG
        elif avg_score <= 45:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def _select_signal_type(self, scores: Dict[str, float], direction: SignalDirection) -> SignalType:
        """Sélectionne le type de signal le plus approprié"""
        # Vérifier les signaux Quantum en premier (haute priorité)
        entropy_signals = self.entropy.get('signals', {})
        if entropy_signals.get('quantum_buy'):
            return SignalType.QUANTUM_BUY
        if entropy_signals.get('quantum_sell'):
            return SignalType.QUANTUM_SELL
        
        # Vérifier les squeeze imminents
        deriv_liq = self.derivatives.get('liquidations', {})
        magnet = deriv_liq.get('magnet', {})
        if magnet.get('distance_pct', 100) < 0.3:
            if magnet.get('direction') == 'UP':
                return SignalType.SHORT_SQUEEZE
            else:
                return SignalType.LONG_FLUSH
        
        # Signaux techniques (Sniper)
        vah = self.vp.get('vah', 0)
        val = self.vp.get('val', 0)
        
        near_vah = abs(self.price - vah) / vah < 0.003 if vah > 0 else False
        near_val = abs(self.price - val) / val < 0.003 if val > 0 else False
        
        wall_ask_danger = self.ob.get('wall_ask', {}).get('is_danger', False)
        wall_bid_danger = self.ob.get('wall_bid', {}).get('is_danger', False)
        
        if near_vah and wall_ask_danger and self.cvd.get('net_cvd', 0) < 0:
            return SignalType.SHORT_SNIPER
        
        if near_val and wall_bid_danger and self.cvd.get('net_cvd', 0) > 0:
            return SignalType.LONG_SNIPER
        
        # Breakouts
        if near_vah and self.cvd.get('net_cvd', 0) > 50:
            return SignalType.LONG_BREAKOUT
        
        if near_val and self.cvd.get('net_cvd', 0) < -50:
            return SignalType.SHORT_BREAKOUT
        
        # Fade setups
        if self.vp.get('shape') == 'D-Shape':
            if near_vah:
                return SignalType.FADE_HIGH
            if near_val:
                return SignalType.FADE_LOW
        
        # Signaux contrarian (sentiment extrême)
        fg_value = self.sentiment.get('fear_greed', {}).get('value', 50)
        if fg_value < 20:
            return SignalType.CONTRARIAN_BUY
        if fg_value > 80:
            return SignalType.CONTRARIAN_SELL
        
        # Signaux macro-alignés
        macro_signal = self.macro.get('btc_impact', {}).get('signal', '')
        if 'BULLISH' in macro_signal and direction == SignalDirection.LONG:
            return SignalType.MACRO_ALIGNED_LONG
        if 'BEARISH' in macro_signal and direction == SignalDirection.SHORT:
            return SignalType.MACRO_ALIGNED_SHORT
        
        return SignalType.NO_SIGNAL
    
    def _build_composite_signal(
        self, signal_type: SignalType, direction: SignalDirection,
        raw_score: float, adjusted_score: float,
        dimension_scores: Dict[str, float], manipulation_penalty: float
    ) -> CompositeSignal:
        """Construit le signal composite final"""
        
        # Descriptions par type
        descriptions = {
            SignalType.QUANTUM_BUY: "⚛️ QUANTUM BUY - Compression + Sweep + Volume",
            SignalType.QUANTUM_SELL: "⚛️ QUANTUM SELL - Compression + Sweep + Volume",
            SignalType.SHORT_SNIPER: "🎯 SHORT SNIPER - VAH + Mur + CVD Divergence",
            SignalType.LONG_SNIPER: "🎯 LONG SNIPER - VAL + Mur + CVD Absorption",
            SignalType.LONG_BREAKOUT: "🚀 LONG BREAKOUT - Agression VAH",
            SignalType.SHORT_BREAKOUT: "💥 SHORT BREAKDOWN - Agression VAL",
            SignalType.FADE_HIGH: "📉 FADE HIGH - Haut de Range",
            SignalType.FADE_LOW: "📈 FADE LOW - Bas de Range",
            SignalType.SHORT_SQUEEZE: "⚡ SHORT SQUEEZE IMMINENT",
            SignalType.LONG_FLUSH: "⚡ LONG FLUSH IMMINENT",
            SignalType.DIAMOND_SETUP: "💎 DIAMOND SETUP - Funding Négatif",
            SignalType.CONTRARIAN_BUY: "🆘 CONTRARIAN BUY - Peur Extrême",
            SignalType.CONTRARIAN_SELL: "🔔 CONTRARIAN SELL - Euphorie Extrême",
            SignalType.MACRO_ALIGNED_LONG: "🌍 MACRO ALIGNED LONG",
            SignalType.MACRO_ALIGNED_SHORT: "🌍 MACRO ALIGNED SHORT",
            SignalType.NO_SIGNAL: "💤 Pas de signal clair"
        }
        
        emojis = {
            SignalType.QUANTUM_BUY: "⚛️🟢",
            SignalType.QUANTUM_SELL: "⚛️🔴",
            SignalType.SHORT_SNIPER: "🎯🔴",
            SignalType.LONG_SNIPER: "🎯🟢",
            SignalType.LONG_BREAKOUT: "🚀",
            SignalType.SHORT_BREAKOUT: "💥",
            SignalType.SHORT_SQUEEZE: "⚡📈",
            SignalType.LONG_FLUSH: "⚡📉",
            SignalType.NO_SIGNAL: "💤"
        }
        
        # Générer les raisons
        reasons = self._generate_reasons(dimension_scores, signal_type)
        
        # Générer les warnings
        warnings = self._generate_warnings(manipulation_penalty)
        
        # Générer les targets
        targets = self._generate_targets(signal_type, direction)
        
        return CompositeSignal(
            type=signal_type,
            direction=direction,
            confidence=adjusted_score,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            emoji=emojis.get(signal_type, "📊"),
            description=descriptions.get(signal_type, "Signal"),
            reasons=reasons,
            targets=targets,
            warnings=warnings,
            dimension_scores=dimension_scores,
            manipulation_penalty=manipulation_penalty
        )
    
    def _generate_reasons(self, scores: Dict[str, float], signal_type: SignalType) -> List[str]:
        """Génère les raisons du signal"""
        reasons = []
        
        # Top 3 dimensions les plus fortes
        sorted_dims = sorted(scores.items(), key=lambda x: abs(x[1] - 50), reverse=True)[:3]
        
        for dim, score in sorted_dims:
            if score > 60:
                reasons.append(f"✅ {dim.title()}: {score:.0f}/100 (Bullish)")
            elif score < 40:
                reasons.append(f"❌ {dim.title()}: {score:.0f}/100 (Bearish)")
        
        # Raisons spécifiques au type
        if signal_type in [SignalType.QUANTUM_BUY, SignalType.QUANTUM_SELL]:
            reasons.append("Low Entropy + Volume Spike détecté")
        
        if signal_type in [SignalType.SHORT_SQUEEZE, SignalType.LONG_FLUSH]:
            reasons.append("Cluster de liquidation très proche")
        
        return reasons
    
    def _generate_warnings(self, manipulation_penalty: float) -> List[str]:
        """Génère les avertissements"""
        warnings = []
        
        if manipulation_penalty > 15:
            warnings.append(f"⚠️ Manipulation détectée (pénalité: -{manipulation_penalty:.0f})")
        
        # Ghost walls
        ghost = self.spoofing.get('ghost_walls', {})
        if ghost.get('detected'):
            warnings.append(f"👻 {ghost.get('count', 0)} mur(s) fantôme détecté(s)")
        
        # Layering
        layering = self.spoofing.get('layering', {})
        if layering.get('detected'):
            warnings.append("📊 Pattern de layering détecté")
        
        # Wash trading
        wash = self.spoofing.get('wash_trading', {})
        if wash.get('detected'):
            warnings.append(f"🔄 Wash trading probable ({wash.get('probability', 0)*100:.1f}%)")
        
        return warnings
    
    def _generate_targets(self, signal_type: SignalType, direction: SignalDirection) -> Dict[str, float]:
        """Génère les targets"""
        targets = {}
        
        poc = self.vp.get('poc', 0)
        vah = self.vp.get('vah', 0)
        val = self.vp.get('val', 0)
        
        if direction == SignalDirection.LONG:
            if poc > 0:
                targets['tp1'] = poc
            if vah > 0:
                targets['tp2'] = vah
            if val > 0:
                targets['sl'] = val * 0.995
        elif direction == SignalDirection.SHORT:
            if poc > 0:
                targets['tp1'] = poc
            if val > 0:
                targets['tp2'] = val
            if vah > 0:
                targets['sl'] = vah * 1.005
        
        return targets
    
    def _build_market_context(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Construit le contexte de marché"""
        return {
            'price': self.price,
            'technical_bias': 'BULLISH' if scores.get('technical', 50) > 55 else 'BEARISH' if scores.get('technical', 50) < 45 else 'NEUTRAL',
            'structure_bias': 'BULLISH' if scores.get('structure', 50) > 55 else 'BEARISH' if scores.get('structure', 50) < 45 else 'NEUTRAL',
            'derivatives_bias': 'BULLISH' if scores.get('derivatives', 50) > 55 else 'BEARISH' if scores.get('derivatives', 50) < 45 else 'NEUTRAL',
            'sentiment_bias': 'BULLISH' if scores.get('sentiment', 50) > 55 else 'BEARISH' if scores.get('sentiment', 50) < 45 else 'NEUTRAL',
            'macro_bias': 'BULLISH' if scores.get('macro', 50) > 55 else 'BEARISH' if scores.get('macro', 50) < 45 else 'NEUTRAL',
            'manipulation_risk': self.spoofing.get('risk_level', 'UNKNOWN'),
            'vp_shape': self.vp.get('shape', 'D-Shape'),
            'quantum_state': self.entropy.get('quantum_state', 'UNKNOWN'),
            'fear_greed': self.sentiment.get('fear_greed', {}).get('value', 50)
        }
    
    def _get_signal_strength(self, score: float) -> str:
        """Retourne la force du signal"""
        if score >= self.CONFIDENCE_THRESHOLDS['STRONG']:
            return "STRONG"
        elif score >= self.CONFIDENCE_THRESHOLDS['MODERATE']:
            return "MODERATE"
        elif score >= self.CONFIDENCE_THRESHOLDS['WEAK']:
            return "WEAK"
        else:
            return "NO_TRADE"
