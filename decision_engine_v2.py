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
from datetime import datetime
import numpy as np

import config
from analyzers.liquidation_zones import LiquidationZoneAnalyzer
from momentum_analyzer import MomentumAnalyzer, MomentumStrength
from smart_entry import SmartEntryAnalyzer, EntryStrategy



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
    
    # Configuration des poids par style de trading
    WEIGHT_CONFIGS = {
        # Style par défaut (mixte)
        'default': {
            'technical': 25,
            'structure': 15,
            'multi_exchange': 15,
            'derivatives': 15,
            'onchain': 10,
            'sentiment': 10,
            'macro': 10
        },
        # Intraday/Swing avec levier - Focus sur structure et flow
        'swing': {
            'technical': 20,     # Légèrement réduit
            'structure': 20,     # Augmenté (support/résistance crucial)
            'multi_exchange': 5, # Réduit (moins critique pour swing)
            'derivatives': 15,   # Inchangé (funding + OI important)
            'onchain': 15,       # Augmenté (whale tracking)
            'sentiment': 10,     # Inchangé
            'macro': 15          # Augmenté (direction générale)
        },
        # Scalping - Focus sur order flow
        'scalp': {
            'technical': 35,
            'structure': 10,
            'multi_exchange': 20,
            'derivatives': 15,
            'onchain': 5,
            'sentiment': 5,
            'macro': 10
        }
    }
    
    # Chemin du fichier de poids adaptatifs
    ADAPTIVE_WEIGHTS_FILE = 'adaptive_weights.json'
    
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
        kdj_data: Dict = None,
        adx_data: Dict = None,
        htf_data: Dict = None,
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
        open_interest: Dict = None,
        # Options Deribit
        options_data: Dict = None,
        # Style de trading
        trading_style: str = 'swing',
        # Bonus/malus de consistency
        consistency_bonus: int = 0,
        # Full consistency data for quality filters
        consistency_data: Dict = None,
        # Candles pour liquidation zones et MTF targets
        candles_5m: List[Dict] = None,
        candles_1h: List[Dict] = None,
        candles_4h: List[Dict] = None,
        # Fluid Dynamics (Venturi et Self-Trading)
        venturi_data: Dict = None,
        self_trading_data: Dict = None,
        # Hyperliquid whale data
        hyperliquid_data: Dict = None
    ):
        self.price = current_price
        self.trading_style = trading_style
        self.consistency_bonus = consistency_bonus
        # Store full consistency data for quality filters
        self.consistency_data = consistency_data or {}
        self.consistency_status = self.consistency_data.get('status', 'UNKNOWN')
        self.consistency_score = self.consistency_data.get('confidence_trend', 0)
        
        # Sélectionner les poids selon le style
        if trading_style == 'adaptive':
            # Charger les poids adaptatifs depuis le fichier JSON
            self.WEIGHT_CONFIG = self._load_adaptive_weights()
        else:
            self.WEIGHT_CONFIG = self.WEIGHT_CONFIGS.get(
                trading_style, 
                self.WEIGHT_CONFIGS['default']
            )
        
        # Core data
        self.ob = order_book_data or {}
        self.cvd = cvd_data or {}
        self.vp = volume_profile_data or {}
        
        # Structure data
        self.fvg = fvg_data or {}
        self.entropy = entropy_data or {}
        self.kdj = kdj_data or {}
        self.adx = adx_data or {}
        self.htf = htf_data or {}
        
        # Advanced data
        self.multi_ex = multi_exchange_data or {}
        self.spoofing = spoofing_data or {}
        self.derivatives = derivatives_data or {}
        self.onchain = onchain_data or {}
        self.sentiment = sentiment_data or {}
        self.macro = macro_data or {}
        self.oi = open_interest or {}
        self.options = options_data or {}
        self.candles_5m = candles_5m or []
        self.candles_1h = candles_1h or []
        self.candles_4h = candles_4h or []
        
        # Liquidation Zone Analyzer
        self.liq_analyzer = LiquidationZoneAnalyzer()
        self.liq_analysis = None
        
        # Momentum Analyzer (MTF Fractals)
        self.momentum_analyzer = MomentumAnalyzer()
        self.momentum_result = None
        
        # Smart Entry Analyzer (Using 1h candles for robustness)
        self.smart_entry_analyzer = SmartEntryAnalyzer()


        
        # Fluid Dynamics data
        self.venturi = venturi_data or {}
        self.self_trading = self_trading_data or {}
        
        # Hyperliquid data (whale tracking)
        self.hyperliquid = hyperliquid_data or {}
    
    def _load_adaptive_weights(self) -> Dict[str, int]:
        """
        Charge les poids adaptatifs depuis un fichier JSON ou le Gist
        Priorité: 1) Fichier local  2) Gist  3) Poids par défaut
        """
        import json
        import os
        
        # 1. Essayer le fichier local d'abord
        try:
            if os.path.exists(self.ADAPTIVE_WEIGHTS_FILE):
                with open(self.ADAPTIVE_WEIGHTS_FILE, 'r') as f:
                    data = json.load(f)
                    weights = data.get('weights', {})
                    if weights:
                        print(f"   🧠 Poids adaptatifs chargés (local): {weights}")
                        return weights
        except Exception as e:
            print(f"   ⚠️ Erreur chargement poids locaux: {e}")
        
        # 2. Essayer le Gist (pour GitHub Actions)
        try:
            from data_store import GistDataStore
            gist_store = GistDataStore()
            gist_data = gist_store.load_adaptive_weights()
            if gist_data and gist_data.get('weights'):
                weights = gist_data['weights']
                print(f"   🧠 Poids adaptatifs chargés (Gist): {weights}")
                # Sauvegarder localement pour le prochain accès
                with open(self.ADAPTIVE_WEIGHTS_FILE, 'w') as f:
                    json.dump(gist_data, f, indent=2)
                return weights
        except Exception as e:
            print(f"   ⚠️ Erreur chargement poids Gist: {e}")
        
        # 3. Fallback sur poids par défaut
        print("   📊 Utilisation des poids par défaut")
        return self.WEIGHT_CONFIGS['default']
    
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
        
        # 4. Appliquer le bonus/malus de consistency
        adjusted_score = max(0, min(100, adjusted_score + self.consistency_bonus))
        
        # 4b. Appliquer les modifiers Fluid Dynamics
        venturi_modifier = self.venturi.get('signal_modifier', 0)
        self_trading_modifier = self.self_trading.get('signal_modifier', 0)
        fluid_dynamics_modifier = venturi_modifier + self_trading_modifier
        adjusted_score = max(0, min(100, adjusted_score + fluid_dynamics_modifier))
        
        # 4c. Appliquer le modifier Hyperliquid Whale Sentiment
        whale_modifier = self.hyperliquid.get('signal_modifier', 0)
        adjusted_score = max(0, min(100, adjusted_score + whale_modifier))
        
        # 5. Déterminer la direction initiale (basée sur le score)
        direction = self._determine_direction(dimension_scores)
        
        # 6. Sélectionner le type de signal
        signal_type = self._select_signal_type(dimension_scores, direction)
        
        # 6a. CORRECTION: Force direction based on signal type logic
        # Some signals have inherent direction that overrides the general score bias
        if signal_type == SignalType.FADE_HIGH:
            direction = SignalDirection.SHORT
        elif signal_type == SignalType.FADE_LOW:
            direction = SignalDirection.LONG
        elif signal_type == SignalType.SHORT_SNIPER:
            direction = SignalDirection.SHORT
        elif signal_type == SignalType.LONG_SNIPER:
            direction = SignalDirection.LONG
        elif signal_type == SignalType.SHORT_BREAKOUT:
            direction = SignalDirection.SHORT
        elif signal_type == SignalType.LONG_BREAKOUT:
            direction = SignalDirection.LONG
        elif signal_type == SignalType.QUANTUM_SELL:
            direction = SignalDirection.SHORT
        elif signal_type == SignalType.QUANTUM_BUY:
            direction = SignalDirection.LONG
        elif signal_type == SignalType.SHORT_SQUEEZE:
            direction = SignalDirection.LONG # Squeeze UP implies LONG
        elif signal_type == SignalType.LONG_FLUSH:
            direction = SignalDirection.SHORT # Flush DOWN implies SHORT
        
        # 6b. QUALITY FILTERS (validated via backtesting: +6.2% WR, +$1590 P&L)
        # Filter 1: Skip LONG_BREAKOUT (0% WR in historical backtests)
        if signal_type == SignalType.LONG_BREAKOUT:
            signal_type = SignalType.NO_SIGNAL
            direction = SignalDirection.NEUTRAL
        
        # Filter 2: Downgrade signals with NEUTRAL consistency + declining confidence
        if self.consistency_status == 'NEUTRAL' and self.consistency_score < -15:
            # Check if it's a tradeable signal being downgraded
            if signal_type not in [SignalType.NO_SIGNAL]:
                signal_type = SignalType.NO_SIGNAL
                direction = SignalDirection.NEUTRAL
        
        # ========== PHASE 1: INSTITUTIONAL GRADE FILTERS ==========
        
        # Filter 3: ADX Market Regime (Block signals in dead/ranging markets)
        adx_regime = self.adx.get('regime', 'UNKNOWN')
        if adx_regime == 'RANGING' and signal_type not in [SignalType.NO_SIGNAL]:
            # Market is dead, no clear trend - block all signals
            signal_type = SignalType.NO_SIGNAL
            direction = SignalDirection.NEUTRAL
        
        # Filter 4: HTF Alignment (Reduce confidence if trading against major trend)
        htf_bias = self.htf.get('bias', 'NEUTRAL')
        htf_penalty = 0
        if htf_bias != 'NEUTRAL' and signal_type not in [SignalType.NO_SIGNAL]:
            # Check alignment
            is_long_signal = direction == SignalDirection.LONG
            is_htf_bullish = htf_bias == 'BULLISH'
            
            if is_long_signal and not is_htf_bullish:
                htf_penalty = 20  # Going long against bearish HTF
            elif not is_long_signal and is_htf_bullish:
                htf_penalty = 20  # Going short against bullish HTF
                
            adjusted_score -= htf_penalty
        
        # Filter 5: Confluence Check (Require 3+ dimensions aligned)
        if signal_type not in [SignalType.NO_SIGNAL]:
            bullish_dims = sum(1 for score in dimension_scores.values() if score > 55)
            bearish_dims = sum(1 for score in dimension_scores.values() if score < 45)
            
            if direction == SignalDirection.LONG and bullish_dims < 3:
                signal_type = SignalType.NO_SIGNAL
                direction = SignalDirection.NEUTRAL
            elif direction == SignalDirection.SHORT and bearish_dims < 3:
                signal_type = SignalType.NO_SIGNAL
                direction = SignalDirection.NEUTRAL
        
        # 6c. Générer les détails du signal
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
            
        # KDJ Momentum (Oscillator)
        # Updated Logic: Parabolic Reversal & Slope
        kdj_signal = self.kdj.get('signal', 'NEUTRAL')
        
        if kdj_signal == 'PARABOLIC_BULL':
            score += 25  # Strong reversal signal
        elif kdj_signal == 'PARABOLIC_BEAR':
            score -= 25  # Strong reversal signal
        elif kdj_signal == 'GOLDEN_CROSS': # Fallback if standard logic persists
             score += 5
        elif kdj_signal == 'DEAD_CROSS':
             score -= 5
             
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
        
        # --- BLACK BOX RECORDER (Data Logging) ---
        # Capture raw module states for Pattern Discovery
        # Use getattr with defaults to avoid AttributeError on missing attributes
        momentum_result = getattr(self, 'momentum_result', None)
        volatility_data = getattr(self, 'volatility', {})
        structure_data = getattr(self, 'structure', {})
        sentiment_data = getattr(self, 'sentiment', {})
        oi_data = getattr(self, 'oi', {})
        
        snapshot = {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'price': getattr(self, 'price', 0),
            'scores': {
                'technical': dimension_scores.get('technical', 50),
                'structure': dimension_scores.get('structure', 50),
                'sentiment': dimension_scores.get('sentiment', 50),
                'onchain': dimension_scores.get('onchain', 50),
                'macro': dimension_scores.get('macro', 50),
                'derivatives': dimension_scores.get('derivatives', 50)
            },
            'momentum': {
                'score': momentum_result.score if momentum_result else 50,
                'strength': momentum_result.strength.value if momentum_result else 'UNKNOWN',
                'direction': momentum_result.direction if momentum_result else 'UNKNOWN'
            },
            'volatility': volatility_data.get('value', 0) if isinstance(volatility_data, dict) else 0,
            'structure_metrics': {
                'fvg_distance': structure_data.get('fvg_distance', 0) if isinstance(structure_data, dict) else 0,
                'support_proximity': structure_data.get('support_proximity', 0) if isinstance(structure_data, dict) else 0
            },
            'sentiment_metrics': {
                'fear_greed': sentiment_data.get('fear_greed', {}).get('value', 50) if isinstance(sentiment_data, dict) else 50,
                'oi_change': oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0
            }
        }
        # Inject into targets as metadata
        targets['_analysis_snapshot'] = snapshot
        
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
        
        # --- R&D: SMART ENTRY INTEGRATION ---
        # Calculate Smart Entry (using 1H candles for robustness as per Backtest)
        if self.candles_1h and signal_type != SignalType.NO_SIGNAL:
             smart_entry_data = self._analyze_smart_entry(signal)
             if smart_entry_data:
                 # Enrich signal with smart entry data
                 signal.reasons.append(f"🎯 Smart Entry: {smart_entry_data['strategy_text']}")

        return signal
        

    
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
        """
        Génère les targets (TP/SL) basés sur les zones de liquidation
        
        Priorité:
        1. Zones de liquidation (aimants à prix)
        2. Volume Profile (fallback)
        3. Pourcentage fixe (dernier recours)
        
        LONG: TP = zones de liq des shorts au-dessus, SL = zone de liq des longs en-dessous
        SHORT: TP = zones de liq des longs en-dessous, SL = zone de liq des shorts au-dessus
        """
        targets = {}
        current_price = self.price
        direction_str = 'LONG' if direction == SignalDirection.LONG else 'SHORT'
        
        # 0. NEW: Analyze momentum to select optimal timeframe for targets
        try:
            self.momentum_result = self.momentum_analyzer.analyze(
                cvd_data=self.cvd,
                oi_data=self.oi,
                candles=self.candles_1h if self.candles_1h else self.candles_5m,
                direction_hint=direction_str
            )
            
            # If we have MTF candles, try to get fractal-based targets
            if self.candles_1h and self.candles_4h:
                fractal_targets = self.momentum_analyzer.get_fractal_targets(
                    candles_5m=self.candles_5m,
                    candles_1h=self.candles_1h,
                    candles_4h=self.candles_4h,
                    direction=direction_str,
                    momentum_strength=self.momentum_result.strength,
                    current_price=current_price
                )
                
                # Store timeframe info for notifications
                targets['_momentum_score'] = self.momentum_result.score
                targets['_momentum_tf'] = fractal_targets.get('_timeframe', '1h')
                
                # --- HYBRID STRATEGY: FADE SCALPING ---
                # Only apply Scalping (Close targets) if Momentum is WEAK and Signal is FADE
                # This matches the +141% Backtest Strategy
                is_fade = 'FADE' in str(signal_type.name)
                is_weak_momentum = self.momentum_result.strength == MomentumStrength.WEAK
                
                can_use_fractals = False
                if is_fade and is_weak_momentum:
                    # Hybrid Strategy: Scalp on FADE + WEAK Momentum (High precision)
                    can_use_fractals = True
                
                # Apply fractal targets ONLY if conditions met
                if can_use_fractals:
                    if fractal_targets.get('tp1'): targets['tp1'] = fractal_targets['tp1']
                    if fractal_targets.get('tp2'): targets['tp2'] = fractal_targets['tp2']
                    if fractal_targets.get('sl'): targets['sl'] = fractal_targets['sl']
                    targets['_scalp_mode'] = True

        except Exception:
            pass  # Fallback to liq zones
        
        # 1. Essayer les zones de liquidation si targets manquants
        if ('tp1' not in targets or 'sl' not in targets) and self.candles_5m and len(self.candles_5m) >= 20:
            try:
                # Calculer l'analyse des zones de liquidation
                self.liq_analysis = self.liq_analyzer.analyze(self.candles_5m, current_price=current_price)
                
                # Obtenir les targets basés sur les liq zones (avec ton levier 20x)
                liq_targets = self.liq_analyzer.get_targets_for_direction(
                    self.liq_analysis, direction_str, user_leverage=20
                )
                
                # Valider que les targets sont cohérents avec la direction
                if direction == SignalDirection.LONG:
                    if 'tp1' not in targets and liq_targets.get('tp1', 0) > current_price:
                        targets['tp1'] = liq_targets['tp1']
                    if 'tp2' not in targets and liq_targets.get('tp2', 0) > current_price:
                        targets['tp2'] = liq_targets['tp2']
                    if 'sl' not in targets and liq_targets.get('sl', float('inf')) < current_price:
                        targets['sl'] = liq_targets['sl']
                elif direction == SignalDirection.SHORT:
                    if 'tp1' not in targets and liq_targets.get('tp1', float('inf')) < current_price:
                        targets['tp1'] = liq_targets['tp1']
                    if 'tp2' not in targets and liq_targets.get('tp2', float('inf')) < current_price:
                        targets['tp2'] = liq_targets['tp2']
                    if 'sl' not in targets and liq_targets.get('sl', 0) > current_price:
                        targets['sl'] = liq_targets['sl']
            except Exception:
                pass  # Fallback to VP if liq zones fail
        
        # 2. Fallback: Volume Profile pour les targets manquants
        poc = self.vp.get('poc', 0)
        vah = self.vp.get('vah', 0)
        val = self.vp.get('val', 0)
        
        if direction == SignalDirection.LONG:
            if 'tp1' not in targets:
                if vah > current_price:
                    targets['tp1'] = vah
                elif poc > current_price:
                    targets['tp1'] = poc
            if 'tp2' not in targets:
                targets['tp2'] = round(current_price * 1.02, 1)
            if 'sl' not in targets:
                if val > 0 and val < current_price:
                    targets['sl'] = round(val * 0.995, 2)
                    
        elif direction == SignalDirection.SHORT:
            if 'tp1' not in targets:
                if val > 0 and val < current_price:
                    targets['tp1'] = val
                elif poc > 0 and poc < current_price:
                    targets['tp1'] = poc
            if 'tp2' not in targets:
                targets['tp2'] = round(current_price * 0.98, 1)
            if 'sl' not in targets:
                if vah > current_price:
                    targets['sl'] = round(vah * 1.005, 2)
        
        # 3. Fallback final: pourcentages fixes
        if 'tp1' not in targets:
            targets['tp1'] = round(current_price * (1.01 if direction == SignalDirection.LONG else 0.99), 1)
        if 'tp2' not in targets:
            targets['tp2'] = round(current_price * (1.02 if direction == SignalDirection.LONG else 0.98), 1)
        if 'sl' not in targets:
            targets['sl'] = round(current_price * (0.995 if direction == SignalDirection.LONG else 1.005), 2)
        if 'tp2' not in targets:
            targets['tp2'] = round(current_price * (1.02 if direction == SignalDirection.LONG else 0.98), 1)
        if 'sl' not in targets:
            targets['sl'] = round(current_price * (0.995 if direction == SignalDirection.LONG else 1.005), 2)
        
        # 4. Validation finale: contraintes R:R et distance max
        targets = self._validate_and_adjust_targets(targets, current_price, direction)
        
        return targets
    
    def _validate_and_adjust_targets(self, targets: Dict[str, float], 
                                     current_price: float, 
                                     direction: SignalDirection) -> Dict[str, float]:
        """
        Valide et ajuste les targets pour assurer:
        - SL max distance: 2% du prix
        - R:R minimum: 1:1.5 (risque 1 pour gain 1.5)
        """
        MAX_SL_DISTANCE_PCT = 0.02  # 2% max
        MIN_RR_RATIO = 1.5  # R:R minimum 1:1.5
        
        tp1 = targets.get('tp1', current_price)
        tp2 = targets.get('tp2', current_price)
        sl = targets.get('sl', current_price)
        
        if direction == SignalDirection.LONG:
            # Calculer les distances
            tp1_distance = tp1 - current_price
            sl_distance = current_price - sl
            sl_distance_pct = sl_distance / current_price
            
            # Si SL trop loin, ajuster
            if sl_distance_pct > MAX_SL_DISTANCE_PCT:
                # Option 1: SL basé sur R:R par rapport à TP1
                ideal_sl_distance = tp1_distance / MIN_RR_RATIO
                
                # Option 2: SL à 2% max
                max_sl_distance = current_price * MAX_SL_DISTANCE_PCT
                
                # Prendre le plus serré des deux
                new_sl_distance = min(ideal_sl_distance, max_sl_distance)
                targets['sl'] = round(current_price - new_sl_distance, 2)
        
        elif direction == SignalDirection.SHORT:
            tp1_distance = current_price - tp1
            sl_distance = sl - current_price
            sl_distance_pct = sl_distance / current_price
            
            if sl_distance_pct > MAX_SL_DISTANCE_PCT:
                ideal_sl_distance = tp1_distance / MIN_RR_RATIO
                max_sl_distance = current_price * MAX_SL_DISTANCE_PCT
                new_sl_distance = min(ideal_sl_distance, max_sl_distance)
                targets['sl'] = round(current_price + new_sl_distance, 2)
        
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

    def _analyze_smart_entry(self, signal: CompositeSignal) -> Optional[Dict]:
        """
        Analyse Smart Entry basée sur la structure 1H (Robustesse Backtest +141%)
        """
        try:
            if not self.candles_1h:
                return None
            
            # Use 1h candles for robust structure analysis
            smart_result = self.smart_entry_analyzer.analyze(
                direction=signal.direction.value,
                current_price=self.price,
                original_tp1=signal.targets.get('tp1', self.price),
                original_tp2=signal.targets.get('tp2', self.price),
                original_sl=signal.targets.get('sl', self.price),
                candles=self.candles_1h,
                liq_zones=self.liq_analyzer.analyze(self.candles_5m, self.price) if self.candles_5m else None
            )
            
            if smart_result.strategy != EntryStrategy.IMMEDIATE:
                if smart_result.strategy == EntryStrategy.WAIT_FOR_DIP:
                    strategy_text = f"ATTENDRE ${smart_result.optimal_entry:,.0f}"
                else:
                    strategy_text = f"LIMITE ${smart_result.optimal_entry:,.0f}"
                    
                # Store result in signal targets for easy access in Notifier
                # We use specific keys starting with _ to denote metadata
                signal.targets['_smart_entry_strategy'] = smart_result.strategy.value
                signal.targets['_smart_entry_price'] = smart_result.optimal_entry
                signal.targets['_smart_entry_text'] = strategy_text
                if smart_result.nearest_liq_zone:
                    signal.targets['_smart_entry_liq_zone'] = smart_result.nearest_liq_zone
                if smart_result.potential_improvement_pct > 0:
                     signal.targets['_smart_entry_rr_improvement'] = smart_result.potential_improvement_pct
                
                return {
                    'strategy': smart_result.strategy.value,
                    'price': smart_result.optimal_entry,
                    'strategy_text': strategy_text
                }
            return None
            
        except Exception as e:
            print(f"⚠️ Error in Smart Entry Analysis: {e}")
            return None
