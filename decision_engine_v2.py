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
    smart_entry: Optional[Dict] = None  # Smart Entry recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
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
        
        # Add smart_entry if available
        if self.smart_entry:
            result['smart_entry'] = self.smart_entry
        
        return result


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
    },
    # Intraday 1H-2D - Optimisé (Post-Backtest 2026)
    # Findings: OI & CVD are King. KDJ needs to be inverted. OB is noise.
    'intraday_1h_2d': {
        'technical': 35,        # Adjusted (-5)
        'structure': 15,        
        'multi_exchange': 10,   
        'derivatives': 25,      
        'onchain': 5,           
        'sentiment': 5,         
        'macro': 5              # +5 (Activé avec nouvelle logique DXY)
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
        cross_asset_data: Dict = None,
        event_data: Dict = None,
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
        hyperliquid_data: Dict = None,
        # MACD 3D (HTF trend confirmation)
        macd_data: Dict = None,
        # NEW R&D Metrics
        squeeze_data: Dict = None,
        spot_perp_data: Dict = None
    ):
        self.price = current_price
        self.squeeze = squeeze_data or {}
        self.spot_perp = spot_perp_data or {}
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
        self.cross_asset = cross_asset_data or {}
        self.event = event_data or {}
        
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
        
        # MACD 3D data
        self.macd = macd_data or {}
    
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
    
    def _apply_dynamic_weight_boost(self, base_weights: Dict[str, int]) -> Dict[str, int]:
        """
        Apply conditional weight boosting based on extreme thresholds
        
        Macro indicators are baseline 2-5%, but spike to 8-12% during extreme events:
        - M2 YoY > ±5%
        - DXY Daily > ±2%
        - SPX Daily > ±3%
        - OnChain Netflow > ±10k BTC
        
        Returns:
            Adjusted weights (still sum to ~100%)
        """
        adjusted = base_weights.copy()
        boost_total = 0
        boost_reasons = []
        
        # Check M2 threshold (Liquidity Shock)
        m2_yoy = self.cross_asset.get('m2', {}).get('yoy_change', 0)
        if abs(m2_yoy) > 5:  # Major liquidity event
            m2_boost = 10 if abs(m2_yoy) > 10 else 6
            boost_total += m2_boost
            direction = "FLOOD" if m2_yoy > 0 else "CRISIS"
            boost_reasons.append(f"M2 {direction} ({m2_yoy:+.1f}% YoY) → +{m2_boost}% Macro")
        
        # Check DXY threshold (Dollar Strength Crisis)
        dxy_change = self.cross_asset.get('dxy', {}).get('daily_change', 0)
        if abs(dxy_change) > 2.0:
            dxy_boost = 8
            boost_total += dxy_boost
            direction = "BEARISH_BTC" if dxy_change > 0 else "BULLISH_BTC"
            boost_reasons.append(f"DXY Spike ({dxy_change:+.1f}%) → +{dxy_boost}% Macro [{direction}]")
        
        # Check SPX threshold (Equity Market Crash/Rally)
        spx_change = self.cross_asset.get('spx', {}).get('change_24h', 0)
        if abs(spx_change) > 3.0:
            spx_boost = 6
            boost_total += spx_boost
            direction = "CRASH" if spx_change < 0 else "RALLY"
            boost_reasons.append(f"SPX {direction} ({spx_change:+.1f}%) → +{spx_boost}% Macro")
        
        # Check OnChain whale flow (Panic/Accumulation)
        netflow = self.onchain.get('exchange_flows', {}).get('netflow_btc_24h', 0)
        if abs(netflow) > 10000:
            oc_boost = 7
            boost_total += oc_boost
            direction = "DUMP" if netflow > 0 else "ACCUMULATION"
            boost_reasons.append(f"Whale {direction} ({netflow:+,.0f} BTC) → +{oc_boost}% OnChain")
        
        # Apply boost by reducing other dimensions proportionally
        if boost_total > 0:
            # Reduce Technical, Structure, Multi-Exchange, Sentiment proportionally
            reduction_pool = ['technical', 'structure', 'multi_exchange', 'sentiment']
            total_reducible = sum(adjusted.get(dim, 0) for dim in reduction_pool)
            
            for dim in reduction_pool:
                if total_reducible > 0:
                    reduction = int(adjusted[dim] * (boost_total / total_reducible))
                    adjusted[dim] = max(0, adjusted[dim] - reduction)
            
            # Add boost to macro/onchain
            adjusted['macro'] += boost_total
            
            # Log the boost
            print(f"   ⚡ DYNAMIC BOOST: {', '.join(boost_reasons)}")
        
        return adjusted
    
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
        # UPDATE: INVERT Logic for Venturi here to match findings (Pressure = Reversal)
        # We handle this inside _score_venturi logic normally, or here globally.
        # But wait, self.venturi dict usually contains 'signal_modifier'.
        # Let's adjust it by checking the raw 'direction' and reversing implication if it's based on pressure.
        
        venturi_modifier = self.venturi.get('signal_modifier', 0)
        self_trading_modifier = self.self_trading.get('signal_modifier', 0)
        
        # If Venturi detected "UP" based on pressure, we want to negate or invert that for mean reversion
        # For now, let's trust the existing logic if we update the analyzer later,
        # OR just reduce its impact here if we are unsure.
        # Backtest showed -0.13 correlation. So +10 for Venturi Up is actually bearish.
        # Let's simple INVERT the venturi modifier.
        
        adjusted_score = max(0, min(100, adjusted_score - venturi_modifier + self_trading_modifier))
        
        # 4c. Appliquer le modifier Hyperliquid Whale Sentiment (réduit pour intraday)
        whale_modifier = self.hyperliquid.get('signal_modifier', 0)
        # Cap à ±5 pour intraday (whales = swing traders, pas scalpers)
        whale_modifier = max(-5, min(5, whale_modifier))
        adjusted_score = max(0, min(100, adjusted_score + whale_modifier))
        
        # 4d. Structural Quality Filter (AMT)
        final_score = self._apply_structural_quality_filter(adjusted_score)
        if final_score != adjusted_score:
            print(f"   🛡️ STRUCTURAL FILTER: Neutralizing score (Edge not clear)")
            adjusted_score = final_score
        
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
                htf_penalty = 10  # Reduced from 20 to 10
            elif not is_long_signal and is_htf_bullish:
                htf_penalty = 10  # Reduced from 20 to 10
                
            adjusted_score -= htf_penalty
        
        # Filter 5: Confluence Check (Asymmetric thresholds)
        # LONG: Require 3+ dimensions > 55
        # SHORT: Require 2+ dimensions < 45 (looser threshold due to bullish bias in whale/macro data)
        if signal_type not in [SignalType.NO_SIGNAL]:
            bullish_dims = sum(1 for score in dimension_scores.values() if score > 55)
            bearish_dims = sum(1 for score in dimension_scores.values() if score < 45)
            
            # Additional: Strong technical bearish (< 35) counts as 2 bearish dims
            tech_score = dimension_scores.get('technical', 50)
            if tech_score < 35:
                bearish_dims += 1  # Bonus for strong technical bearish
            
            if direction == SignalDirection.LONG and bullish_dims < 2: # Reduced from 3 to 2
                signal_type = SignalType.NO_SIGNAL
                direction = SignalDirection.NEUTRAL
            elif direction == SignalDirection.SHORT and bearish_dims < 2:  # Reduced from 3 to 2
                signal_type = SignalType.NO_SIGNAL
                direction = SignalDirection.NEUTRAL
        
        # Filter 6: Macro Regime Check (Phase 2 - Cross-Asset)
        macro_regime = self.cross_asset.get('overall_regime', 'MIXED')
        macro_btc_impact = self.cross_asset.get('regime_btc_impact', 'NEUTRAL')
        macro_penalty = 0
        
        if signal_type not in [SignalType.NO_SIGNAL]:
            is_long_signal = direction == SignalDirection.LONG
            
            if macro_regime == 'RISK_OFF' and is_long_signal:
                macro_penalty = 15  # Going long in risk-off environment
            elif macro_regime == 'RISK_ON' and not is_long_signal and direction != SignalDirection.NEUTRAL:
                macro_penalty = 15  # Going short in risk-on environment
                
            adjusted_score -= macro_penalty
        
        # Filter 8: Venturi Confirmation (R&D Logic)
        # Prevent Shorting Macro Bullish dips unless Fluid Dynamics (Venturi) confirms breakdown
        if direction == SignalDirection.SHORT and signal_type not in [SignalType.NO_SIGNAL]:
            tech_score = dimension_scores.get('technical', 50)
            macro_score = dimension_scores.get('macro', 50)
            
            # Conflict: Technicals scream BEARISH (<35) but Macro says BULLISH (>55)
            if tech_score < 35 and macro_score > 55:
                # We need Venturi confirmation to proceed with SHORT
                venturi_direction = self.venturi.get('direction', 'NEUTRAL')
                breakout_prob = self.venturi.get('breakout_probability', 0)
                
                # If Venturi is not DOWN, we block the signal
                is_venturi_bearish = (venturi_direction == 'DOWN' or 
                                     (breakout_prob > 50 and venturi_direction == 'DOWN'))
                
                
                if not is_venturi_bearish:
                    # Convert BLOCK to PENALTY
                    # signal_type = SignalType.NO_SIGNAL
                    # direction = SignalDirection.NEUTRAL
                    adjusted_score -= 15
                    warnings.append(f"⚠️ Contre-tendance Macro ({macro_score}/100) sans confirmation Venturi")
        if self.event.get('event_active') and signal_type not in [SignalType.NO_SIGNAL]:
            event_penalty = self.event.get('confidence_penalty', 30)
            adjusted_score -= event_penalty
            # Event warning will be added in _build_composite_signal
        
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
        
        # Order Book imbalance (Weight: LOW - Not predictive in backtest)
        # Reduced max impact from 20 to 10
        bid_ratio = self.ob.get('bid_ratio_pct', 50)
        if bid_ratio > 60:
            score += (bid_ratio - 50) * 0.2  # Reduced
        elif bid_ratio < 40:
            score -= (50 - bid_ratio) * 0.2  # Reduced
        
        # CVD (Weight: HIGH - Predictive)
        # MTF Composite Score Strategy
        # score > 60 = Bullish
        # score < 40 = Bearish
        cvd_mtf_score = self.cvd.get('composite_score', 50)
        
        # Apply score deviation from 50 directly
        # Max impact: ±25 points
        cvd_deviation = cvd_mtf_score - 50
        score += (cvd_deviation * 0.5) 
        
        # Bonus for Confluence (All aligned)
        confluence = self.cvd.get('confluence', 'MIXED')
        if confluence == 'ALL_BULLISH':
            score += 5
        elif confluence == 'ALL_BEARISH':
            score -= 5
        
        # Volume Profile: Structural Context (R&D Point 3 - refined with AMT)
        vp_context = self.vp.get('context', 'NEUTRAL')
        
        # Scoring based on structural position (Auction Market Theory)
        if vp_context == 'IMBALANCE_EXPANSION_UP':
            score += 20 # Strong breakout confirmed by structure
        elif vp_context == 'IMBALANCE_EXPANSION_DOWN':
            score -= 20 # Strong breakdown confirmed by structure
        elif vp_context == 'TRAVERSING_LIQUID_GAP':
            # Fast travel zone - follow momentum
            score += (10 if score > 50 else -10)
        elif vp_context == 'VALUE_AREA_ROTATION_UP':
            score += 10 # Mean reversion from POC to VAH
        elif vp_context == 'VALUE_AREA_ROTATION_DOWN':
            score -= 10 # Mean reversion from POC to VAL
        elif vp_context == 'STUCK_AT_POC':
            # High efficiency zone - price sticky - pull towards neutral
            score = 50 + (score - 50) * 0.4
            
        # Varies slightly based on POC proximity
        poc = self.vp.get('poc', 0)
        if poc > 0:
            dist_to_poc = abs(self.price - poc) / poc
            if dist_to_poc < 0.001: # Ultra near POC
                score = 50 + (score - 50) * 0.8 # De-prioritize signal in high liquidity zone
            
        # KDJ Momentum (Oscillator)
        # Updated Logic (User Correction + Statistical Analysis):
        # High J (>80) = Overbought/Sell -> Penalize Score
        # Low J (<20) = Oversold/Buy -> Boost Score
        # This aligns with the negative correlation found (-0.36)
        
        kdj_values = self.kdj.get('values', {})
        j_val = kdj_values.get('j', 50)
        
        # Penalize High J (Overbought)
        if j_val > 80:
            penalty = (j_val - 80) * 1.5  # Max penalty ~30 points if J=100
            score -= penalty
            
        # Boost Low J (Oversold)
        elif j_val < 20:
            bonus = (20 - j_val) * 1.5    # Max bonus ~30 points if J=0
            score += bonus
            
        # Slope confirmation (only if favorable)
        j_slope = kdj_values.get('j_slope', 0)
        if j_val < 20 and j_slope > 0:
            score += 5 # Turning up from oversold
        elif j_val > 80 and j_slope < 0:
            score -= 5 # Turning down from overbought
        
        # ========== ADX MULTIPLIER (Trend Strength Amplifier) ==========
        # ADX doesn't tell direction, it tells confidence
        # When ADX > 25, we're in a trend - amplify the signal
        adx_value = self.adx.get('adx', 0)
        adx_regime = self.adx.get('regime', 'RANGING')
        
        if adx_value > 25 and adx_regime == 'TRENDING':
            # Calculate multiplier: caps at +20% amplification
            mult = 1 + min((adx_value - 25) / 100, 0.2)
            # Amplify deviation from neutral (50)
            deviation = score - 50
            score = 50 + (deviation * mult)
            
            # ========== ADX DIRECTION (DI+ vs DI-) ==========
            # Use DI comparison to confirm trend direction
            adx_direction = self.adx.get('trend_direction', 'NEUTRAL')
            if adx_direction == 'BEARISH':  # DI- > DI+
                score -= 10  # Confirmed bearish momentum
            elif adx_direction == 'BULLISH':  # DI+ > DI-
                score += 10  # Confirmed bullish momentum
            
        # ========== R&D: ABSORPTION PENALTY ==========
        # If CVD shows strong aggression but absorption is detected, the trend is likely stalled
        if self.cvd.get('absorption_risk'):
            deviation = score - 50
            if abs(deviation) > 10:
                # Reduce the score by 30% towards neutral - "Trust but verify"
                score = 50 + (deviation * 0.7)
            # Add a flat penalty if agression is very high
            score -= (5 if deviation > 0 else -5)
            
        # ========== R&D: QUANTUM SQUEEZE BOOST ==========
        # Compression + High OI = High probability of edge expansion
        if self.squeeze.get('is_squeeze'):
            squeeze_score = self.squeeze.get('squeeze_score', 0)
            # Boost the impact of other technical signals by up to 25%
            boost_factor = 1.0 + min(squeeze_score / 20, 0.25)
            deviation = score - 50
            score = 50 + (deviation * boost_factor)
            
        return max(0, min(100, score))
    
    def _score_structure(self) -> float:
        """Score structure (FVG + Entropy + MACD MTF)"""
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
        
        # ========== MTF MACD ANALYSIS (Intraday Optimized) ==========
        # Multi-Timeframe MACD: 1H (50%), 4H (30%), 1D (15%), 3D (5%)
        # Composite score ranges from -100 (very bearish) to +100 (very bullish)
        macd_available = self.macd.get('available', False)
        
        if macd_available:
            composite_score = self.macd.get('composite_score', 0)
            confluence = self.macd.get('confluence', 'MIXED')
            
            # Scale composite score from [-100, +100] to [-15, +15] for structure
            # We reduce the impact to not overwhelm other structure signals
            macd_contribution = (composite_score / 100) * 15
            score += macd_contribution
            
            # Bonus for full confluence (all timeframes aligned)
            if confluence == 'ALL_BULLISH':
                score += 5  # Extra boost for alignment
            elif confluence == 'ALL_BEARISH':
                score -= 5  # Extra penalty for aligned bear
        
        # ========== FALLBACK: 24h Price Change Trend Detector ==========
        # If we have 1h candles, use them to detect major trend changes
        # This helps when MACD 3D might be stale or lagging
        if self.candles_1h and len(self.candles_1h) >= 24:
            try:
                # Get closes from last 24 candles (24h)
                closes = [float(c.get('close', c.get('c', 0))) for c in self.candles_1h[-24:]]
                if closes[0] > 0 and closes[-1] > 0:
                    price_change_24h_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
                    
                    # Strong bearish: dropped >3% in 24h
                    if price_change_24h_pct < -3:
                        score -= 10  # Bearish pressure
                    elif price_change_24h_pct < -1.5:
                        score -= 5
                    # Strong bullish: rallied >3% in 24h
                    elif price_change_24h_pct > 3:
                        score += 10
                    elif price_change_24h_pct > 1.5:
                        score += 5
            except Exception:
                pass  # Fallback silently fails
        
        return max(0, min(100, score))
    
    def _score_multi_exchange(self) -> float:
        """Score multi-exchange"""
        score = 50.0
        
        # 1. Spot vs Perp Divergence (Priority)
        # ---------------------------------------------
        # Check 1H timeframe for intraday signal
        sp_data = self.spot_perp.get('1h', {})
        sp_signal = sp_data.get('signal', 'NEUTRAL')
        
        if sp_signal == 'BULLISH_QUALITY': # Spot Lead Rally
            score += 15
        elif sp_signal == 'BEARISH_DIVERGENCE': # Perp Lead Rally (Trap)
            score -= 10
        elif sp_signal == 'BEARISH_HEAVY': # Spot Selling into Perp Buying
            score -= 15
        elif sp_signal == 'BULLISH': # Broad Buying
            score += 10
        elif sp_signal == 'BEARISH': # Broad Selling
            score -= 10
            
        if not self.multi_ex:
            return score
        
        # 2. Funding divergence
        # ---------------------------------------------
        funding = self.multi_ex.get('funding_analysis', {})
        if funding.get('signal') == 'SHORTS_PAYING':
            score += 15  # Short squeeze potential
        elif funding.get('signal') == 'LONGS_EXPENSIVE':
            score -= 10
        
        # 3. Arbitrage opportunity
        # ---------------------------------------------
        arb = self.multi_ex.get('arbitrage', {})
        if arb.get('opportunity'):
            spread_pct = arb.get('spread_pct', 0)
            if spread_pct > 0.01:
                score += 5  # Légèrement bullish si arb vers le haut
        
        # 4. Global order book imbalance
        # ---------------------------------------------
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
        """
        Score macro REFONDU (User Request):
        - 80% Poids sur DXY (Corrélation prouvée)
        - Règles conditionnelles pour SPX/M2 (Chocs uniquement)
        """
        score = 50.0
        
        if not self.macro:
            return score
            
        # 1. DXY DOMINANCE (80% de l'impact standard)
        # Corrélation Inverse: DXY Monte -> BTC Baisse
        dxy = self.cross_asset.get('dxy', {})
        dxy_change = dxy.get('change_24h', 0)
        
        # Formule: -20 points par 1% de variation DXY
        # Exemple: DXY +0.5% -> Score -10 (40)
        # Exemple: DXY -0.5% -> Score +10 (60)
        dxy_impact = dxy_change * -20
        score += dxy_impact
        
        # 2. RÈGLES DE CHOC (SPX & M2)
        # Ne s'activent qu'en cas de mouvement violent
        
        # SPX: Crash Protection
        spx = self.cross_asset.get('spx', {})
        spx_change = spx.get('daily_change', 0)
        
        if spx_change < -2.0: # Krach boursier (> -2%)
            score -= 25 # Pénalité majeure "Risk Off"
        elif spx_change > 2.0: # Rallye euphorique
            score += 15 # Boost "Risk On"
            
        # M2: Liquidity Crisis
        m2 = self.cross_asset.get('m2', {})
        m2_yoy = m2.get('yoy_change', 0)
        
        if m2_yoy < -5.0: # Contraction sévère (Crise de liquidité)
            score -= 20
        
        return max(0, min(100, score))
    
    def _calculate_composite_score(self, dimension_scores: Dict[str, float]) -> Tuple[float, Dict]:
        """Calcule le score composite pondéré avec dynamic boosting"""
        # Apply dynamic weight adjustment for extreme market events
        adjusted_weights = self._apply_dynamic_weight_boost(self.WEIGHT_CONFIG)
        
        weighted_scores = {}
        total_weight = sum(adjusted_weights.values())
        
        composite = 0.0
        for dim, weight in adjusted_weights.items():
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
        
        # Fade setups (Range Trading D-Shape)
        if self.vp.get('shape') == 'D-Shape':
            skew = self.vp.get('skew', 0)
            adx_val = self.adx.get('adx', 0)
            
            # FADE HIGH (Short VAH)
            # Relaxed: Removed Skew requirement, rely on ADX < 30
            if near_vah:
                if adx_val < 30: 
                    return SignalType.FADE_HIGH
            
            # FADE LOW (Long VAL)
            # Relaxed: Removed Skew requirement
            if near_val:
                if adx_val < 30:
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
        
        # --- SMART ENTRY INTEGRATION ---
        # Calculate Smart Entry recommendation (using 1H candles for robustness)
        smart_entry_data = None
        if self.candles_1h and signal_type != SignalType.NO_SIGNAL:
            try:
                smart_result = self.smart_entry_analyzer.analyze(
                    direction=direction.value,
                    current_price=self.price,
                    original_tp1=targets.get('tp1', self.price),
                    original_tp2=targets.get('tp2', self.price),
                    original_sl=targets.get('sl', self.price),
                    candles=self.candles_1h,
                    candles_15m=self.candles_15m if hasattr(self, 'candles_15m') else None,
                    liq_zones=self.liq_analyzer.analyze(self.candles_5m, self.price) if self.candles_5m else None,
                    mtf_macd_context=self.macd  # Pass MTF MACD context
                )
                
                # Build smart_entry dict
                smart_entry_data = {
                    'strategy': smart_result.strategy.value,
                    'optimal_entry': smart_result.optimal_entry,
                    'current_price': self.price,
                    'liq_zone': smart_result.nearest_liq_zone,
                    'rr_improvement': smart_result.potential_improvement_pct,
                    'timeout_hours': smart_result.entry_timeout_hours
                }
                
                # Add to reasons if not immediate
                if smart_result.strategy != EntryStrategy.IMMEDIATE:
                    if smart_result.strategy == EntryStrategy.WAIT_FOR_DIP:
                        strategy_text = f"⏳ Attendre ${smart_result.optimal_entry:,.0f} (+{smart_result.potential_improvement_pct:.0f}% R:R)"
                    else:
                        strategy_text = f"📝 Limite ${smart_result.optimal_entry:,.0f}"
                    reasons.append(strategy_text)
                    
            except Exception as e:
                print(f"   ⚠️ Smart Entry failed: {e}")
                smart_entry_data = None
        
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
            manipulation_penalty=manipulation_penalty,
            smart_entry=smart_entry_data
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
        amt_target = self.vp.get('target_price', 0)
        
        if direction == SignalDirection.LONG:
            if 'tp1' not in targets:
                # Priorité au target AMT (HVN suivant)
                if amt_target > current_price:
                    targets['tp1'] = amt_target
                elif vah > current_price:
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
                # Priorité au target AMT (HVN suivant)
                if amt_target > 0 and amt_target < current_price:
                    targets['tp1'] = amt_target
                elif val > 0 and val < current_price:
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
            'vp_context': self.vp.get('context', 'NEUTRAL'),
            'quantum_state': self.entropy.get('quantum_state', 'UNKNOWN'),
            'fear_greed': self.sentiment.get('fear_greed', {}).get('value', 50)
        }
    
    def _apply_structural_quality_filter(self, score: float) -> float:
        """
        Refines the signal by filtering out low-quality zones (AMT)
        Force NO_TRADE if stuck in balance without clear driver.
        """
        vp_context = self.vp.get('context', 'NEUTRAL')
        regime = self.vp.get('regime', 'BALANCE')
        oi_trend = self.oi.get('signal', {}).get('sentiment', 'NEUTRAL')
        
        # 1. POC Danger Zone (High Efficiency = No Edge)
        if vp_context == 'STUCK_AT_POC' and regime == 'BALANCE':
            # Even if technicals suggest something, if OI is dropping, stay out
            if oi_trend == 'BEARISH' or abs(score - 50) < 15:
                return 50.0 # Force Neutral -> NO_SIGNAL
                
        # 2. Gap Traversal without Momentum
        if vp_context == 'TRAVERSING_LIQUID_GAP':
            adx_val = self.adx.get('adx', 0)
            if adx_val < 20: # No trend
                return 50.0 # Don't gamble on the gap without momentum
                
        return score

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
