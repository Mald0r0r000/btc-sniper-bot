"""
Moteur de Décision - Logique décisionnelle combinée
Combine tous les analyseurs pour générer des signaux de trading
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

import config


class SignalType(Enum):
    """Types de signaux générés"""
    SHORT_SNIPER = "SHORT_SNIPER"
    LONG_SNIPER = "LONG_SNIPER"
    LONG_BREAKOUT = "LONG_BREAKOUT"
    SHORT_BREAKOUT = "SHORT_BREAKOUT"
    FADE_HIGH = "FADE_HIGH"
    FADE_LOW = "FADE_LOW"
    QUANTUM_BUY = "QUANTUM_BUY"
    QUANTUM_SELL = "QUANTUM_SELL"
    DIAMOND_SETUP = "DIAMOND_SETUP"
    SHORT_SQUEEZE = "SHORT_SQUEEZE"
    LONG_FLUSH = "LONG_FLUSH"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class Signal:
    """Structure d'un signal de trading"""
    type: SignalType
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: int  # 1-10
    emoji: str
    description: str
    reasons: List[str]
    targets: Dict[str, float]
    warnings: List[str]


class DecisionEngine:
    """
    Moteur de décision combinant tous les indicateurs
    pour générer des signaux de trading priorisés
    """
    
    def __init__(
        self,
        current_price: float,
        order_book_data: Dict,
        cvd_data: Dict,
        volume_profile_data: Dict,
        funding_liq_data: Dict,
        fvg_data: Dict,
        entropy_data: Dict,
        open_interest: Dict
    ):
        self.price = current_price
        self.ob = order_book_data
        self.cvd = cvd_data
        self.vp = volume_profile_data
        self.fl = funding_liq_data
        self.fvg = fvg_data
        self.entropy = entropy_data
        self.oi = open_interest
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Génère tous les signaux possibles et retourne le meilleur
        
        Returns:
            Dict avec primary_signal, all_signals, market_context
        """
        signals = []
        warnings = []
        
        # 1. Vérifier les conditions Quantum (priorité haute)
        quantum_signal = self._check_quantum_signals()
        if quantum_signal:
            signals.append(quantum_signal)
        
        # 2. Vérifier les setups Sniper (VAL/VAH + Murs + CVD)
        sniper_signal = self._check_sniper_signals()
        if sniper_signal:
            signals.append(sniper_signal)
        
        # 3. Vérifier les breakouts
        breakout_signal = self._check_breakout_signals()
        if breakout_signal:
            signals.append(breakout_signal)
        
        # 4. Vérifier les setups Fade (range D-Shape)
        fade_signal = self._check_fade_signals()
        if fade_signal:
            signals.append(fade_signal)
        
        # 5. Vérifier les Diamond Setups (Funding)
        diamond_signal = self._check_diamond_signals()
        if diamond_signal:
            signals.append(diamond_signal)
        
        # 6. Vérifier les squeeze imminents
        squeeze_signal = self._check_squeeze_signals()
        if squeeze_signal:
            signals.append(squeeze_signal)
        
        # Collecter les warnings
        warnings.extend(self._collect_warnings())
        
        # Trier par confiance
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Signal principal
        primary = signals[0] if signals else self._no_signal(warnings)
        
        return {
            'primary_signal': self._signal_to_dict(primary),
            'all_signals': [self._signal_to_dict(s) for s in signals],
            'market_context': self._get_market_context(),
            'warnings': warnings
        }
    
    def _check_quantum_signals(self) -> Optional[Signal]:
        """Vérifie les signaux Quantum (Low Entropy + Sweep)"""
        signals = self.entropy.get('signals', {})
        compression = self.entropy.get('compression', {})
        
        if signals.get('quantum_buy'):
            return Signal(
                type=SignalType.QUANTUM_BUY,
                direction='LONG',
                confidence=9,
                emoji='⚛️🟢',
                description='QUANTUM BUY - Compression + Sweep + Volume',
                reasons=[
                    f"Low Entropy ({compression.get('current', 0):.2f} < {compression.get('threshold', 0.8)})",
                    "Sweep du support quantique",
                    "Volume spike détecté",
                    "Rejection haussière confirmée"
                ],
                targets={
                    'tp1': self.vp.get('poc', 0),
                    'tp2': self.vp.get('vah', 0)
                },
                warnings=[]
            )
        
        if signals.get('quantum_sell'):
            return Signal(
                type=SignalType.QUANTUM_SELL,
                direction='SHORT',
                confidence=9,
                emoji='⚛️🔴',
                description='QUANTUM SELL - Compression + Sweep + Volume',
                reasons=[
                    f"Low Entropy ({compression.get('current', 0):.2f} < {compression.get('threshold', 0.8)})",
                    "Sweep de la résistance quantique",
                    "Volume spike détecté",
                    "Rejection baissière confirmée"
                ],
                targets={
                    'tp1': self.vp.get('poc', 0),
                    'tp2': self.vp.get('val', 0)
                },
                warnings=[]
            )
        
        return None
    
    def _check_sniper_signals(self) -> Optional[Signal]:
        """Vérifie les signaux Sniper (VAL/VAH + Murs + CVD)"""
        vah = self.vp.get('vah', 0)
        val = self.vp.get('val', 0)
        poc = self.vp.get('poc', 0)
        
        near_vah = abs(self.price - vah) / vah < config.NEAR_LEVEL_PCT if vah > 0 else False
        near_val = abs(self.price - val) / val < config.NEAR_LEVEL_PCT if val > 0 else False
        
        wall_ask = self.ob.get('wall_ask', {})
        wall_bid = self.ob.get('wall_bid', {})
        
        cvd_net = self.cvd.get('net_cvd', 0)
        
        # SHORT SNIPER: Near VAH + Mur vendeur dangereux + CVD négatif
        if near_vah and wall_ask.get('is_danger') and cvd_net < 0:
            return Signal(
                type=SignalType.SHORT_SNIPER,
                direction='SHORT',
                confidence=8,
                emoji='🔴🎯',
                description='SHORT SNIPER - Rejet VAH + Mur + CVD Divergence',
                reasons=[
                    f"Prix proche VAH (${vah:.2f})",
                    f"Mur vendeur de ${wall_ask.get('value_m', 0):.1f}M à ${wall_ask.get('price', 0):.2f}",
                    f"CVD négatif ({cvd_net:+.2f} BTC)",
                    "Les acheteurs s'épuisent sur le mur"
                ],
                targets={
                    'tp1': poc,
                    'tp2': val,
                    'sl': vah * 1.005
                },
                warnings=[]
            )
        
        # LONG SNIPER: Near VAL + Mur acheteur dangereux + CVD positif
        if near_val and wall_bid.get('is_danger') and cvd_net > 0:
            return Signal(
                type=SignalType.LONG_SNIPER,
                direction='LONG',
                confidence=8,
                emoji='🟢🎯',
                description='LONG SNIPER - Rebond VAL + Mur + CVD Absorption',
                reasons=[
                    f"Prix proche VAL (${val:.2f})",
                    f"Mur acheteur de ${wall_bid.get('value_m', 0):.1f}M à ${wall_bid.get('price', 0):.2f}",
                    f"CVD positif ({cvd_net:+.2f} BTC)",
                    "Les vendeurs s'écrasent sur le mur"
                ],
                targets={
                    'tp1': poc,
                    'tp2': vah,
                    'sl': val * 0.995
                },
                warnings=[]
            )
        
        return None
    
    def _check_breakout_signals(self) -> Optional[Signal]:
        """Vérifie les signaux de breakout"""
        vah = self.vp.get('vah', 0)
        val = self.vp.get('val', 0)
        
        near_vah = abs(self.price - vah) / vah < config.NEAR_LEVEL_PCT if vah > 0 else False
        near_val = abs(self.price - val) / val < config.NEAR_LEVEL_PCT if val > 0 else False
        
        cvd_net = self.cvd.get('net_cvd', 0)
        agg_ratio = self.cvd.get('aggression_ratio', 1.0)
        
        # LONG BREAKOUT: Near VAH + CVD fort + Agression haute
        if near_vah and cvd_net > config.CVD_BREAKOUT_THRESHOLD and agg_ratio > config.AGGRESSION_BULLISH:
            return Signal(
                type=SignalType.LONG_BREAKOUT,
                direction='LONG',
                confidence=7,
                emoji='🚀',
                description='LONG BREAKOUT - Agression VAH',
                reasons=[
                    f"CVD explosif ({cvd_net:+.2f} BTC)",
                    f"Ratio agression: {agg_ratio:.2f}",
                    "Le mur vendeur est en train de sauter"
                ],
                targets={
                    'tp1': vah * 1.01,
                    'tp2': vah * 1.02
                },
                warnings=["⚠️ Breakout = risque élevé si faux signal"]
            )
        
        # SHORT BREAKOUT: Near VAL + CVD négatif fort + Agression basse
        if near_val and cvd_net < -config.CVD_BREAKOUT_THRESHOLD and agg_ratio < config.AGGRESSION_BEARISH:
            return Signal(
                type=SignalType.SHORT_BREAKOUT,
                direction='SHORT',
                confidence=7,
                emoji='💥',
                description='SHORT BREAKDOWN - Agression VAL',
                reasons=[
                    f"CVD négatif fort ({cvd_net:+.2f} BTC)",
                    f"Ratio agression: {agg_ratio:.2f}",
                    "Le support VAL cède sous pression"
                ],
                targets={
                    'tp1': val * 0.99,
                    'tp2': val * 0.98
                },
                warnings=["⚠️ Breakdown = risque élevé si faux signal"]
            )
        
        return None
    
    def _check_fade_signals(self) -> Optional[Signal]:
        """Vérifie les signaux Fade (range D-Shape)"""
        shape = self.vp.get('shape', '')
        vah = self.vp.get('vah', 0)
        val = self.vp.get('val', 0)
        poc = self.vp.get('poc', 0)
        
        if shape != 'D-Shape':
            return None
        
        near_vah = abs(self.price - vah) / vah < config.NEAR_LEVEL_PCT if vah > 0 else False
        near_val = abs(self.price - val) / val < config.NEAR_LEVEL_PCT if val > 0 else False
        
        if near_vah:
            return Signal(
                type=SignalType.FADE_HIGH,
                direction='SHORT',
                confidence=5,
                emoji='📉',
                description='FADE SETUP - Haut de Range D-Shape',
                reasons=[
                    "Profil D-Shape = distribution équilibrée",
                    f"Prix au haut du range (VAH: ${vah:.2f})",
                    "Probabilité de retour au POC"
                ],
                targets={
                    'tp1': poc
                },
                warnings=["Setup de range, pas de tendance forte"]
            )
        
        if near_val:
            return Signal(
                type=SignalType.FADE_LOW,
                direction='LONG',
                confidence=5,
                emoji='📈',
                description='FADE SETUP - Bas de Range D-Shape',
                reasons=[
                    "Profil D-Shape = distribution équilibrée",
                    f"Prix au bas du range (VAL: ${val:.2f})",
                    "Probabilité de retour au POC"
                ],
                targets={
                    'tp1': poc
                },
                warnings=["Setup de range, pas de tendance forte"]
            )
        
        return None
    
    def _check_diamond_signals(self) -> Optional[Signal]:
        """Vérifie les Diamond Setup (Funding négatif + CVD positif)"""
        funding = self.fl.get('funding', {})
        agg_ratio = self.cvd.get('aggression_ratio', 1.0)
        
        if funding.get('is_negative') and agg_ratio > 1.0:
            return Signal(
                type=SignalType.DIAMOND_SETUP,
                direction='LONG',
                confidence=6,
                emoji='💎',
                description='DIAMOND SETUP - Funding Négatif + Achat Spot',
                reasons=[
                    f"Funding négatif: {funding.get('current_pct', 0):.4f}%",
                    f"Ratio agression acheteur: {agg_ratio:.2f}",
                    "Les shorts paient les longs = position favorable"
                ],
                targets={},
                warnings=["Les shorts sont potentiellement piégés"]
            )
        
        return None
    
    def _check_squeeze_signals(self) -> Optional[Signal]:
        """Vérifie si un squeeze est imminent"""
        magnet = self.fl.get('magnet', {})
        
        if magnet.get('distance_pct', 100) < 0.2:
            direction = magnet.get('direction', '')
            
            if direction == 'HAUSSIER':
                return Signal(
                    type=SignalType.SHORT_SQUEEZE,
                    direction='LONG',
                    confidence=7,
                    emoji='⚡📈',
                    description='SHORT SQUEEZE IMMINENT',
                    reasons=[
                        f"Aimant à ${magnet.get('price', 0):.0f}",
                        f"Distance: {magnet.get('distance_pct', 0):.2f}%",
                        "Liquidations x100 shorts très proches"
                    ],
                    targets={
                        'tp1': magnet.get('price', 0)
                    },
                    warnings=["Mouvement violent possible"]
                )
            else:
                return Signal(
                    type=SignalType.LONG_FLUSH,
                    direction='SHORT',
                    confidence=7,
                    emoji='⚡📉',
                    description='LONG FLUSH IMMINENT',
                    reasons=[
                        f"Aimant à ${magnet.get('price', 0):.0f}",
                        f"Distance: {magnet.get('distance_pct', 0):.2f}%",
                        "Liquidations x100 longs très proches"
                    ],
                    targets={
                        'tp1': magnet.get('price', 0)
                    },
                    warnings=["Mouvement violent possible"]
                )
        
        return None
    
    def _collect_warnings(self) -> List[str]:
        """Collecte les avertissements généraux"""
        warnings = []
        
        wall_ask = self.ob.get('wall_ask', {})
        wall_bid = self.ob.get('wall_bid', {})
        
        if wall_ask.get('is_danger'):
            warnings.append(f"⚠️ Gros mur vendeur à ${wall_ask.get('price', 0):.2f} (${wall_ask.get('value_m', 0):.1f}M)")
        
        if wall_bid.get('is_danger'):
            warnings.append(f"⚠️ Gros mur acheteur à ${wall_bid.get('price', 0):.2f} (${wall_bid.get('value_m', 0):.1f}M)")
        
        funding = self.fl.get('funding', {})
        if funding.get('is_expensive_for_longs'):
            warnings.append(f"⚠️ Funding élevé ({funding.get('current_pct', 0):.4f}%) - Cher pour les longs")
        
        return warnings
    
    def _no_signal(self, warnings: List[str]) -> Signal:
        """Signal par défaut quand aucune condition n'est remplie"""
        return Signal(
            type=SignalType.NO_SIGNAL,
            direction='NEUTRAL',
            confidence=0,
            emoji='💤',
            description='Pas de setup clair - Attendre les bornes',
            reasons=["Aucune condition de signal remplie"],
            targets={},
            warnings=warnings
        )
    
    def _get_market_context(self) -> Dict[str, Any]:
        """Génère un résumé du contexte de marché"""
        return {
            'price': self.price,
            'ob_pressure': self.ob.get('pressure', 'NEUTRE'),
            'ob_bid_pct': self.ob.get('bid_ratio_pct', 50),
            'cvd_status': self.cvd.get('status', 'NEUTRE'),
            'cvd_net': self.cvd.get('net_cvd', 0),
            'vp_shape': self.vp.get('shape', 'D-Shape'),
            'vp_poc': self.vp.get('poc', 0),
            'vp_vah': self.vp.get('vah', 0),
            'vp_val': self.vp.get('val', 0),
            'funding_rate': self.fl.get('funding', {}).get('current_pct', 0),
            'magnet': self.fl.get('magnet', {}).get('description', ''),
            'quantum_state': self.entropy.get('quantum_state', 'UNKNOWN'),
            'compression': self.entropy.get('compression', {}).get('current', 1.0),
            'oi_btc': self.oi.get('amount', 0)
        }
    
    def _signal_to_dict(self, signal: Signal) -> Dict[str, Any]:
        """Convertit un Signal en dict"""
        return {
            'type': signal.type.value,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'emoji': signal.emoji,
            'description': signal.description,
            'reasons': signal.reasons,
            'targets': signal.targets,
            'warnings': signal.warnings
        }
