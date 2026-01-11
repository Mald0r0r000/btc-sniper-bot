"""
Analyseur Entropy / Quantum State
- Bollinger Bands Width Compression
- Détection état Low Entropy (marché comprimé)
- Pre-move state analysis (état AVANT le mouvement)
- Signal de potentiel d'explosion
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List

import config


class EntropyAnalyzer:
    """
    Analyse l'entropie du marché basée sur la compression de volatilité
    
    Concept "Move 42" : 
    - Compression (Low Entropy) → Singularité → Expansion
    - Un marché comprimé a un potentiel d'explosion élevé
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame OHLCV
        """
        self.df = df.copy()
        self.bb_length = config.BB_LENGTH
        self.compression_lookback = config.COMPRESSION_LOOKBACK
        self.compression_threshold = config.COMPRESSION_THRESHOLD
        self.pivot_left = config.PIVOT_LEFT_BARS
        self.pivot_right = config.PIVOT_RIGHT_BARS
        self.vol_multiplier = config.VOL_MULTIPLIER
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète de l'état quantique du marché
        
        Returns:
            Dict avec compression_state, is_low_entropy, quantum_barriers, signals
        """
        if self.df.empty or len(self.df) < self.compression_lookback:
            return self._empty_result()
        
        # 1. Calculer le Bollinger Bands Width (mesure de volatilité)
        self._calculate_bb_width()
        
        # 2. Détecter l'état de compression
        compression_data = self._analyze_compression()
        
        # 3. Trouver les Quantum Barriers (pivots)
        barriers = self._find_quantum_barriers()
        
        # 4. Détecter les sweeps
        sweeps = self._detect_sweeps(barriers)
        
        # 5. Volume spike detection
        vol_spike = self._detect_volume_spike()
        
        # 6. Générer les signaux Quantum
        signals = self._generate_quantum_signals(compression_data, sweeps, vol_spike)
        
        return {
            'compression': compression_data,
            'barriers': barriers,
            'sweeps': sweeps,
            'volume_spike': vol_spike,
            'signals': signals,
            'quantum_state': self._get_quantum_state_label(compression_data)
        }
    
    def _calculate_bb_width(self):
        """Calcule le Bollinger Bands Width"""
        self.df['bb_basis'] = self.df['close'].rolling(self.bb_length).mean()
        self.df['bb_dev'] = self.df['close'].rolling(self.bb_length).std()
        
        # BB Width = (2 * std) / basis
        self.df['bb_width'] = (self.df['bb_dev'] * 2) / self.df['bb_basis']
        
        # Moyenne historique (état normal)
        self.df['avg_width'] = self.df['bb_width'].rolling(self.compression_lookback).mean()
        
        # Compression state = current width / avg width
        # < 1.0 = comprimé, > 1.0 = volatil
        self.df['compression_state'] = self.df['bb_width'] / self.df['avg_width']
        
        # État AVANT le mouvement (5 bougies avant)
        self.df['pre_move_state'] = self.df['compression_state'].shift(5)
        
        # Flag low entropy
        self.df['is_low_entropy'] = self.df['compression_state'] < self.compression_threshold
    
    def _analyze_compression(self) -> Dict[str, Any]:
        """Analyse l'état de compression actuel"""
        current_compression = self.df['compression_state'].iloc[-1]
        pre_move_compression = self.df['pre_move_state'].iloc[-1]
        is_low_entropy = self.df['is_low_entropy'].iloc[-1]
        
        # Nombre de bougies en low entropy récemment
        low_entropy_count = self.df['is_low_entropy'].tail(50).sum()
        
        # Tendance de compression (compression croissante = danger)
        compression_trend = self.df['compression_state'].tail(10).mean() - self.df['compression_state'].tail(20).mean()
        
        return {
            'current': round(current_compression, 4) if pd.notna(current_compression) else 1.0,
            'pre_move': round(pre_move_compression, 4) if pd.notna(pre_move_compression) else 1.0,
            'is_low_entropy': bool(is_low_entropy) if pd.notna(is_low_entropy) else False,
            'low_entropy_count_50': int(low_entropy_count),
            'trend': round(compression_trend, 4) if pd.notna(compression_trend) else 0,
            'threshold': self.compression_threshold
        }
    
    def _find_quantum_barriers(self) -> Dict[str, Any]:
        """Trouve les Quantum Barriers (Pivot High/Low)"""
        pivot_highs = []
        pivot_lows = []
        
        left = self.pivot_left
        right = self.pivot_right
        
        highs = self.df['high']
        lows = self.df['low']
        
        for i in range(left, len(self.df) - right):
            # Pivot High
            window_high = highs.iloc[i-left:i+right+1]
            if highs.iloc[i] == window_high.max():
                pivot_highs.append({'index': i, 'price': highs.iloc[i]})
            
            # Pivot Low
            window_low = lows.iloc[i-left:i+right+1]
            if lows.iloc[i] == window_low.min():
                pivot_lows.append({'index': i, 'price': lows.iloc[i]})
        
        # Dernières barrières
        last_barrier_high = pivot_highs[-1]['price'] if pivot_highs else self.df['high'].max()
        last_barrier_low = pivot_lows[-1]['price'] if pivot_lows else self.df['low'].min()
        
        return {
            'high': round(last_barrier_high, 2),
            'low': round(last_barrier_low, 2),
            'all_highs': pivot_highs[-5:],  # 5 derniers
            'all_lows': pivot_lows[-5:]  # 5 derniers
        }
    
    def _detect_sweeps(self, barriers: Dict) -> Dict[str, Any]:
        """Détecte les sweeps des barrières quantiques"""
        current_high = self.df['high'].iloc[-1]
        current_low = self.df['low'].iloc[-1]
        current_close = self.df['close'].iloc[-1]
        current_open = self.df['open'].iloc[-1]
        
        barrier_high = barriers['high']
        barrier_low = barriers['low']
        
        # Sweep up: prix dépasse le high mais ferme en dessous
        sweep_up = (current_high > barrier_high) and (current_close < barrier_high)
        
        # Sweep down: prix dépasse le low mais ferme au dessus
        sweep_down = (current_low < barrier_low) and (current_close > barrier_low)
        
        # Rejection patterns
        rejection_bear = (current_close < barrier_high) and (current_close < current_open)
        rejection_bull = (current_close > barrier_low) and (current_close > current_open)
        
        return {
            'sweep_up': sweep_up,
            'sweep_down': sweep_down,
            'rejection_bear': rejection_bear,
            'rejection_bull': rejection_bull,
            'current_vs_barrier_high': round(current_high - barrier_high, 2),
            'current_vs_barrier_low': round(current_low - barrier_low, 2)
        }
    
    def _detect_volume_spike(self) -> Dict[str, Any]:
        """Détecte les spikes de volume"""
        if 'volume' not in self.df.columns:
            return {'has_spike': False, 'ratio': 1.0}
        
        current_vol = self.df['volume'].iloc[-1]
        vol_ma = self.df['volume'].rolling(20).mean().iloc[-1]
        
        ratio = current_vol / vol_ma if vol_ma > 0 else 1.0
        has_spike = ratio > self.vol_multiplier
        
        return {
            'has_spike': has_spike,
            'ratio': round(ratio, 2),
            'current': current_vol,
            'average': round(vol_ma, 2) if pd.notna(vol_ma) else 0
        }
    
    def _generate_quantum_signals(self, compression: Dict, sweeps: Dict, vol_spike: Dict) -> Dict[str, Any]:
        """
        Génère les signaux quantiques combinant compression + sweep + volume
        
        Signal QUANTUM = Low Entropy + Sweep + Volume Spike + Rejection
        """
        is_low_entropy = compression['is_low_entropy']
        has_vol_spike = vol_spike['has_spike']
        
        # Quantum SELL: Sweep up + Low Entropy + Volume Spike + Bear Rejection
        quantum_sell = (
            sweeps['sweep_up'] and 
            is_low_entropy and 
            has_vol_spike and 
            sweeps['rejection_bear']
        )
        
        # Quantum BUY: Sweep down + Low Entropy + Volume Spike + Bull Rejection
        quantum_buy = (
            sweeps['sweep_down'] and 
            is_low_entropy and 
            has_vol_spike and 
            sweeps['rejection_bull']
        )
        
        # Score de potentiel (0-4)
        potential_score = sum([
            is_low_entropy,
            sweeps['sweep_up'] or sweeps['sweep_down'],
            has_vol_spike,
            sweeps['rejection_bear'] or sweeps['rejection_bull']
        ])
        
        return {
            'quantum_buy': quantum_buy,
            'quantum_sell': quantum_sell,
            'potential_score': potential_score,
            'conditions_met': {
                'low_entropy': is_low_entropy,
                'sweep': sweeps['sweep_up'] or sweeps['sweep_down'],
                'volume_spike': has_vol_spike,
                'rejection': sweeps['rejection_bear'] or sweeps['rejection_bull']
            }
        }
    
    def _get_quantum_state_label(self, compression: Dict) -> str:
        """Retourne un label descriptif de l'état quantique"""
        current = compression['current']
        
        if current < 0.5:
            return "EXTREME_COMPRESSION"
        elif current < self.compression_threshold:
            return "LOW_ENTROPY"
        elif current < 1.0:
            return "SLIGHT_COMPRESSION"
        elif current < 1.5:
            return "NORMAL"
        else:
            return "HIGH_VOLATILITY"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Résultat vide en cas d'erreur"""
        return {
            'compression': {
                'current': 1.0,
                'pre_move': 1.0,
                'is_low_entropy': False,
                'low_entropy_count_50': 0,
                'trend': 0,
                'threshold': self.compression_threshold
            },
            'barriers': {'high': 0, 'low': 0, 'all_highs': [], 'all_lows': []},
            'sweeps': {
                'sweep_up': False, 'sweep_down': False,
                'rejection_bear': False, 'rejection_bull': False,
                'current_vs_barrier_high': 0, 'current_vs_barrier_low': 0
            },
            'volume_spike': {'has_spike': False, 'ratio': 1.0},
            'signals': {
                'quantum_buy': False, 'quantum_sell': False,
                'potential_score': 0, 'conditions_met': {}
            },
            'quantum_state': 'UNKNOWN'
        }
