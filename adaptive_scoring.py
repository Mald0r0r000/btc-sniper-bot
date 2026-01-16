"""
Adaptive Scoring Layer
Ajuste dynamiquement les poids du Decision Engine basé sur les performances passées
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timezone


@dataclass
class DimensionPerformance:
    """Statistiques de performance pour une dimension"""
    name: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    avg_score_on_win: float = 50.0
    avg_score_on_loss: float = 50.0
    correlation_with_outcome: float = 0.0
    recommended_weight_adjustment: float = 0.0


class AdaptiveScoringLayer:
    """
    Analyse les signaux validés pour ajuster les poids du Decision Engine
    
    Concept:
    - Analyser quelles dimensions corrèlent avec les WIN
    - Identifier les dimensions qui génèrent des faux positifs
    - Proposer des ajustements de poids
    
    Limite les ajustements à ±10% pour éviter l'overfitting
    """
    
    # Dimensions du Decision Engine V2
    DIMENSIONS = [
        'technical', 'structure', 'multi_exchange', 
        'derivatives', 'onchain', 'sentiment', 'macro'
    ]
    
    # Poids de base (doivent correspondre à decision_engine_v2.py)
    BASE_WEIGHTS = {
        'technical': 25,
        'structure': 15,
        'multi_exchange': 10,
        'derivatives': 15,
        'onchain': 15,
        'sentiment': 10,
        'macro': 10
    }
    
    # Limite max d'ajustement par dimension (en points)
    MAX_ADJUSTMENT = 3  # ±3 points (ex: 25% -> 22-28%)
    
    def __init__(self):
        self.performance_data: Dict[str, DimensionPerformance] = {}
        self.adjusted_weights: Dict[str, int] = self.BASE_WEIGHTS.copy()
        
    def analyze_signals(self, validated_signals: List[Dict]) -> Dict[str, DimensionPerformance]:
        """
        Analyse les signaux validés pour calculer la performance de chaque dimension
        """
        # Initialiser les structures
        dim_scores_win = {dim: [] for dim in self.DIMENSIONS}
        dim_scores_loss = {dim: [] for dim in self.DIMENSIONS}
        
        for signal in validated_signals:
            validation = signal.get('validation', {})
            status = validation.get('status')
            
            if status not in ['WIN', 'LOSS']:
                continue
            
            # Récupérer les scores par dimension
            dim_scores = signal.get('dimension_scores', {})
            
            for dim in self.DIMENSIONS:
                score = dim_scores.get(dim, 50)
                if status == 'WIN':
                    dim_scores_win[dim].append(score)
                else:
                    dim_scores_loss[dim].append(score)
        
        # Calculer les statistiques pour chaque dimension
        for dim in self.DIMENSIONS:
            wins_scores = dim_scores_win[dim]
            loss_scores = dim_scores_loss[dim]
            
            perf = DimensionPerformance(
                name=dim,
                total_signals=len(wins_scores) + len(loss_scores),
                wins=len(wins_scores),
                losses=len(loss_scores),
                avg_score_on_win=np.mean(wins_scores) if wins_scores else 50,
                avg_score_on_loss=np.mean(loss_scores) if loss_scores else 50
            )
            
            # Calculer la corrélation avec l'outcome
            if wins_scores and loss_scores:
                # Score moyen sur WIN - Score moyen sur LOSS
                # Positif = cette dimension prédit bien les WIN
                score_diff = perf.avg_score_on_win - perf.avg_score_on_loss
                perf.correlation_with_outcome = round(score_diff, 2)
                
                # Recommandation d'ajustement
                # Si corrélation positive forte → augmenter le poids
                # Si corrélation négative → réduire le poids
                adjustment = np.clip(score_diff / 10, -self.MAX_ADJUSTMENT, self.MAX_ADJUSTMENT)
                perf.recommended_weight_adjustment = round(adjustment, 1)
            
            self.performance_data[dim] = perf
        
        return self.performance_data
    
    def calculate_adjusted_weights(self) -> Dict[str, int]:
        """
        Calcule les nouveaux poids ajustés basés sur l'analyse
        """
        total_base = sum(self.BASE_WEIGHTS.values())
        
        for dim, perf in self.performance_data.items():
            base = self.BASE_WEIGHTS.get(dim, 10)
            adjustment = perf.recommended_weight_adjustment
            
            # Appliquer l'ajustement
            new_weight = base + adjustment
            
            # Limiter à un minimum de 5%
            new_weight = max(5, min(30, new_weight))
            
            self.adjusted_weights[dim] = round(new_weight)
        
        # Normaliser pour que le total = 100
        total = sum(self.adjusted_weights.values())
        if total != total_base:
            # Ajuster proportionnellement
            factor = total_base / total
            self.adjusted_weights = {
                k: round(v * factor) for k, v in self.adjusted_weights.items()
            }
        
        return self.adjusted_weights
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        Génère des recommandations basées sur l'analyse
        """
        if not self.performance_data:
            return {'error': 'No performance data. Run analyze_signals first.'}
        
        # Trier par corrélation
        sorted_dims = sorted(
            self.performance_data.values(),
            key=lambda x: x.correlation_with_outcome,
            reverse=True
        )
        
        best_predictors = [d for d in sorted_dims if d.correlation_with_outcome > 5]
        worst_predictors = [d for d in sorted_dims if d.correlation_with_outcome < -5]
        
        recommendations = []
        
        for dim in best_predictors:
            recommendations.append(
                f"✅ {dim.name.upper()}: Forte corrélation avec WIN (+{dim.correlation_with_outcome:.0f}). "
                f"Recommandation: augmenter poids de +{dim.recommended_weight_adjustment:.0f}%"
            )
        
        for dim in worst_predictors:
            recommendations.append(
                f"⚠️ {dim.name.upper()}: Corrélation négative ({dim.correlation_with_outcome:.0f}). "
                f"Recommandation: réduire poids de {dim.recommended_weight_adjustment:.0f}%"
            )
        
        return {
            'summary': {
                'best_predictor': sorted_dims[0].name if sorted_dims else None,
                'worst_predictor': sorted_dims[-1].name if sorted_dims else None,
                'total_signals_analyzed': sum(d.total_signals for d in self.performance_data.values()) // len(self.DIMENSIONS)
            },
            'dimension_analysis': {
                d.name: {
                    'wins': d.wins,
                    'losses': d.losses,
                    'avg_on_win': round(d.avg_score_on_win, 1),
                    'avg_on_loss': round(d.avg_score_on_loss, 1),
                    'correlation': d.correlation_with_outcome,
                    'weight_adjustment': d.recommended_weight_adjustment
                } for d in self.performance_data.values()
            },
            'current_weights': self.BASE_WEIGHTS,
            'adjusted_weights': self.adjusted_weights,
            'recommendations': recommendations
        }
    
    def generate_weight_config(self) -> str:
        """
        Génère le code Python pour les nouveaux poids
        Peut être copié dans decision_engine_v2.py
        """
        lines = ["# Poids ajustés basés sur analyse de performance"]
        lines.append("WEIGHT_CONFIG = {")
        for dim, weight in self.adjusted_weights.items():
            change = weight - self.BASE_WEIGHTS.get(dim, 0)
            change_str = f"+{change}" if change > 0 else str(change)
            lines.append(f"    '{dim}': {weight},  # base {self.BASE_WEIGHTS.get(dim, 0)} ({change_str})")
        lines.append("}")
        return "\n".join(lines)


def test_adaptive_scoring():
    """Test avec des signaux simulés"""
    # Signaux simulés avec validation
    test_signals = [
        {
            'dimension_scores': {'technical': 75, 'structure': 60, 'derivatives': 80, 'onchain': 55, 'sentiment': 50, 'macro': 60, 'multi_exchange': 45},
            'validation': {'status': 'WIN'}
        },
        {
            'dimension_scores': {'technical': 70, 'structure': 65, 'derivatives': 75, 'onchain': 60, 'sentiment': 45, 'macro': 55, 'multi_exchange': 50},
            'validation': {'status': 'WIN'}
        },
        {
            'dimension_scores': {'technical': 55, 'structure': 70, 'derivatives': 45, 'onchain': 40, 'sentiment': 65, 'macro': 50, 'multi_exchange': 55},
            'validation': {'status': 'LOSS'}
        },
        {
            'dimension_scores': {'technical': 45, 'structure': 55, 'derivatives': 50, 'onchain': 45, 'sentiment': 70, 'macro': 45, 'multi_exchange': 40},
            'validation': {'status': 'LOSS'}
        },
    ]
    
    layer = AdaptiveScoringLayer()
    layer.analyze_signals(test_signals)
    layer.calculate_adjusted_weights()
    
    recs = layer.get_recommendations()
    
    print("=== Analyse Adaptive Scoring ===")
    print(f"\nMeilleur prédicteur: {recs['summary']['best_predictor']}")
    print(f"Pire prédicteur: {recs['summary']['worst_predictor']}")
    
    print("\nAjustements recommandés:")
    for rec in recs['recommendations']:
        print(f"  {rec}")
    
    print(f"\nPoids actuels: {recs['current_weights']}")
    print(f"Poids ajustés: {recs['adjusted_weights']}")
    
    print("\n" + layer.generate_weight_config())


if __name__ == "__main__":
    test_adaptive_scoring()
