import sys
import unittest
from decision_engine_v2 import DecisionEngineV2

class TestDecisionEngineKDJ(unittest.TestCase):
    def test_kdj_scoring(self):
        print("\n=== Testing KDJ Scoring Logic ===")
        
        # Base case: Neutral KDJ (J=50)
        engine_neutral = DecisionEngineV2(
            current_price=100000,
            kdj_data={'values': {'j': 50, 'j_slope': 0}, 'signal': 'NEUTRAL'},
            trading_style='default' # Default style uses default weights
        )
        score_neutral = engine_neutral._score_technical()
        print(f"J=50 (Neutral) Score: {score_neutral}")
        
        # Test Case 1: High J (Overbought) -> Should be significantly LOWER than neutral
        engine_high = DecisionEngineV2(
            current_price=100000,
            kdj_data={'values': {'j': 95, 'j_slope': -1}, 'signal': 'NEUTRAL'}, # Turning down
            trading_style='default'
        )
        score_high = engine_high._score_technical()
        print(f"J=95 (Overbought) Score: {score_high}")
        
        # Test Case 2: Low J (Oversold) -> Should be significantly HIGHER than neutral
        engine_low = DecisionEngineV2(
            current_price=100000,
            kdj_data={'values': {'j': 5, 'j_slope': 1}, 'signal': 'NEUTRAL'}, # Turning up
            trading_style='default'
        )
        score_low = engine_low._score_technical()
        print(f"J=5 (Oversold) Score: {score_low}")
        
        # Verification
        self.assertLess(score_high, score_neutral, "High J should lower score")
        self.assertGreater(score_low, score_neutral, "Low J should boost score")
        self.assertLess(score_high, 40, "J=95 should result in bearish score (<40)")
        self.assertGreater(score_low, 60, "J=5 should result in bullish score (>60)")
        
        print("✅ Logic Correct: High J punishes score, Low J boosts score.")

if __name__ == '__main__':
    unittest.main()
