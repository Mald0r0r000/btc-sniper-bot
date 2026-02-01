import sys
import os
# Add project root to path (one level up from scripts)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from exchange import BitgetConnector
from analyzers.adx_analyzer import ADXAnalyzer
import pandas as pd

def debug_adx():
    print("🔍 Debugging ADX Calculation...")
    connector = BitgetConnector()
    
    # Fetch 1H candles (Meso)
    print("📡 Fetching 1H candles...")
    df = connector.fetch_ohlcv('1h', limit=100)
    
    if df is None or len(df) == 0:
        print("❌ Failed to fetch data")
        return

    print(f"📊 {len(df)} candles fetched.")
    
    analyzer = ADXAnalyzer(df['high'], df['low'], df['close'], period=14)
    result = analyzer.analyze()
    
    print("\n🧮 Results:")
    print(f"   ADX: {result['adx']}")
    print(f"   +DI: {result['plus_di']}")
    print(f"   -DI: {result['minus_di']}")
    print(f"   Regime: {result['regime']}")
    print(f"   Trend: {result['trend_direction']}")
    
    # Verify calculation manually for last candle
    plus = result['plus_di']
    minus = result['minus_di']
    if plus + minus > 0:
        dx = 100 * abs(plus - minus) / (plus + minus)
        print(f"   Calculated DX (snapshot): {dx:.2f}")
    
    # Show last 5 values to see smoothing
    full_df = analyzer.calculate_adx()
    print("\n📉 Last 5 periods:")
    print(full_df.tail(5)[['adx', 'plus_di', 'minus_di']])

if __name__ == "__main__":
    debug_adx()
