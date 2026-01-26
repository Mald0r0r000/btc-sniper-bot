import sys
import os
from pprint import pprint

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange_aggregator import MultiExchangeAggregator
from analyzers.spot_perp_divergence import SpotPerpDivergenceAnalyzer
import config

def verify_spot_perp():
    print("🚀 Starting Spot/Perp Verification...")
    
    # Disable sensitive API usage (we just want public data)
    config.API_KEY = '' 
    
    try:
        # 1. Initialize Aggregator
        print("\n1. Initializing MultiExchangeAggregator...")
        aggregator = MultiExchangeAggregator()
        status = aggregator.get_connection_status()
        print(f"   Exchanges: {len(status['connected'])} connected / {len(status['requested'])} requested")

        # 2. Fetch Global CVD Data
        print("\n2. Fetching Global CVD Data (1h candles)...")
        # Use a small limit to be fast
        global_data = aggregator.fetch_global_cvd_candles(timeframes=['1h', '4h'])
        
        spot_count = len(global_data.get('1h', {}).get('spot', {}))
        perp_count = len(global_data.get('1h', {}).get('swap', {}))
        
        print(f"   Received Data: {spot_count} Spot sources | {perp_count} Perp sources")
        
        if spot_count == 0 or perp_count == 0:
            print("   ❌ ERROR: Not enough data sources.")
            return
            
        # 3. Analyze Divergence
        print("\n3. Inspecting Raw Volumes (Last Candle)...")
        
        # Check 1H data
        data_1h = global_data.get('1h', {})
        
        print(f"{'Exchange':<15} | {'Type':<6} | {'Price':<10} | {'Volume':<15} | {'Vol/Price':<10}")
        print("-" * 70)
        
        for m_type in ['spot', 'swap']:
            for ex, candles in data_1h.get(m_type, {}).items():
                if candles:
                    c = candles[-1]
                    price = c['close']
                    vol = c['volume']
                    ratio = vol / price if price else 0
                    print(f"{ex:<15} | {m_type:<6} | {price:<10.1f} | {vol:<15.1f} | {ratio:<10.4f}")

        print("\n4. Running SpotPerpDivergenceAnalyzer...")
        analyzer = SpotPerpDivergenceAnalyzer()
        results = analyzer.analyze(global_data)
        
        print("\n📊 ANALYSIS RESULTS:")
        pprint(results)
        
        # 4. specific checks
        res_1h = results.get('1h', {})
        print(f"\n🔍 Check 1H:")
        print(f"   Regime: {res_1h.get('regime')}")
        print(f"   Signal: {res_1h.get('signal')}")
        print(f"   Divergence: {res_1h.get('divergence')}")
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_spot_perp()
