import sys
import os
from pprint import pprint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange_aggregator import MultiExchangeAggregator
import config

def verify_oi():
    print("🚀 Starting Open Interest Verification...")
    
    # Disable sensitive API usage
    config.API_KEY = '' 
    
    try:
        # 1. Initialize Aggregator
        print("\n1. Initializing MultiExchangeAggregator...")
        aggregator = MultiExchangeAggregator()
        status = aggregator.get_connection_status()
        print(f"   Exchanges: {len(status['connected'])} connected / {len(status['requested'])} requested")

        # 2. Fetch Aggregated Data (triggers OI fetch)
        print("\n2. Fetching Consolidated Data...")
        data = aggregator.get_aggregated_data()
        
        oi_data = data.get('open_interest', {})
        total_oi = oi_data.get('total_btc', 0)
        by_exchange = oi_data.get('by_exchange', {})
        
        print(f"\n📊 Total Aggregated OI: {total_oi:,.2f} BTC")
        print("\n📋 Details by Exchange:")
        print(f"{'Exchange':<15} | {'OI (BTC)':<15}")
        print("-" * 35)
        
        for ex, val in by_exchange.items():
            print(f"{ex:<15} | {val:<15.2f}")
            
        # Check if total is reasonable (should be < 1M)
        if total_oi > 2000000:
             print("\n❌ WARNING: Total OI seems suspiciously high (> 2M BTC). Normalization might be failing.")
        else:
             print("\n✅ Total OI seems reasonable (< 2M BTC).")

    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_oi()
