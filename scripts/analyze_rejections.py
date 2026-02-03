import pandas as pd
import ast

DATASET_FILE = "dataset.csv"

def analyze_rejections():
    print(f"📊 Loading {DATASET_FILE}...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter for recent signals (last 100)
    recent = df.tail(100).copy()
    
    # Filter for rejected trades (Confidence < 60)
    # Note: 'NO_SIGNAL' might have confidence 0
    rejected = recent[recent['signal_confidence'] < 60].copy()
    total = len(recent)
    rejected_count = len(rejected)
    
    print(f"\n--- RECENT ACTIVITY (Last {total} Snapshots) ---")
    print(f"Total Snapshots: {total}")
    print(f"Rejected/Weak Signals (<60%): {rejected_count} ({rejected_count/total*100:.1f}%)")
    
    if rejected_count == 0:
        print("All signals are STRONG! (Check thresholds)")
        return
        
    print(f"\n--- BOTTLENECK ANALYSIS (Average Scores on Rejection) ---")
    # Technical, Structure, Multi-Exchange, Derivatives, OnChain, Sentiment, Macro
    dims = {
        'Technical': 'dimension_scores_technical',
        'Structure': 'dimension_scores_structure',
        'Multi-Exchange': 'dimension_scores_multi_exchange',
        'Derivatives': 'dimension_scores_derivatives',
        'OnChain': 'dimension_scores_onchain',
        'Sentiment': 'dimension_scores_sentiment',
        'Macro': 'dimension_scores_macro'
    }
    
    # Actual columns available?
    available_cols = []
    for d, col_guess in dims.items():
        if col_guess in rejected.columns:
            available_cols.append((d, col_guess))
            
    for dim_name, col in available_cols:
        avg_score = rejected[col].mean()
        print(f"{dim_name:<15}: {avg_score:.1f} / 100")
        
    print(f"\n--- OPPORTUNITY COST (Did we miss moves?) ---")
    # Check 'future_4h_return_pct' for rejected trades
    # Big moves (> 2%) that we missed
    missed_opportunities = rejected[rejected['future_4h_return_pct'].abs() > 2.0]
    
    print(f"Missed Big Moves (>2%): {len(missed_opportunities)}")
    
    if len(missed_opportunities) > 0:
        print("\nTop 5 Missed Opportunities:")
        cols_to_show = ['timestamp', 'price', 'signal_type', 'signal_confidence', 'future_4h_return_pct'] + [c[1] for c in available_cols]
        # Handle missing columns gracefully
        valid_cols = [c for c in cols_to_show if c in missed_opportunities.columns]
        print(missed_opportunities[valid_cols].head(5).to_string())

if __name__ == "__main__":
    analyze_rejections()
