import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

DATASET_FILE = "dataset.csv"
OUTPUT_REPORT = "correlation_report.md"

def analyze_correlations():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset not found: {DATASET_FILE}")
        return

    print(f"📊 Loading {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
    # Filter for finalized outcomes only
    completed_trades = df[df['outcome_result'].isin(['WIN', 'LOSS'])].copy()
    
    if len(completed_trades) < 10:
        print("⚠️ Not enough completed trades for correlation analysis (<10).")
        return

    # Encode Result: WIN=1, LOSS=0
    completed_trades['target'] = completed_trades['outcome_result'].apply(lambda x: 1 if x == 'WIN' else 0)
    
    # Select feature columns (numerical indicators)
    # We explicitly exclude non-predictive columns like timestamps, IDs, etc.
    feature_cols = [c for c in completed_trades.columns if 
                    c.startswith('indicators_') and 
                    pd.api.types.is_numeric_dtype(completed_trades[c])]
    
    print(f"🔍 Analyzing correlations for {len(feature_cols)} features against {len(completed_trades)} trades...")
    
    correlations = []
    
    for col in feature_cols:
        # Skip columns with 0 variance (constants)
        if completed_trades[col].nunique() <= 1:
            continue
            
        # Point-Biserial Correlation (Pearson for binary target)
        corr = completed_trades[col].corr(completed_trades['target'])
        
        if not np.isnan(corr):
            correlations.append({
                'Feature': col.replace('indicators_', ''),
                'Correlation': corr,
                'AbsCorrelation': abs(corr)
            })
    
    # Create DataFrame and sort
    corr_df = pd.DataFrame(correlations)
    
    if corr_df.empty:
        print("⚠️ No valid correlations found.")
        return
        
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    # Generate Report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 🧬 Deep Correlation Analysis (Fine-Tuning)\n\n")
        f.write(f"Analysis based on **{len(completed_trades)}** completed trades.\n\n")
        f.write("### 🟢 Top Positive Correlations (Predictors of WIN)\n")
        f.write("Higher values of these indicators increase probability of WIN.\n\n")
        f.write(corr_df.head(20)[['Feature', 'Correlation']].to_markdown(index=False))
        
        f.write("\n\n### 🔴 Top Negative Correlations (Predictors of LOSS)\n")
        f.write("Higher values of these indicators increase probability of LOSS.\n\n")
        f.write(corr_df.tail(20).sort_values('Correlation', ascending=True)[['Feature', 'Correlation']].to_markdown(index=False))
        
        f.write("\n\n### ⚖️ Feature Importance (Absolute Impact)\n")
        f.write("These features have the strongest signal, regardless of direction.\n\n")
        f.write(corr_df.sort_values('AbsCorrelation', ascending=False).head(20)[['Feature', 'Correlation']].to_markdown(index=False))

    print(f"✅ Correlation report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_correlations()
