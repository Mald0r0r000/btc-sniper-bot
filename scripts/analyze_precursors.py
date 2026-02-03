import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

DATASET_FILE = "dataset.csv"
OUTPUT_REPORT = "precursor_report.md"

def analyze_precursors():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset not found: {DATASET_FILE}")
        return

    print(f"📊 Loading {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
    # Needs Future Return columns
    if 'future_4h_return_pct' not in df.columns:
        print("❌ Dataset missing 'future_4h_return_pct'. Please re-run build_dataset.py.")
        return
        
    # Filter valid rows (where future is known)
    valid_df = df.dropna(subset=['future_4h_return_pct', 'future_4h_volatility_pct']).copy()
    
    if len(valid_df) < 50:
        print(f"⚠️ Not enough data points ({len(valid_df)}) for meaningful analysis.")
        return

    # Select features (numerical indicators)
    feature_cols = [c for c in valid_df.columns if 
                    c.startswith('indicators_') and 
                    pd.api.types.is_numeric_dtype(valid_df[c])]

    print(f"🔍 Searching for precursors in {len(valid_df)} snapshots...")
    
    # 1. Volatility Precursors (What predicts a big move?)
    vol_correlations = []
    
    # 2. Directional Precursors (What predicts UP vs DOWN?)
    dir_correlations = []
    
    for col in feature_cols:
        if valid_df[col].nunique() <= 1: continue
            
        # Correlate with Future Volatility
        vol_corr = valid_df[col].corr(valid_df['future_4h_volatility_pct'])
        if not np.isnan(vol_corr):
            vol_correlations.append({'Feature': col.replace('indicators_', ''), 'Correlation': vol_corr, 'Abs': abs(vol_corr)})
            
        # Correlate with Future Return
        dir_corr = valid_df[col].corr(valid_df['future_4h_return_pct'])
        if not np.isnan(dir_corr):
            dir_correlations.append({'Feature': col.replace('indicators_', ''), 'Correlation': dir_corr, 'Abs': abs(dir_corr)})
            
    # Create DFs
    vol_df = pd.DataFrame(vol_correlations).sort_values('Abs', ascending=False)
    dir_df = pd.DataFrame(dir_correlations).sort_values('Abs', ascending=False)
    
    # Generate Report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 🔮 Market Precursor Analysis (Alpha Discovery)\n\n")
        f.write(f"Based on **{len(valid_df)}** market snapshots (15m intervals).\n")
        f.write("Analysis focuses on **Future 4h Movements** independent of bot signals.\n\n")
        
        f.write("## 🌪️ Volatility Precursors (Predictors of Big Moves)\n")
        f.write("These indicators signal that a big move is coming (up OR down).\n\n")
        f.write(vol_df.head(20)[['Feature', 'Correlation']].to_markdown(index=False))
        
        f.write("\n\n## 🧭 Directional Precursors (Predictors of Price)\n")
        f.write("These indicators signal specific direction (Positive = Bullish, Negative = Bearish).\n\n")
        f.write(dir_df.head(20)[['Feature', 'Correlation']].to_markdown(index=False))
        
        f.write("\n\n### 💡 Insight Generator\n")
        # Generate some narrative insights based on top correlations
        top_bull = dir_df.sort_values('Correlation', ascending=False).iloc[0]
        top_bear = dir_df.sort_values('Correlation', ascending=True).iloc[0]
        
        f.write(f"- **Top Bullish Signal**: `{top_bull['Feature']}` (Corr: {top_bull['Correlation']:.2f})\n")
        f.write(f"- **Top Bearish Signal**: `{top_bear['Feature']}` (Corr: {top_bear['Correlation']:.2f})\n")
        
        top_vol = vol_df.iloc[0]
        f.write(f"- **Top Volatility Warning**: `{top_vol['Feature']}` (Corr: {top_vol['Correlation']:.2f})\n")

    print(f"✅ Precursor report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_precursors()
