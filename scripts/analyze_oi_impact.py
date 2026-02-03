import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

DATASET_FILE = "dataset.csv"
OUTPUT_REPORT = "oi_impact_report.md"

def analyze_oi_impact():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset not found: {DATASET_FILE}")
        return

    print(f"📊 Loading {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
    # Ensure necessary columns exist
    required_cols = [
        'indicators_open_interest_delta_1h_delta_oi_pct',
        'indicators_open_interest_delta_4h_delta_oi_pct',
        'future_4h_return_pct',
        'future_4h_volatility_pct',
        'indicators_adx_trend_direction' # For Regime segmentation
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return

    # Filter valid rows
    df = df.dropna(subset=required_cols).copy()
    print(f"✅ Analyzing {len(df)} snapshots...")

    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 🔬 Open Interest (OI) Impact Analysis\n\n")
        f.write("Does OI change predict future price movement? \n\n")

        # --- 1. GLOBAL CORRELATIONS ---
        f.write("## 1. Global Correlations (All Regimes)\n")
        
        # OI Delta 1h vs Future Return 4h
        corr_dir = df['indicators_open_interest_delta_1h_delta_oi_pct'].corr(df['future_4h_return_pct'])
        # OI Delta 1h vs Future Volatility 4h
        corr_vol = df['indicators_open_interest_delta_1h_delta_oi_pct'].corr(df['future_4h_volatility_pct'])
        
        f.write(f"- **OI Change (1h) vs Future Price Direction**: `{corr_dir:.3f}`\n")
        f.write(f"- **OI Change (1h) vs Future Volatility**: `{corr_vol:.3f}`\n\n")
        
        if abs(corr_dir) > 0.2:
            f.write("👉 **Significant Directional Link**: ")
            f.write("Rising OI leads to Price UP" if corr_dir > 0 else "Rising OI leads to Price DOWN")
            f.write("\n\n")
        
        # --- 2. REGIME SEGMENTATION (Bull vs Bear) ---
        f.write("## 2. Regime Segmentation (Bull vs Bear)\n")
        f.write("Does the market react differently when trending UP vs DOWN?\n\n")
        
        modes = df['indicators_adx_trend_direction'].unique()
        
        for mode in ['BULLISH', 'BEARISH']:
            subset = df[df['indicators_adx_trend_direction'] == mode]
            if len(subset) < 50: continue
            
            c_dir = subset['indicators_open_interest_delta_1h_delta_oi_pct'].corr(subset['future_4h_return_pct'])
            c_vol = subset['indicators_open_interest_delta_1h_delta_oi_pct'].corr(subset['future_4h_volatility_pct'])
            
            f.write(f"### Regime: {mode} ({len(subset)} samples)\n")
            f.write(f"- Correlation OI -> Direction: `{c_dir:.3f}`\n")
            f.write(f"- Correlation OI -> Volatility: `{c_vol:.3f}`\n")
            
            # Narrative (User Question: Inverted situations?)
            interpretation = "Neutral"
            if c_dir > 0.15: interpretation = "Trend Continuation (Fuel)"
            elif c_dir < -0.15: interpretation = "Reversion / Trap"
            
            f.write(f"- **Interpretation**: {interpretation}\n\n")

        # --- 3. EXTREME OI EVENTS ---
        f.write("## 3. Extreme OI Spikes (> +2% in 1h)\n")
        spikes = df[df['indicators_open_interest_delta_1h_delta_oi_pct'] > 2.0]
        
        if len(spikes) > 0:
            avg_return = spikes['future_4h_return_pct'].mean()
            win_rate = (spikes['future_4h_return_pct'] > 0).mean() * 100
            
            f.write(f"When OI surges > 2% in 1 hour ({len(spikes)} cases):\n")
            f.write(f"- **Average Next 4h Return**: `{avg_return:.2f}%`\n")
            f.write(f"- **Probability of Pump**: `{win_rate:.1f}%`\n")
        else:
            f.write("No extreme OI spikes (>2%) detected in dataset.\n")

    print(f"✅ Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_oi_impact()
