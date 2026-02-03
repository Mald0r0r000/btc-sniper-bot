import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATASET_FILE = "dataset.csv"
OUTPUT_REPORT = "performance_report.md"

def analyze_performance():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Dataset not found: {DATASET_FILE}")
        return

    print(f"📊 Loading {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
    # Filter for finalized outcomes only
    completed_trades = df[df['outcome_result'].isin(['WIN', 'LOSS'])]
    
    if len(completed_trades) == 0:
        print("⚠️ No completed trades found in dataset.")
        return

    # Basic Stats
    total_trades = len(completed_trades)
    wins = len(completed_trades[completed_trades['outcome_result'] == 'WIN'])
    winrate = (wins / total_trades) * 100
    
    avg_pnl = completed_trades['outcome_pnl'].mean()
    
    print(f"📈 Analyzed {total_trades} trades.")
    print(f"🏆 Overall Winrate: {winrate:.2f}%")
    print(f"💰 Avg PnL per trade: {avg_pnl:.2f}%")
    
    # Generate Report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 Blackbox Performance Analysis\n\n")
        f.write(f"- **Total Signal Count**: {len(df)}\n")
        f.write(f"- **Completed Trades**: {total_trades}\n")
        f.write(f"- **Global Winrate**: {winrate:.2f}%\n")
        f.write(f"- **Average PnL**: {avg_pnl:.2f}%\n\n")
        
        f.write("## 🔍 Segmentation by Signal Type\n\n")
        type_stats = completed_trades.groupby('signal_type')['outcome_result'].value_counts(normalize=True).unstack().fillna(0)
        if 'WIN' in type_stats.columns:
            type_stats['Winrate'] = type_stats['WIN'] * 100
            f.write(type_stats[['Winrate']].sort_values('Winrate', ascending=False).to_markdown())
            f.write("\n\n")
            
        f.write("## 🧠 High Confidence Analysis (>60%)\n\n")
        high_conf = completed_trades[completed_trades['signal_confidence'] >= 60]
        if len(high_conf) > 0:
            hc_wins = len(high_conf[high_conf['outcome_result'] == 'WIN'])
            hc_winrate = (hc_wins / len(high_conf)) * 100
            f.write(f"- **High Confidence Trades**: {len(high_conf)}\n")
            f.write(f"- **Winrate**: {hc_winrate:.2f}%\n")
        else:
            f.write("No high confidence trades found.\n")

        f.write("\n## 🐳 Whale Analysis (Hyperliquid)\n\n")
        # Check simple correlation: Whale Count vs Outcome
        if 'hyperliquid_whale_analysis_whale_count' in df.columns:
            whales = completed_trades.groupby('outcome_result')['hyperliquid_whale_analysis_whale_count'].mean()
            f.write("Average Whale Count per Outcome:\n")
            f.write(whales.to_markdown())
            
    print(f"✅ Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_performance()
