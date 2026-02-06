import sys
import csv
import math

# Add project root to path
sys.path.append("/Users/antoinebedos/ownapps/ANTIGRAVITY/btc-sniper-bot")

class KalmanFilter1D:
    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.x = [0.0, 0.0]
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.dt = 1.0
        q_pos = (self.dt**4)/4 * process_noise
        q_vel = self.dt**2 * process_noise
        self.Q = [
            [q_pos, (self.dt**3)/2 * process_noise],
            [(self.dt**3)/2 * process_noise, q_vel]
        ]
        self.R = measurement_noise
        self.initialized = False

    def update(self, measurement: float) -> tuple:
        if not self.initialized:
            self.x = [float(measurement), 0.0]
            self.initialized = True
            return measurement, 0.0
        
        # 1. Predict
        x_pred_0 = self.x[0] + self.x[1] * self.dt
        x_pred_1 = self.x[1]
        
        p00 = self.P[0][0] + self.P[1][0] * self.dt
        p01 = self.P[0][1] + self.P[1][1] * self.dt
        p10 = self.P[1][0]
        p11 = self.P[1][1]
        
        pp00 = p00 + p01 * self.dt
        pp01 = p01
        pp10 = p10 + p11 * self.dt
        pp11 = p11
        
        P_pred = [
            [pp00 + self.Q[0][0], pp01 + self.Q[0][1]],
            [pp10 + self.Q[1][0], pp11 + self.Q[1][1]]
        ]
        
        # 2. Update
        y = measurement - x_pred_0
        S = P_pred[0][0] + self.R
        K = [P_pred[0][0] / S, P_pred[1][0] / S]
        
        self.x[0] = x_pred_0 + K[0] * y
        self.x[1] = x_pred_1 + K[1] * y
        
        i_kh_00 = 1 - K[0]
        i_kh_10 = -K[1]
        
        one = i_kh_00 * P_pred[0][0]
        two = i_kh_00 * P_pred[0][1]
        three = i_kh_10 * P_pred[0][0] + 1 * P_pred[1][0]
        four = i_kh_10 * P_pred[0][1] + 1 * P_pred[1][1]
        
        self.P = [[one, two], [three, four]]
        
        return self.x[0], self.x[1]

HAS_KALMAN = True

def calculate_ema(prices, span):
    if not prices: return []
    alpha = 2 / (span + 1)
    ema_values = [prices[0]]
    for p in prices[1:]:
        ema_values.append(alpha * p + (1 - alpha) * ema_values[-1])
    return ema_values

def run_benchmark():
    # Load user CSV
    csv_path = "/Users/antoinebedos/Downloads/BTC_Sniper_Bot - Sheet1(7).csv"
    
    prices = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"CSV Headers: {headers}")
            
            # Identify Price Column
            price_col = None
            for h in headers:
                if 'close' in h.lower() or 'price' in h.lower():
                    price_col = h
                    break
            
            if not price_col:
                # Fallback: Guess 2nd column? Or hardcode 'Price'
                if 'Price' in headers: price_col = 'Price'
                elif 'Close' in headers: price_col = 'Close'
                else: 
                     print("Could not identify Price column.")
                     return
            
            print(f"Using Price Column: {price_col}")
            
            for row in reader:
                val = row.get(price_col)
                if val:
                    try:
                        prices.append(float(val.replace(',', '')))
                    except:
                        pass
                        
        print(f"Loaded {len(prices)} price points.")
            
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Run Kalman (Aggressive Tuning)
    # Q=0.05 (High process noise = nimble)
    # R=0.1 (Low measurement noise = trust price)
    kf = KalmanFilter1D(process_noise=0.05, measurement_noise=0.1)
    kalman_estimates = []
    kalman_velocities = []
    
    for p in prices:
        est_price, est_vel = kf.update(float(p))
        kalman_estimates.append(est_price)
        kalman_velocities.append(est_vel)
        
    # Calculate EMAs
    ema9 = calculate_ema(prices, 9)
    
    # Analyze Reversal
    # Find global min
    min_price = min(prices)
    min_idx = prices.index(min_price)
    
    print(f"\n--- Reversal Analysis ---")
    print(f"Bottom found at Row {min_idx}: ${min_price}")
    
    # Check Flips
    ema_flip = -1
    kf_flip = -1
    
    for i in range(min_idx, len(prices)):
        p = prices[i]
        
        if ema_flip == -1 and p > ema9[i]:
            ema_flip = i
            
        if kf_flip == -1 and p > kalman_estimates[i]:
            kf_flip = i
            
        if ema_flip != -1 and kf_flip != -1:
            break
            
    print(f"[EMA 9] Flipped at Row {ema_flip} (Lag: {ema_flip - min_idx})")
    print(f"[Kalman] Flipped at Row {kf_flip} (Lag: {kf_flip - min_idx})")
    
    if kf_flip < ema_flip:
        print(f"\n✅ SUCCESS: Kalman detected reversal {ema_flip - kf_flip} candles earlier!")
    elif kf_flip == ema_flip:
         print(f"\n⚠️ RESULT: Kalman matched EMA. Tuning needed for more aggression.")
    else:
        print(f"\n❌ FAIL: Kalman was slower.")

if __name__ == "__main__":
    run_benchmark()
