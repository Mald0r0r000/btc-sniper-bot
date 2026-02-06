"""
Kalman Filter for Price Trend Estimation (1D)
Designed to replace lagging EMAs with adaptive state estimation.
Tracks: Position (Price) and Velocity (Momentum).
Pure Python Implementation (No Numpy dependency).
"""
import math

class KalmanFilter1D:
    def __init__(self, process_noise: float = 0.05, measurement_noise: float = 0.1):
        """
        Initialize 1D Kalman Filter (Constant Velocity Model).
        
        Args:
            process_noise (Q): Confidence in the model/physics. Default 0.05 (High = Nimble).
            measurement_noise (R): Confidence in the measurement. Default 0.1 (Low = Trust Price).
        """
        # State Vector [Price, Velocity]
        self.x = [0.0, 0.0]
        
        # State Covariance (Uncertainty) P
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        
        # State Transition Matrix F
        # [[1, dt], [0, 1]]
        self.dt = 1.0
        
        # Process Noise Covariance Q
        # Approximated for Constant Velocity
        q_pos = (self.dt**4)/4 * process_noise
        q_vel = self.dt**2 * process_noise
        self.Q = [
            [q_pos, (self.dt**3)/2 * process_noise],
            [(self.dt**3)/2 * process_noise, q_vel]
        ]
        
        # Measurement Noise R
        self.R = measurement_noise
        
        self.initialized = False

    def update(self, measurement: float) -> tuple:
        """
        Update the filter with a new price measurement.
        Returns: (Estimated_Price, Estimated_Velocity)
        """
        if not self.initialized:
            self.x = [float(measurement), 0.0]
            self.initialized = True
            return measurement, 0.0
        
        # 1. Predict
        # x_pred = F * x
        x_pred_0 = self.x[0] + self.x[1] * self.dt
        x_pred_1 = self.x[1]
        
        # P_pred = F * P * F.T + Q
        # Manual matrix multiplication for 2x2
        p00 = self.P[0][0] + self.P[1][0] * self.dt
        p01 = self.P[0][1] + self.P[1][1] * self.dt
        p10 = self.P[1][0]
        p11 = self.P[1][1]
        
        # P * F.T (Transposed)
        # F.T = [[1, 0], [dt, 1]]
        pp00 = p00 + p01 * self.dt
        pp01 = p01
        pp10 = p10 + p11 * self.dt
        pp11 = p11
        
        P_pred = [
            [pp00 + self.Q[0][0], pp01 + self.Q[0][1]],
            [pp10 + self.Q[1][0], pp11 + self.Q[1][1]]
        ]
        
        # 2. Update
        # Innovation y = z - H * x_pred
        y = measurement - x_pred_0
        
        # Innovation Covariance S = H * P_pred * H.T + R
        # H = [1, 0]
        S = P_pred[0][0] + self.R
        
        # Kalman Gain K = P_pred * H.T * inv(S)
        # K = [P_pred[0][0] / S, P_pred[1][0] / S]
        K = [P_pred[0][0] / S, P_pred[1][0] / S]
        
        # Update State x = x_pred + K * y
        self.x[0] = x_pred_0 + K[0] * y
        self.x[1] = x_pred_1 + K[1] * y
        
        # Update Covariance P = (I - K * H) * P_pred
        # I - K * H = [[1 - K0, 0], [-K1, 1]]
        i_kh_00 = 1 - K[0]
        i_kh_10 = -K[1]
        
        one = i_kh_00 * P_pred[0][0] # + 0 * P_pred[1][0]
        two = i_kh_00 * P_pred[0][1] # + 0 * P_pred[1][1]
        three = i_kh_10 * P_pred[0][0] + 1 * P_pred[1][0]
        four = i_kh_10 * P_pred[0][1] + 1 * P_pred[1][1]
        
        self.P = [[one, two], [three, four]]
        
        return self.x[0], self.x[1]
