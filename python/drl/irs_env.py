import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add parent directory to path to import physics models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.simris_channel import SimRISChannel
from physics.irs_array import IRSArray

class IRSEnv(gym.Env):
    """
    Custom Environment for optimizing IRS phases using DRL with SimRIS channel physics.
    """
    def __init__(self, room_size=75.0, num_elements_x=8, num_elements_y=8):
        super(IRSEnv, self).__init__()
        
        self.room_size = room_size
        self.bs_pos = (0, room_size/2, 5) # Tx
        self.irs_pos = (room_size/2, 0, 3) # RIS
        
        # Fixed user position as requested: "don't move the user"
        self.ue_pos = np.array([room_size * 0.7, room_size * 0.6, 1.5]) 
        
        self.irs_panel = IRSArray(num_elements_x, num_elements_y)
        self.N = self.irs_panel.N
        
        # State: [ue_x, ue_y, d_bs, d_irs, prev_snr, mean_phase, phase_std, channel_gain, channel_phase, D_real, D_imag] 
        # + [N cascaded channel real parts] + [N cascaded channel imaginary parts] = 11 + 2*N dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11 + 2*self.N,), dtype=np.float32)
        
        # Action: Continuous delta-phases for each element in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)
        
        self.current_phases = np.zeros(self.N, dtype=np.float32)
        self.prev_snr_db = 0.0
        self.channel_gain = 0.0
        self.channel_phase = 0.0
        self.cascaded_channel = np.ones(self.N, dtype=np.complex128)
        self.direct_channel = 0.0j
        
        self.env_model = SimRISChannel(environment=1, N=self.N) # Indoor Hotspot
        
        self.current_step = 0
        self.max_steps = 100 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.current_phases = np.zeros(self.N, dtype=np.float32)
        self.prev_snr_db = 0.0
        self.channel_gain = 0.0
        self.channel_phase = 0.0
        
        # SimRIS channels
        H, G, D = self.env_model.generate_channel(self.bs_pos, self.ue_pos, self.irs_pos)
        
        # H is (N, 1) Tx->RIS, G is (N, 1) RIS->Rx. Cascaded channel for each element is H_n * G_n
        self.cascaded_channel = (H.flatten() * G.flatten())
        self.direct_channel = D[0, 0]
        
        return self._get_obs(), {}

    def _get_obs(self):
        d_bs = np.linalg.norm(self.ue_pos[:2] - np.array(self.bs_pos[:2])) / self.room_size
        d_irs = np.linalg.norm(self.ue_pos[:2] - np.array(self.irs_pos[:2])) / self.room_size
        snr_norm = np.clip((self.prev_snr_db + 10.0) / 100.0, 0.0, 1.0)
        mean_phase = np.mean(self.current_phases) / np.pi 
        phase_std = np.std(self.current_phases) / np.pi 
        
        channel_gain_db = 10 * np.log10(self.channel_gain + 1e-20)
        channel_gain_norm = np.clip((channel_gain_db + 120) / 60.0, 0.0, 1.0)
        
        # Scale cascaded channel up so it's not tiny values
        # SimRIS gives very small path loss (e.g. 1e-6 to 1e-8)
        scale_factor = 1e6
        cascaded_real = np.real(self.cascaded_channel * scale_factor).astype(np.float32)
        cascaded_imag = np.imag(self.cascaded_channel * scale_factor).astype(np.float32)
        
        scalar_obs = np.array([
            self.ue_pos[0] / self.room_size, 
            self.ue_pos[1] / self.room_size,
            d_bs,
            d_irs,
            snr_norm,
            mean_phase,
            phase_std,
            channel_gain_norm,
            self.channel_phase,
            np.real(self.direct_channel * scale_factor),
            np.imag(self.direct_channel * scale_factor)
        ], dtype=np.float32)
        
        return np.concatenate([scalar_obs, cascaded_real, cascaded_imag])

    def step(self, action):
        self.current_step += 1
        
        # 1. Map action directly to absolute phase [-pi, pi]
        self.current_phases = action * np.pi
        self.irs_panel.phases = self.current_phases
        Phi = self.irs_panel.get_reflection_matrix() # Diagonal matrix of exp(j * phase)
        
        # 2. Get Channels (Since UE doesn't move, we cache this from reset)
        # H, G, D = self.env_model.generate_channel(self.bs_pos, self.ue_pos, self.irs_pos)
        # self.cascaded_channel = (H.flatten() * G.flatten())
        # self.direct_channel = D[0, 0]
        
        self.channel_gain = np.mean(np.abs(self.cascaded_channel))
        self.channel_phase = np.angle(np.sum(self.cascaded_channel)) / np.pi
        
        # 3. Calculate True SNR (Assume 100W Base Station Transmit Power)
        # y = (G^T \Phi H + D) x + n
        # Since we use cascaded_channel directly, the gain is sum(cascaded * Phi)
        phi_diag = np.diag(Phi)
        gain = np.sum(self.cascaded_channel * phi_diag) + self.direct_channel
        
        transmit_power_linear = 100.0  # 100 Watts
        power_linear = transmit_power_linear * (np.abs(gain)**2)
        
        # Thermal noise at 28 GHz with 100 MHz bandwidth (-94 dBm = -124 dBW)
        noise_power_linear = 10**(-12.4)
        snr_linear = power_linear / noise_power_linear
        snr_db = 10 * np.log10(snr_linear + 1e-20)
        
        # Reward Formulation
        reward = snr_db / 20.0
        
        # Penalty for aggressive phase flipping
        action_penalty = 0.01 * np.mean(action**2)
        reward -= action_penalty
        
        self.prev_snr_db = snr_db
        
        # 4. UE does not move (static position)
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), float(reward), done, truncated, {"snr_db": float(snr_db)}
