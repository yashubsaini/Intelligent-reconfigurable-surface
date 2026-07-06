import numpy as np
import math

class SimRISChannel:
    def __init__(self, environment=1, scenario=1, frequency=28, array_type=2, 
                 N=64, Nt=1, Nr=1):
        self.environment = environment
        self.scenario = scenario
        self.frequency = frequency
        self.array_type = array_type
        self.N = N
        self.Nt = Nt
        self.Nr = Nr
        
        self.lambda_c = (3 * 10**8) / (self.frequency * 10**9)
        self.k = 2 * np.pi / self.lambda_c
        self.dis = self.lambda_c / 2
        
        # Element Radiation Pattern
        self.q = 0.285
        self.Gain = np.pi
        
        # Path Loss Parameters (InH - Office)
        if self.environment == 1:
            # NLOS
            self.n_NLOS = 3.19
            self.sigma_NLOS = 8.29
            self.b_NLOS = 0.06
            self.f0 = 24.2
            
            # LOS
            self.n_LOS = 1.73
            self.sigma_LOS = 3.02
            self.b_LOS = 0
            
            self.dim = [75, 50, 3.5]
            
        else: # UMi - Street Canyon
            self.n_NLOS = 3.19
            self.sigma_NLOS = 8.2
            self.b_NLOS = 0
            self.f0 = 24.2
            
            self.n_LOS = 1.98
            self.sigma_LOS = 3.1
            self.b_LOS = 0
            
            self.dim = [150, 150, 50]

        if self.frequency == 28:
            self.lambda_p = 1.8
        elif self.frequency == 73:
            self.lambda_p = 1.9
            
    def generate_channel(self, Tx_xyz, Rx_xyz, RIS_xyz):
        x_Tx, y_Tx, z_Tx = Tx_xyz
        x_Rx, y_Rx, z_Rx = Rx_xyz
        x_RIS, y_RIS, z_RIS = RIS_xyz
        
        d_T_RIS = np.linalg.norm(np.array(Tx_xyz) - np.array(RIS_xyz))
        d_RIS_R = np.linalg.norm(np.array(RIS_xyz) - np.array(Rx_xyz))
        d_T_R = np.linalg.norm(np.array(Tx_xyz) - np.array(Rx_xyz))
        
        # 1. Tx-RIS Link (H)
        # LOS Probability
        if self.environment == 1:
            if z_RIS < z_Tx:
                if d_T_RIS <= 1.2:
                    p_LOS = 1.0
                elif d_T_RIS < 6.5:
                    p_LOS = np.exp(-(d_T_RIS - 1.2)/4.7)
                else:
                    p_LOS = 0.32 * np.exp(-(d_T_RIS - 6.5)/32.6)
            else:
                p_LOS = 1.0 # High mounted RIS
        else:
            p_LOS = min(20/d_T_RIS, 1) * (1 - np.exp(-d_T_RIS/39)) + np.exp(-d_T_RIS/39)
            
        I_LOS = np.random.choice([1, 0], p=[p_LOS, 1-p_LOS])
        
        h_LOS = 0
        if I_LOS == 1:
            if self.scenario == 1:
                I_phi = np.sign(x_RIS - x_Tx)
                phi_T_RIS_LOS = I_phi * np.degrees(np.arctan(abs(x_RIS - x_Tx) / (abs(y_RIS - y_Tx) + 1e-9)))
                I_theta = np.sign(z_Tx - z_RIS)
                theta_T_RIS_LOS = I_theta * np.degrees(np.arcsin(abs(z_RIS - z_Tx) / d_T_RIS))
                
                I_phi_Tx = np.sign(y_Tx - y_RIS)
                phi_Tx_LOS = I_phi_Tx * np.degrees(np.arctan(abs(y_Tx - y_RIS) / (abs(x_Tx - x_RIS) + 1e-9)))
                I_theta_Tx = np.sign(z_Tx - z_RIS)
                theta_Tx_LOS = I_theta_Tx * np.degrees(np.arcsin(abs(z_RIS - z_Tx) / d_T_RIS))
            else:
                # Simplified Scenario 2 (opposite wall) handling
                phi_T_RIS_LOS = 0
                theta_T_RIS_LOS = 0
                phi_Tx_LOS = 0
                theta_Tx_LOS = 0
                
            array_RIS_LOS = np.zeros(self.N, dtype=complex)
            c = 0
            sqrt_N = int(np.sqrt(self.N))
            for x in range(sqrt_N):
                for y in range(sqrt_N):
                    phase = self.k * self.dis * (x * np.sin(np.radians(theta_T_RIS_LOS)) + 
                                                 y * np.sin(np.radians(phi_T_RIS_LOS)) * np.cos(np.radians(theta_T_RIS_LOS)))
                    array_RIS_LOS[c] = np.exp(1j * phase)
                    c += 1
                    
            array_Tx_LOS = np.zeros(self.Nt, dtype=complex)
            c = 0
            if self.array_type == 1: # ULA
                for x in range(self.Nt):
                    phase = self.k * self.dis * (x * np.sin(np.radians(phi_Tx_LOS)) * np.cos(np.radians(theta_Tx_LOS)))
                    array_Tx_LOS[c] = np.exp(1j * phase)
                    c += 1
            else: # UPA
                sqrt_Nt = int(np.sqrt(self.Nt))
                for x in range(sqrt_Nt):
                    for y in range(sqrt_Nt):
                        phase = self.k * self.dis * (x * np.sin(np.radians(phi_Tx_LOS)) * np.cos(np.radians(theta_Tx_LOS)) + 
                                                     y * np.sin(np.radians(theta_Tx_LOS)))
                        array_Tx_LOS[c] = np.exp(1j * phase)
                        c += 1
            
            L_dB_LOS = -20 * np.log10(4 * np.pi / self.lambda_c) - 10 * self.n_LOS * (1 + self.b_LOS * ((self.frequency - self.f0) / self.f0)) * np.log10(d_T_RIS) - np.random.randn() * self.sigma_LOS
            L_LOS = 10**(L_dB_LOS / 10)
            
            h_LOS = np.sqrt(L_LOS) * np.outer(array_RIS_LOS, array_Tx_LOS) * np.exp(1j * np.random.rand() * 2 * np.pi) * np.sqrt(self.Gain * (np.cos(np.radians(theta_T_RIS_LOS)))**(2 * self.q))
            
        # Simplified NLOS generation
        C = max(1, np.random.poisson(self.lambda_p))
        S = np.random.randint(1, 30, size=C)
        h_NLOS = np.zeros((self.N, self.Nt), dtype=complex)
        
        # Rather than calculating exact 3D geometry for 100 scatterers, we generate random 
        # Angles of Arrival and Departure that follow Laplacian distribution as in 3GPP.
        # This keeps the code fast and RL-friendly while retaining the mathematical structure of SimRIS.
        for c_idx in range(C):
            phi_av = np.random.rand() * 180 - 90
            theta_av = np.random.rand() * 90 - 45
            
            for s_idx in range(S[c_idx]):
                phi_T = np.random.laplace(loc=phi_av, scale=np.sqrt(25/2))
                theta_T = np.random.laplace(loc=theta_av, scale=np.sqrt(25/2))
                
                phi_R = np.random.laplace(loc=phi_av, scale=np.sqrt(25/2))
                theta_R = np.random.laplace(loc=theta_av, scale=np.sqrt(25/2))
                
                array_R = np.zeros(self.N, dtype=complex)
                count = 0
                for x in range(int(np.sqrt(self.N))):
                    for y in range(int(np.sqrt(self.N))):
                        array_R[count] = np.exp(1j * self.k * self.dis * (x * np.sin(np.radians(theta_R)) + y * np.sin(np.radians(phi_R)) * np.cos(np.radians(theta_R))))
                        count += 1
                
                array_T = np.zeros(self.Nt, dtype=complex)
                if self.Nt == 1:
                    array_T[0] = 1.0
                
                d_c = np.random.uniform(1, d_T_RIS) + np.random.uniform(1, d_T_RIS)
                X_sigma = np.random.randn() * self.sigma_NLOS
                Lcs_dB = -20 * np.log10(4 * np.pi / self.lambda_c) - 10 * self.n_NLOS * (1 + self.b_NLOS * ((self.frequency - self.f0) / self.f0)) * np.log10(d_c) - X_sigma
                Lcs = 10**(Lcs_dB / 10)
                beta = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                
                h_NLOS += beta * np.sqrt(self.Gain * (np.cos(np.radians(theta_R)))**(2 * self.q)) * np.sqrt(Lcs) * np.outer(array_R, array_T)
                
        h_NLOS = h_NLOS * np.sqrt(1 / np.sum(S))
        H = h_NLOS + h_LOS
        
        # 2. RIS-Rx Link (G)
        if self.environment == 1:
            I_theta = np.sign(z_Rx - z_RIS)
            theta_Rx_RIS = I_theta * np.degrees(np.arcsin(abs(z_Rx - z_RIS) / d_RIS_R))
            
            I_phi = np.sign(x_RIS - x_Rx)
            phi_Rx_RIS = I_phi * np.degrees(np.arctan(abs(x_Rx - x_RIS) / (abs(y_Rx - y_RIS) + 1e-9)))
            
            phi_Rx = np.random.laplace(loc=np.random.rand()*180-90, scale=np.sqrt(25/2))
            theta_Rx = np.random.laplace(loc=np.random.rand()*180-90, scale=np.sqrt(25/2))
            
            array_2 = np.zeros(self.N, dtype=complex)
            c = 0
            for x in range(int(np.sqrt(self.N))):
                for y in range(int(np.sqrt(self.N))):
                    array_2[c] = np.exp(1j * self.k * self.dis * (x * np.sin(np.radians(theta_Rx_RIS)) + y * np.sin(np.radians(phi_Rx_RIS)) * np.cos(np.radians(theta_Rx_RIS))))
                    c += 1
            
            array_Rx = np.zeros(self.Nr, dtype=complex)
            if self.Nr == 1:
                array_Rx[0] = 1.0
                
            L_dB_LOS_2 = -20 * np.log10(4 * np.pi / self.lambda_c) - 10 * self.n_LOS * (1 + self.b_LOS * ((self.frequency - self.f0) / self.f0)) * np.log10(d_RIS_R) - np.random.randn() * self.sigma_LOS
            L_LOS_2 = 10**(L_dB_LOS_2 / 10)
            
            g = np.sqrt(self.Gain * (np.cos(np.radians(theta_Rx_RIS)))**(2 * self.q)) * np.sqrt(L_LOS_2) * np.outer(array_2, array_Rx) * np.exp(1j * np.random.rand() * 2 * np.pi)
            
            # Simplified NLOS for G
            g_NLOS = np.zeros((self.N, self.Nr), dtype=complex)
            for c_idx in range(C):
                phi_av = np.random.rand() * 180 - 90
                theta_av = np.random.rand() * 90 - 45
                for s_idx in range(S[c_idx]):
                    phi_T = np.random.laplace(loc=phi_av, scale=np.sqrt(25/2))
                    theta_T = np.random.laplace(loc=theta_av, scale=np.sqrt(25/2))
                    
                    array_T = np.zeros(self.N, dtype=complex)
                    count = 0
                    for x in range(int(np.sqrt(self.N))):
                        for y in range(int(np.sqrt(self.N))):
                            array_T[count] = np.exp(1j * self.k * self.dis * (x * np.sin(np.radians(theta_T)) + y * np.sin(np.radians(phi_T)) * np.cos(np.radians(theta_T))))
                            count += 1
                            
                    array_R = np.zeros(self.Nr, dtype=complex)
                    if self.Nr == 1:
                        array_R[0] = 1.0
                        
                    d_c = np.random.uniform(1, d_RIS_R) + np.random.uniform(1, d_RIS_R)
                    X_sigma = np.random.randn() * self.sigma_NLOS
                    Lcs_dB = -20 * np.log10(4 * np.pi / self.lambda_c) - 10 * self.n_NLOS * (1 + self.b_NLOS * ((self.frequency - self.f0) / self.f0)) * np.log10(d_c) - X_sigma
                    Lcs = 10**(Lcs_dB / 10)
                    beta = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                    
                    g_NLOS += beta * np.sqrt(self.Gain * (np.cos(np.radians(theta_T)))**(2 * self.q)) * np.sqrt(Lcs) * np.outer(array_T, array_R)
                    
            g_NLOS = g_NLOS * np.sqrt(1 / np.sum(S))
            G = g_NLOS + g
            
        else:
            G = np.zeros((self.N, self.Nr), dtype=complex)
            
        # 3. Tx-Rx Link (D) - simplified to standard fading with path loss
        L_SISO_dB = -20 * np.log10(4 * np.pi / self.lambda_c) - 10 * self.n_NLOS * np.log10(d_T_R) - np.random.randn() * self.sigma_NLOS
        L_SISO = 10**(L_SISO_dB / 10)
        D = np.zeros((self.Nr, self.Nt), dtype=complex)
        D[0, 0] = np.sqrt(L_SISO) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

        return H, G, D

if __name__ == "__main__":
    channel = SimRISChannel()
    H, G, D = channel.generate_channel([0, 25, 2], [38, 48, 1], [40, 50, 2])
    print("H shape:", H.shape)
    print("G shape:", G.shape)
    print("D shape:", D.shape)
    print("Mean H norm:", np.linalg.norm(H))
