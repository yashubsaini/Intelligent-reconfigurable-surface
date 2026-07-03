import numpy as np
from irs_array import IRSArray
from channel_model import ChannelEnvironment

def main():
    print("--- IRS-Sim Channel Verification ---")
    
    # 1. Setup Environment
    bs = (0, 0, 5)   # Base station at height 5m
    irs = (50, 50, 3) # IRS at (50,50) height 3m
    ue = (60, 50, 1.5) # UE moving near IRS at 1.5m
    
    env = ChannelEnvironment(bs, irs, ue, frequency=28e9)
    
    # 2. Setup IRS (8x8 panel = 64 elements)
    irs_panel = IRSArray(8, 8, frequency=28e9)
    N = irs_panel.N
    print(f"IRS Elements: {N}")
    
    # 3. Get Channels
    G, h_r, h_d = env.get_channels(N)
    print(f"G matrix shape: {G.shape}")
    print(f"h_r vector shape: {h_r.shape}")
    
    # 4. Calculate Received Power without optimization (Random phases)
    irs_panel.set_random_phases()
    Phi = irs_panel.get_reflection_matrix()
    
    # Cascaded channel: h_cascaded = h_r^H * Phi * G
    # Assuming G is N, h_r is N for single antenna BS/UE
    cascaded_gain = np.dot(np.conj(h_r).T, np.dot(Phi, G))
    total_gain = cascaded_gain + h_d
    rx_power_random = 10 * np.log10(np.abs(total_gain)**2) # arbitrary power metric
    
    print(f"Received Power (Random Phase): {rx_power_random:.2f} dB")
    
    # 5. Optimize Phases (Continuous)
    irs_panel.optimize_phases_continuous(G, h_r)
    Phi_opt = irs_panel.get_reflection_matrix()
    
    cascaded_gain_opt = np.dot(np.conj(h_r).T, np.dot(Phi_opt, G))
    total_gain_opt = cascaded_gain_opt + h_d
    rx_power_opt = 10 * np.log10(np.abs(total_gain_opt)**2)
    
    print(f"Received Power (Optimized Phase - Continuous): {rx_power_opt:.2f} dB")
    print(f"Gain from Continuous Optimization: {rx_power_opt - rx_power_random:.2f} dB")
    
    # 6. Quantize Phases (Discrete - e.g. 2-bit)
    irs_panel.set_discrete_phases(num_bits=2)
    Phi_discrete = irs_panel.get_reflection_matrix()
    
    cascaded_gain_disc = np.dot(np.conj(h_r).T, np.dot(Phi_discrete, G))
    total_gain_disc = cascaded_gain_disc + h_d
    rx_power_disc = 10 * np.log10(np.abs(total_gain_disc)**2)
    
    print(f"Received Power (Optimized Phase - 2-Bit Discrete): {rx_power_disc:.2f} dB")
    print(f"Gain from Discrete Optimization: {rx_power_disc - rx_power_random:.2f} dB")
    print(f"Quantization Loss: {rx_power_opt - rx_power_disc:.2f} dB")
    print("------------------------------------")
    
if __name__ == '__main__':
    main()
