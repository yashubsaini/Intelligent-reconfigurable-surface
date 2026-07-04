import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from channel_model import ChannelEnvironment
from irs_array import IRSArray

def calculate_power_for_grid(env, irs_panel, Phi_matrix, ue_pos):
    """Calculates power at ue_pos using a FIXED IRS reflection matrix."""
    env.ue_pos = np.array(ue_pos)
    G, h_r, h_d = env.get_channels(irs_panel.N)
    
    gain = np.dot(np.conj(h_r).T, np.dot(Phi_matrix, G)) + h_d
    power = 10 * np.log10(np.abs(gain)**2 + 1e-20)
    
    power_no_irs = 10 * np.log10(np.abs(h_d)**2 + 1e-20)
    return power_no_irs, power

def main():
    print("Generating Optimized SNR Heatmaps... This will look much better!")
    
    bs_pos = (0, 50, 5)     
    irs_pos = (50, 0, 3)    
    target_ue_pos = (80, 80, 1.5) # The location we want to focus the beam on
    
    # Grid parameters
    x = np.linspace(0, 100, 80) # higher resolution
    y = np.linspace(0, 100, 80)
    X, Y = np.meshgrid(x, y)
    
    Z_no_irs = np.zeros_like(X)
    Z_rand = np.zeros_like(X)
    Z_opt = np.zeros_like(X)
    
    # Use an 8x8 array (64 elements) for a sharper "spotlight" beam
    irs_panel = IRSArray(8, 8)
    
    # 1. Prepare Random Phase Matrix
    irs_panel.set_random_phases()
    Phi_rand = irs_panel.get_reflection_matrix()
    
    # 2. Prepare Optimized Phase Matrix (Steered towards target_ue_pos)
    env_target = ChannelEnvironment(bs_pos, irs_pos, target_ue_pos)
    G_target, h_r_target, _ = env_target.get_channels(irs_panel.N)
    irs_panel.optimize_phases_continuous(G_target, h_r_target)
    irs_panel.set_discrete_phases(num_bits=2)
    Phi_opt = irs_panel.get_reflection_matrix()
    
    env_grid = ChannelEnvironment(bs_pos, irs_pos, (0,0,0))
    
    for i in range(len(x)):
        for j in range(len(y)):
            ue_p = (X[i,j], Y[i,j], 1.5)
            
            p_no_irs, p_rand = calculate_power_for_grid(env_grid, irs_panel, Phi_rand, ue_p)
            _, p_opt = calculate_power_for_grid(env_grid, irs_panel, Phi_opt, ue_p)
            
            Z_no_irs[i,j] = p_no_irs
            Z_rand[i,j] = p_rand
            Z_opt[i,j] = p_opt
            
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    vmin = max(np.min(Z_no_irs), -180)
    vmax = min(np.max(Z_opt), -90)
    
    c1 = axes[0].contourf(X, Y, Z_no_irs, levels=100, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title('No IRS (Dead Zone)')
    
    c2 = axes[1].contourf(X, Y, Z_rand, levels=100, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title('IRS with Random Phases (Scattered)')
    
    c3 = axes[2].contourf(X, Y, Z_opt, levels=100, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[2].set_title('IRS Steered towards Target (x=80, y=80)')
    
    for ax in axes:
        ax.plot(bs_pos[0], bs_pos[1], 'w^', markersize=12, label='Base Station')
        ax.plot(irs_pos[0], irs_pos[1], 'ws', markersize=12, label='IRS Panel')
        ax.plot(target_ue_pos[0], target_ue_pos[1], 'cx', markersize=12, markeredgewidth=3, label='Target User')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left')
        
    fig.colorbar(c3, ax=axes.ravel().tolist(), label='Received Power (dB)')
    
    plt.savefig('snr_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved beautiful heatmap to 'snr_heatmap.png'")
    

if __name__ == '__main__':
    main()
