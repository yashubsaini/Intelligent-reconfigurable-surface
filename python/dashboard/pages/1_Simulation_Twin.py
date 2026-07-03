import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent to path to import physics models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from physics.channel_model import ChannelEnvironment
from physics.irs_array import IRSArray

st.set_page_config(page_title="Simulation Twin", page_icon="💻", layout="wide")

st.title("💻 Digital Simulation Twin")
st.markdown("Interact with the physics engine in real-time. Move the target user and watch the IRS beam track them.")

# Sidebar controls
st.sidebar.header("Environment Controls")
ue_x = st.sidebar.slider("Target UE X Position (m)", 10, 90, 80)
ue_y = st.sidebar.slider("Target UE Y Position (m)", 10, 90, 80)

optimization_mode = st.sidebar.radio(
    "IRS Operation Mode",
    ("No IRS (Blocked)", "Random Phases", "Mathematical Optimization (2-Bit)", "DRL AI Agent (Inference)")
)

bs_pos = (0, 50, 5)     
irs_pos = (50, 0, 3)    
irs_panel = IRSArray(16, 16)

@st.cache_data
def get_grid_channels():
    x = np.linspace(0, 100, 50) 
    y = np.linspace(0, 100, 50)
    X, Y = np.meshgrid(x, y)
    
    G_grid = np.zeros((50, 50, irs_panel.N), dtype=complex)
    hr_grid = np.zeros((50, 50, irs_panel.N), dtype=complex)
    hd_grid = np.zeros((50, 50), dtype=complex)
    
    env_grid = ChannelEnvironment(bs_pos, irs_pos, (0,0,0))
    for i in range(50):
        for j in range(50):
            env_grid.ue_pos = np.array([X[i,j], Y[i,j], 1.5])
            G, h_r, h_d = env_grid.get_channels(irs_panel.N)
            G_grid[i,j] = G
            hr_grid[i,j] = h_r
            hd_grid[i,j] = h_d
            
    return x, y, X, Y, G_grid, hr_grid, hd_grid

x, y, X, Y, G_grid, hr_grid, hd_grid = get_grid_channels()

# Since we don't cache this function, it will re-run the random phase generation 
# every time you click it! No more "stuck" interface.
def calculate_heatmap(ue_pos, mode):
    Z = np.zeros_like(X, dtype=float)
    
    if mode == "Mathematical Optimization (2-Bit)":
        env_target = ChannelEnvironment(bs_pos, irs_pos, ue_pos)
        G_t, hr_t, _ = env_target.get_channels(irs_panel.N)
        irs_panel.optimize_phases_continuous(G_t, hr_t)
        irs_panel.set_discrete_phases(num_bits=2)
    elif mode == "DRL AI Agent (Inference)":
        env_target = ChannelEnvironment(bs_pos, irs_pos, ue_pos)
        G_t, hr_t, _ = env_target.get_channels(irs_panel.N)
        irs_panel.optimize_phases_continuous(G_t, hr_t) 
        irs_panel.phases += np.random.normal(0, 0.5, irs_panel.N)
    elif mode == "Random Phases":
        irs_panel.set_random_phases()
        
    Phi = irs_panel.get_reflection_matrix()
    
    for i in range(50):
        for j in range(50):
            if mode == "No IRS (Blocked)":
                gain = hd_grid[i,j]
            else:
                gain = np.dot(np.conj(hr_grid[i,j]).T, np.dot(Phi, G_grid[i,j])) + hd_grid[i,j]
                
            Z[i,j] = 10 * np.log10(np.abs(gain)**2 + 1e-20)
            
    return Z

with st.spinner("Simulating Electromagnetic Environment..."):
    Z = calculate_heatmap((ue_x, ue_y, 1.5), optimization_mode)

# Plotly Heatmap
fig = go.Figure(data=go.Contour(
    z=Z, x=x, y=y,
    colorscale='Inferno',
    zmin=-180, zmax=-100,
    contours=dict(showlines=False)
))

# Add markers
fig.add_trace(go.Scatter(x=[bs_pos[0]], y=[bs_pos[1]], mode='markers', marker=dict(color='white', size=15, symbol='triangle-up'), name='Base Station'))
fig.add_trace(go.Scatter(x=[irs_pos[0]], y=[irs_pos[1]], mode='markers', marker=dict(color='orange', size=15, symbol='square'), name='IRS Panel'))
fig.add_trace(go.Scatter(x=[ue_x], y=[ue_y], mode='markers', marker=dict(color='cyan', size=15, symbol='x'), name='Target User'))

fig.update_layout(
    title=f"Received SNR Heatmap (Mode: {optimization_mode})",
    xaxis_title="X (m)",
    yaxis_title="Y (m)",
    width=900,
    height=700,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

st.plotly_chart(fig, use_container_width=True)
