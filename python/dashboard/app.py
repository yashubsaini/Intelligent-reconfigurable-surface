import streamlit as st

# Setup page config for a premium research aesthetic
st.set_page_config(
    page_title="IRS-Sim | 5G/6G Research",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium aesthetic
st.markdown("""
<style>
    /* Global theme adjustments */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    h1, h2, h3 {
        color: #00D2FF;
        font-family: 'Inter', sans-serif;
    }
    
    /* Highlight gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .subtitle {
        color: #A0AAB2;
        font-size: 1.2em;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Styling info boxes */
    div.stInfo {
        background-color: rgba(0, 210, 255, 0.1);
        border-left: 4px solid #00D2FF;
    }
</style>
""", unsafe_allow_html=True)

# Main Content
st.markdown('<div class="gradient-text">IRS-Sim</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Reconfigurable Surface Simulator for 6G</div>', unsafe_allow_html=True)

st.markdown("""
Welcome to the **IRS-Sim Research Platform**. This dashboard bridges the gap between theoretical channel physics, Artificial Intelligence (Deep Reinforcement Learning), and physical laboratory experiments.

### 🔬 Architecture Overview
This platform is split into two primary research modules, accessible via the sidebar:

1. **Simulation Twin**: A purely mathematical and AI-driven environment. Watch in real-time as a PyTorch Deep Q-Network dynamically alters the phase shifts of a 256-element IRS to steer a 28 GHz signal around physical blockages.
2. **Lab Experiments**: The hardware interface. Upload raw `.csv` telemetry data from Software Defined Radios (SDRs) and the physical IRS board to validate the simulation models against real-world scattering and hardware quantization loss.

---
**Developed for Advanced Agentic Coding Research.**
""")
