<<<<<<< HEAD
# IRS-5G-LAB 📡

**Intelligent Reconfigurable Surface Simulator for 5G/6G Networks**

This repository contains an end-to-end simulation framework and research platform for testing and validating Intelligent Reconfigurable Surfaces (IRS/RIS). It bridges theoretical electromagnetic channel physics, Deep Reinforcement Learning (DRL) for dynamic beam steering, and real-world hardware integration.

## 🚀 Features

- **Mathematical Channel Modeling**: A rigorous implementation of a cascaded physical channel model (Base Station → IRS → User Equipment). Includes Rician fading for the dominant Line-of-Sight link, Rayleigh fading for scattered links, and Uniform Rectangular Array (URA) spatial steering vectors.
- **Deep Reinforcement Learning (DRL) Engine**: A custom Gymnasium environment with a PyTorch Deep Deterministic Policy Gradient (DDPG) AI agent. The AI dynamically calculates continuous phase shifts for a 256-element IRS array to track a moving user in real-time.
- **Interactive Digital Twin**: A stunning Streamlit dashboard that visualizes the signal-to-noise ratio (SNR) heatmap of the room. It allows users to physically drag the target coordinates and watch the beam instantly adapt.
- **Hardware Telemetry Integration**: Dedicated modules for uploading `.csv` telemetry data collected from real Software Defined Radios (SDR) and physical IRS hardware, capable of calculating the "Reality Gap" between simulation and physical measurements.

## 📁 Repository Structure

```text
IRS-5G-LAB/
├── python/
│   ├── dashboard/                 # Streamlit web application
│   │   ├── app.py                 # Main landing page
│   │   └── pages/                 # Multi-page routing
│   │       ├── 1_Simulation_Twin.py
│   │       └── 2_Lab_Experiments.py
│   ├── drl/                       # PyTorch Deep Reinforcement Learning models
│   │   ├── ddpg_agent.py          # Actor-Critic network architecture
│   │   ├── irs_env.py             # Custom Gymnasium physics wrapper
│   │   ├── train_agent.py         # Training loop script
│   │   └── ddpg_actor.pth         # Pre-trained neural network weights
│   └── physics/                   # Core electromagnetic physics engine
│       ├── channel_model.py       # Fading, path loss, and spatial geometry
│       └── irs_array.py           # IRS phase optimization and matrix reflection
├── matlab/                        # Offline hardware validation scripts
│   └── baseline_channel.m         # 3D Array Factor radiation pattern
└── requirements.txt               # Python package dependencies
```

## 💻 How to Run the Dashboard

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Navigate to the dashboard directory:
   ```bash
   cd python/dashboard
   ```
3. Boot the Digital Twin:
   ```bash
   streamlit run app.py
   ```

## 🔬 Research Goals

This software stack was built to transition a purely mathematical concept into physical reality. By simulating the exact array factor and path loss physics of a 28 GHz mmWave setup, researchers can establish a theoretical ceiling for signal enhancement. When the physical IRS array is tested in the lab, the discrepancies between the simulated heatmap and the collected SDR data provide critical insights into hardware quantization loss, imperfect scattering, and RF leakage.
=======
# Intelligent-reconfigurable-surface
>>>>>>> 353bb8a8af08f6fe25e25b3b4773f40de2ffa384
