import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Lab Experiments", page_icon="🧪", layout="wide")

st.title("🧪 Physical Lab Experiments")
st.markdown("Upload telemetry data from the Software Defined Radios (SDR) and hardware IRS board to validate the simulation against real-world scattering.")

st.info("Awaiting physical experiments. Please upload the raw CSV data below when ready.", icon="ℹ️")

uploaded_file = st.file_uploader("Upload Lab Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        # Expecting columns like 'X', 'Y', 'Measured_SNR_dB', 'Simulated_SNR_dB'
        if 'Measured_SNR_dB' in df.columns and 'Simulated_SNR_dB' in df.columns:
            st.subheader("Reality Gap Analysis: Simulation vs Measurement")
            
            # Scatter plot comparing the two
            fig = px.scatter(df, x='Simulated_SNR_dB', y='Measured_SNR_dB', 
                             trendline="ols", 
                             title="Correlation: Sim vs Reality",
                             color_discrete_sequence=['#00D2FF'])
            
            # Add ideal y=x line
            min_val = min(df['Simulated_SNR_dB'].min(), df['Measured_SNR_dB'].min())
            max_val = max(df['Simulated_SNR_dB'].max(), df['Measured_SNR_dB'].max())
            fig.add_shape(
                type="line", line=dict(dash='dash', color='white'),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate mean absolute error
            mae = (df['Simulated_SNR_dB'] - df['Measured_SNR_dB']).abs().mean()
            st.metric("Mean Absolute Error (Reality Gap)", f"{mae:.2f} dB")
            
        else:
            st.warning("Please ensure your CSV contains 'Measured_SNR_dB' and 'Simulated_SNR_dB' columns for the correlation analysis.")
            
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        
else:
    st.markdown("""
    ### Expected Data Format
    The file should be a standard `.csv` with the following columns:
    - `Timestamp`
    - `UE_X`, `UE_Y`
    - `IRS_Config_Mode` (e.g., Random, Opt_2Bit)
    - `Measured_SNR_dB`
    - `Simulated_SNR_dB`
    """)
