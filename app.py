import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import stats
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Universal Gear Health Monitor", layout="wide")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open('universal_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ---------------------------------------------------------
# FEATURE EXTRACTION (Same as before)
# ---------------------------------------------------------
def extract_features(signal):
    features = []
    features.extend([
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
        np.sqrt(np.mean(signal**2)), stats.skew(signal), stats.kurtosis(signal),
        np.mean(np.abs(signal)), np.sum(signal**2),
    ])
    fft_vals = np.abs(np.fft.fft(signal))
    if len(fft_vals) > 2:
        features.extend([
            np.sum(fft_vals), 
            np.std(fft_vals),
            np.max(fft_vals)
        ])
    else:
        features.extend([0, 0, 0])
    return np.array(features)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🌐 Universal Gear Health Monitor")
st.markdown("""
This tool analyzes vibration signals to determine if **ANY** type of gear (Spur, Helical, Epicyclic) 
is **Healthy** or **Faulty**.
""")

st.sidebar.header("Analysis Settings")
window_size = st.sidebar.slider("Window Size (samples)", 100, 5000, 1000)

uploaded_file = st.file_uploader("Upload Vibration Data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Read Data
        df = pd.read_csv(uploaded_file, header=2)
        
        if df.shape[1] < 2:
            st.error("CSV must have at least 2 columns.")
        else:
            signal = df.iloc[:, 1].values
            
            # Slice window
            if len(signal) >= window_size:
                signal_window = signal[:window_size]
            else:
                signal_window = signal

            # Extract Features
            features = extract_features(signal_window)
            
            # Predict
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]

            # --- RESULTS DISPLAY ---
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Diagnosis Result")
                
                if prediction == 0:
                    st.success("## HEALTHY ✅")
                    st.markdown("No significant faults detected.")
                else:
                    st.error("## FAULTY ⚠️")
                    st.markdown("Abnormal vibration patterns detected.")
                
                # Confidence
                confidence = np.max(probability) * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")

            with col2:
                st.markdown("### Probability Distribution")
                
                # Create a Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability[1] * 100, # Probability of being FAULTY
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fault Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': "darkgray"},
                        'steps': [
                            {'range': [0, 40], 'color': "#85e3d6"}, # Healthy Green
                            {'range': [40, 60], 'color': "#fadc9e"}, # Warning Yellow
                            {'range': [60, 100], 'color': "#ff6f6c"} # Faulty Red
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50 # Decision boundary
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)

            # --- SIGNAL PLOT ---
            st.subheader("Input Signal Visualization")
            st.line_chart(signal_window)
            st.caption(f"Showing first {len(signal_window)} samples of the uploaded signal.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload a CSV file to begin analysis.")