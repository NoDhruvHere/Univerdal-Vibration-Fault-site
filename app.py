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
    try:
        with open('universal_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file 'universal_model.pkl' not found!")
        st.stop()

model = load_model()

# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_features(signal):
    """Extract features robustly, handling edge cases"""
    features = []
    
    # Safety check
    if len(signal) == 0:
        return np.zeros(12)

    # Time domain features
    features.extend([
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
        np.sqrt(np.mean(signal**2)), 
        stats.skew(signal) if len(signal) > 3 else 0, 
        stats.kurtosis(signal) if len(signal) > 3 else 0,
        np.mean(np.abs(signal)), 
        np.sum(signal**2),
    ])
    
    # Frequency domain features
    try:
        fft_vals = np.abs(np.fft.fft(signal))
        if len(fft_vals) > 2:
            features.extend([
                np.sum(fft_vals), 
                np.std(fft_vals),
                np.max(fft_vals)
            ])
        else:
            features.extend([0, 0, 0])
    except:
        features.extend([0, 0, 0])
        
    return np.array(features)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# KEY FIX: Allow user to skip rows if header is different
skip_rows = st.sidebar.number_input(
    "Skip Top Rows (Headers)", 
    min_value=0, 
    max_value=10, 
    value=2, 
    help="If your CSV has 2 text rows at top, keep as 2. If data starts immediately, set to 0."
)

window_size = st.sidebar.slider("Window Size (samples)", 100, 10000, 1000)

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("🌐 Universal Gear Health Monitor")
st.markdown("""
Upload a CSV vibration signal. The system analyzes statistical features (RMS, Kurtosis, etc.) 
to determine if the gear is Healthy or Faulty.
""")

uploaded_file = st.file_uploader("Upload Vibration Data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # -----------------------------------------------------
        # STEP 1: SMART DATA LOADING
        # -----------------------------------------------------
        # Read CSV skipping specified rows
        df = pd.read_csv(uploaded_file, header=skip_rows)
        
        # Assume vibration data is in the second column (index 1)
        if df.shape[1] < 2:
            st.error("CSV must have at least 2 columns.")
        else:
            # Convert to numeric, turning errors (text) into NaN
            raw_signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            
            # Check if we have valid numbers
            if raw_signal.isnull().all():
                st.error("❌ No numeric data found in Column 2!")
                st.warning("Try increasing 'Skip Top Rows' in the sidebar if you are seeing header text as data.")
            else:
                # Clean data: Remove NaNs
                signal = raw_signal.dropna().values

                # Slice window
                if len(signal) >= window_size:
                    signal_window = signal[:window_size]
                else:
                    signal_window = signal
                    st.warning(f"Signal length ({len(signal)}) is smaller than window size. Using full signal.")

                # -----------------------------------------------------
                # STEP 2: EXTRACT & PREDICT
                # -----------------------------------------------------
                features = extract_features(signal_window)
                
                # Reshape for sklearn (1, n_features)
                features_reshaped = features.reshape(1, -1)
                
                prediction = model.predict(features_reshaped)[0]
                probability = model.predict_proba(features_reshaped)[0]

                # -----------------------------------------------------
                # STEP 3: DEBUG INFO (NEW!)
                # -----------------------------------------------------
                with st.expander("🔍 Debug: What is the Model Seeing?"):
                    st.write("If the values below look like 'NaN' or huge numbers, change 'Skip Top Rows' in the sidebar.")
                    feature_names = ['Mean', 'Std', 'Max', 'Min', 'RMS', 'Skew', 'Kurtosis', 'MAD', 'Energy', 'FFT_Sum', 'FFT_Std', 'FFT_Max']
                    debug_df = pd.DataFrame([features], columns=feature_names)
                    st.dataframe(debug_df)

                # -----------------------------------------------------
                # STEP 4: DISPLAY RESULTS
                # -----------------------------------------------------
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("### Diagnosis Result")
                    
                    if prediction == 0:
                        st.success("## HEALTHY ✅")
                        st.markdown("Vibration patterns match healthy training data.")
                    else:
                        st.error("## FAULTY ⚠️")
                        st.markdown("Abnormal vibration patterns detected (High kurtosis/energy or weird frequencies).")
                    
                    # Confidence
                    confidence = np.max(probability) * 100
                    st.metric("Model Confidence", f"{confidence:.1f}%")

                    # Text explanation of probability
                    fault_prob = probability[1] * 100
                    if fault_prob > 80:
                        st.caption("High confidence fault detected.")
                    elif fault_prob > 50:
                        st.caption("Moderate suspicion. Check signal manually.")
                    else:
                        st.caption("Low confidence, but leaning towards fault.")

                with col2:
                    st.markdown("### Probability Gauge")
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = fault_prob, # Probability of being FAULTY
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fault Probability (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                            'bar': {'color': "darkgray"},
                            'steps': [
                                {'range': [0, 40], 'color': "#85e3d6"}, 
                                {'range': [40, 60], 'color': "#fadc9e"}, 
                                {'range': [60, 100], 'color': "#ff6f6c"} 
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50 
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)

                # -----------------------------------------------------
                # STEP 5: PLOT SIGNAL
                # -----------------------------------------------------
                st.subheader("Signal Visualization")
                st.line_chart(signal_window)
                st.caption(f"Displaying first {len(signal_window)} samples.")

    except Exception as e:
        st.error(f"Critical Error: {e}")
        st.info("Try changing the 'Skip Top Rows' setting in the sidebar.")

else:
    st.info("👆 Upload a CSV file to begin.")