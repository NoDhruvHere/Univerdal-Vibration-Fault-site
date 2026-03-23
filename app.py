import streamlit as st
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tftensorflow as tfimport
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import os
import glob
from datetime import datetime

@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path='results/cnn_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]
# Page configuration
st.set_page_config(
    page_title="Gearbox Fault Detection System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .fault-healthy { color: #27ae60; font-weight: bold; }
    .fault-stage1 { color: #e74c3c; font-weight: bold; }
    .fault-stage2 { color: #f39c12; font-weight: bold; }
    .fault-stage3 { color: #9b59b6; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Configuration
CONFIG = {
    'window_size': 1000,
    'stride': 400,
    'sampling_rate': 10000,
    'fft_points': 500,  # window_size // 2
    'model_path': 'results/cnn_model.h5',
    'scaler_path': 'results/scaler.pkl',
    'config_path': 'results/config.pkl'
}

CLASS_INFO = {
    0: {'name': 'Healthy', 'color': '#27ae60', 'stage': 'Normal', 'severity': 'None'},
    1: {'name': 'Planet surface crack (S1)', 'color': '#e74c3c', 'stage': 'Stage 1', 'severity': 'High'},
    2: {'name': 'Ring Broken tooth (S1)', 'color': '#e74c3c', 'stage': 'Stage 1', 'severity': 'Critical'},
    3: {'name': 'Sun Chipped tooth (S1)', 'color': '#e74c3c', 'stage': 'Stage 1', 'severity': 'Medium'},
    4: {'name': 'Planet gear defect 75% (S2)', 'color': '#f39c12', 'stage': 'Stage 2', 'severity': 'High'},
    5: {'name': 'Ring gear one tooth Missing (S2)', 'color': '#f39c12', 'stage': 'Stage 2', 'severity': 'Critical'},
    6: {'name': 'Sun Gear Defect Chipped Tooth 2nd (S2)', 'color': '#f39c12', 'stage': 'Stage 2', 'severity': 'Medium'},
    7: {'name': 'Planet 2 Broken tooth 180 (S3)', 'color': '#9b59b6', 'stage': 'Stage 3', 'severity': 'Critical'},
    8: {'name': 'RING 2 TOOTH 120 (S3)', 'color': '#9b59b6', 'stage': 'Stage 3', 'severity': 'High'},
    9: {'name': 'Sun Gear 2 tooth taper crack (S3)', 'color': '#9b59b6', 'stage': 'Stage 3', 'severity': 'Medium'}
}

@st.cache_resource
def load_prediction_resources():
    """Load model and preprocessing objects"""
    try:
        model = load_model(CONFIG['model_path'])
        scaler = joblib.load(CONFIG['scaler_path'])
        config = joblib.load(CONFIG['config_path'])
        return model, scaler, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def parse_uploaded_file(uploaded_file):
    """Parse CSV with multiple format support"""
    try:
        # Try different parsing strategies
        df = None
        
        # Strategy 1: Standard comma
        try:
            df = pd.read_csv(uploaded_file, header=None, sep=',', decimal='.')
        except:
            uploaded_file.seek(0)
            
        # Strategy 2: Semicolon
        if df is None or df.empty:
            try:
                df = pd.read_csv(uploaded_file, header=None, sep=';', decimal=',')
            except:
                uploaded_file.seek(0)
                
        # Strategy 3: Auto-detect
        if df is None or df.empty:
            df = pd.read_csv(uploaded_file, header=None, sep=None, engine='python')
        
        signal = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        signal = signal[~np.isnan(signal)]
        
        return signal.astype(np.float64)
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def preprocess_signal(signal, window_size=1000):
    """Preprocess signal: normalize and create windows"""
    # Normalize
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val > 1e-10:
        signal = (signal - mean_val) / std_val
    else:
        signal = signal - mean_val
    
    # Create single window (center crop or pad)
    if len(signal) >= window_size:
        # Take center portion
        start = (len(signal) - window_size) // 2
        window = signal[start:start + window_size]
    else:
        # Pad with zeros
        pad_left = (window_size - len(signal)) // 2
        pad_right = window_size - len(signal) - pad_left
        window = np.pad(signal, (pad_left, pad_right), mode='constant')
    
    return window

def extract_features(spectrum):
    """Extract statistical features for display"""
    features = {
        'Mean': np.mean(spectrum),
        'RMS': np.sqrt(np.mean(spectrum**2)),
        'Peak': np.max(spectrum),
        'Crest Factor': np.max(spectrum) / np.sqrt(np.mean(spectrum**2)) if np.mean(spectrum**2) > 0 else 0,
        'Kurtosis': stats.kurtosis(spectrum),
        'Skewness': stats.skew(spectrum)
    }
    return features

def predict_fault(model, signal):
    """Run prediction on preprocessed signal"""
    # Convert to frequency domain
    fft_vals = np.fft.fft(signal)
    spectrum = np.abs(fft_vals[:len(signal)//2])
    
    # Reshape for CNN (batch, timesteps, channels)
    spectrum_input = spectrum.reshape(1, len(spectrum), 1)
    
    # Predict
    predictions = model.predict(spectrum_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence, predictions[0], spectrum

def create_spectrum_plot(spectrum, sampling_rate=10000):
    """Create frequency spectrum plot"""
    freqs = np.fft.fftfreq(len(spectrum)*2, 1/sampling_rate)[:len(spectrum)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, 
        y=spectrum,
        mode='lines',
        name='Amplitude Spectrum',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.update_layout(
        title='Frequency Domain Spectrum (FFT)',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_plot(prediction_probs):
    """Create bar chart of prediction confidence"""
    classes = [CLASS_INFO[i]['name'] for i in range(10)]
    colors = [CLASS_INFO[i]['color'] for i in range(10)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=prediction_probs * 100,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in prediction_probs * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence by Class',
        xaxis_title='Fault Type',
        yaxis_title='Confidence (%)',
        template='plotly_white',
        xaxis_tickangle=-45,
        height=500,
        yaxis_range=[0, 100]
    )
    
    return fig

def get_recommendation(class_id):
    """Get maintenance recommendation based on fault type"""
    recommendations = {
        0: "✅ System operating normally. Continue regular maintenance schedule.",
        1: "⚠️ **Stage 1 Fault Detected**: Planet gear surface crack identified. Inspect planet gears for crack propagation. Schedule maintenance within 1-2 weeks.",
        2: "🚨 **Stage 1 Critical**: Ring gear has broken tooth. Immediate shutdown recommended. Replace ring gear to prevent secondary damage.",
        3: "⚠️ **Stage 1 Fault**: Sun gear chipped tooth detected. Monitor vibration levels closely. Schedule inspection within 3-5 days.",
        4: "⚠️ **Stage 2 Fault**: Significant planet gear damage (75%). Gear replacement required. Reduce load capacity until repair.",
        5: "🚨 **Stage 2 Critical**: Missing tooth on ring gear. High risk of catastrophic failure. Immediate maintenance required.",
        6: "⚠️ **Stage 2 Fault**: Sun gear chipped tooth (2nd occurrence). Check lubrication system and alignment. Schedule repair.",
        7: "🚨 **Stage 3 Critical**: Multiple broken teeth detected. Severe damage imminent. Stop operation immediately.",
        8: "⚠️ **Stage 3 Fault**: Ring gear damage (2 teeth, 120° apart). Advanced wear stage. Major overhaul recommended.",
        9: "⚠️ **Stage 3 Fault**: Sun gear crack propagation detected. Final stage before failure. Urgent replacement needed."
    }
    return recommendations.get(class_id, "Consult maintenance manual.")

# Sidebar
st.sidebar.title("⚙️ Gearbox Health Monitor")
st.sidebar.markdown("---")
st.sidebar.info("Upload vibration data to analyze gearbox health using 1D-CNN deep learning model.")

# Model status
model, scaler, config = load_prediction_resources()
if model is not None:
    st.sidebar.success("🟢 Model Loaded Successfully")
else:
    st.sidebar.error("🔴 Model Not Found")
    st.sidebar.info("Please ensure 'cnn_model.h5' exists in the results folder")

st.sidebar.markdown("---")
st.sidebar.markdown("### Supported Formats")
st.sidebar.markdown("- CSV files (.csv)")
st.sidebar.markdown("- Single column vibration data")
st.sidebar.markdown("- Time-domain signals")

# Main content
st.title("🔧 Epicyclic Gearbox Fault Diagnosis System")
st.markdown("**AI-Powered Predictive Maintenance using 1D-Convolutional Neural Networks**")

# File upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload Vibration Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing vibration data",
        type=['csv'],
        help="Upload time-domain vibration signals from accelerometer"
    )

with col2:
    st.subheader("📊 System Status")
    st.metric("Classes Supported", "10")
    st.metric("Model Type", "1D-CNN")
    st.metric("Input Window", f"{CONFIG['window_size']} samples")

if uploaded_file is not None and model is not None:
    # Process file
    with st.spinner('Processing vibration data...'):
        signal = parse_uploaded_file(uploaded_file)
        
        if signal is not None and len(signal) > 0:
            st.success(f"✅ File loaded: {len(signal)} samples")
            
            # Preprocess
            window = preprocess_signal(signal, CONFIG['window_size'])
            
            # Predict
            pred_class, confidence, probs, spectrum = predict_fault(model, window)
            
            # Layout
            left_col, right_col = st.columns([1, 1])
            
            with left_col:
                st.subheader("🎯 Prediction Results")
                
                # Main prediction box
                info = CLASS_INFO[pred_class]
                
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3 style='margin-top:0;'>Detected Condition</h3>
                    <h2 style='color:{info['color']}; margin:0;'>{info['name']}</h2>
                    <p><strong>Stage:</strong> {info['stage']} | <strong>Severity:</strong> {info['severity']}</p>
                    <h4 style='margin-bottom:0;'>Confidence: {confidence*100:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation
                st.subheader("🔧 Maintenance Recommendation")
                recommendation = get_recommendation(pred_class)
                st.markdown(recommendation)
                
                # Key metrics
                st.subheader("📈 Signal Characteristics")
                features = extract_features(spectrum)
                metrics_cols = st.columns(3)
                for idx, (key, value) in enumerate(features.items()):
                    with metrics_cols[idx % 3]:
                        st.metric(key, f"{value:.3f}")
            
            with right_col:
                st.subheader("📊 Spectrum Analysis")
                fig_spectrum = create_spectrum_plot(spectrum)
                st.plotly_chart(fig_spectrum, use_container_width=True)
            
            # Confidence chart
            st.subheader("📊 Prediction Probability Distribution")
            fig_conf = create_confidence_plot(probs)
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Detailed analysis table
            with st.expander("📋 Detailed Analysis Report"):
                report_data = []
                for i in range(10):
                    report_data.append({
                        'Class ID': i,
                        'Fault Type': CLASS_INFO[i]['name'],
                        'Stage': CLASS_INFO[i]['stage'],
                        'Severity': CLASS_INFO[i]['severity'],
                        'Probability': f"{probs[i]*100:.4f}%",
                        'Status': '🔴 Detected' if i == pred_class else '⚪ Not Detected'
                    })
                
                df_report = pd.DataFrame(report_data)
                st.dataframe(df_report, use_container_width=True)
                
                # Download report
                csv = df_report.to_csv(index=False)
                st.download_button(
                    label="📥 Download Report (CSV)",
                    data=csv,
                    file_name=f'gearbox_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
        else:
            st.error("❌ Could not process the uploaded file. Please check the format.")
else:
    # Show example/demo when no file uploaded
    st.info("👆 Upload a CSV file to begin analysis")
    
    # Demo section
    st.markdown("---")
    st.subheader("📚 About This System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🧠 1D-CNN Architecture**
        - 3 Convolutional Layers
        - Batch Normalization
        - Dropout Regularization
        - Real-time Inference
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Fault Classes**
        - 1 Healthy State
        - 3 Stages (Planet, Ring, Sun)
        - 9 Distinct Fault Types
        - Severity Classification
        """)
    
    with col3:
        st.markdown("""
        **📊 Analysis Features**
        - FFT Spectrum Analysis
        - Confidence Scoring
        - Statistical Features
        - Maintenance Recommendations
        """)

# Footer
st.markdown("---")
st.markdown(
    "<center>Developed for Predictive Maintenance | "
    "Uses TensorFlow 1D-CNN | "
    "Processes Frequency Domain Features</center>", 
    unsafe_allow_html=True
)
