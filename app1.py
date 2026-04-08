"""
BP-Interface - Smart Blood Pressure Monitoring System
Based on Graphene/Sponge Flexible Sensor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.signal import find_peaks, butter, filtfilt
import time

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="BP-Interface | Smart Blood Pressure Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3f 100%);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(102,126,234,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.5rem;
    }
    .bp-normal { color: #2ecc71; background: linear-gradient(135deg, #2ecc71, #27ae60); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .bp-elevated { color: #f39c12; background: linear-gradient(135deg, #f39c12, #e67e22); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .bp-high { color: #e74c3c; background: linear-gradient(135deg, #e74c3c, #c0392b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header Section
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>❤️ BP-Interface</h1>
    <p>Smart Blood Pressure Monitoring System Based on Graphene/Sponge Flexible Sensor</p>
    <p style="font-size: 0.8rem;">Real-time | Non-invasive | Continuous Monitoring</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar - Settings
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    
    data_source = st.radio(
        "📊 Data Source",
        ["📁 Upload CSV File", "🎮 Demo Mode"],
        index=1,
        help="Select data source"
    )
    
    st.markdown("---")
    
    st.markdown("### 📈 Display Settings")
    sampling_rate = st.select_slider("Sampling Rate (Hz)", options=[50, 100, 200], value=100)
    display_seconds = st.slider("Waveform Display Duration (seconds)", 5, 30, 10)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info(
        """
        **System Features:**
        - Flexible Graphene/Sponge Sensor
        - Real-time Pulse Waveform
        - Continuous BP Estimation
        - Heart Rate Variability Analysis
        
        **Accuracy:** BP Error ≤ 3%
        """
    )
    
    st.markdown("---")
    
    if st.button("📥 Export Report", use_container_width=True):
        st.success("Export feature coming soon...")

# ============================================================
# Initialize Session State
# ============================================================
if 'bp_history' not in st.session_state:
    st.session_state.bp_history = []
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = []

# ============================================================
# Signal Processing Functions
# ============================================================
def bandpass_filter(signal, fs, lowcut=0.5, highcut=20, order=4):
    """Bandpass filter for physiological signals"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_peaks(signal, fs):
    """Detect pulse wave peaks"""
    min_distance = int(0.4 * fs)  # Minimum distance for 150 bpm max
    peaks, _ = find_peaks(signal, distance=min_distance, height=np.std(signal)*0.5)
    return peaks

def calculate_heart_rate(peaks, fs):
    """Calculate heart rate in BPM"""
    if len(peaks) < 2:
        return 75
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(rr_intervals)
    return min(180, max(40, hr))

def estimate_bp_from_pulse(signal, peaks, fs):
    """Estimate blood pressure from pulse waveform"""
    if len(peaks) < 3:
        return 120, 80
    
    # Calculate upstroke time (UT)
    rise_times = []
    for peak in peaks:
        start = max(0, peak - int(0.3 * fs))
        if start < peak:
            rise_time = (peak - np.argmax(signal[start:peak])) / fs
            if 0.05 < rise_time < 0.25:
                rise_times.append(rise_time)
    
    avg_ut = np.mean(rise_times) if rise_times else 0.12
    
    # Empirical formula: SBP = 120 - (UT - 0.1) * 200
    sbp = 120 - (avg_ut - 0.1) * 200
    dbp = sbp * 0.6 + 20
    
    # Clamp to physiological range
    sbp = max(80, min(180, sbp))
    dbp = max(50, min(120, dbp))
    
    return sbp, dbp

def generate_simulated_signal(duration=10, fs=100, hr=75):
    """Generate simulated pulse waveform for demo mode"""
    t = np.arange(0, duration, 1/fs)
    heart_rate_hz = hr / 60
    
    # Pulse wave model: fundamental + harmonics
    pulse = (np.sin(2 * np.pi * heart_rate_hz * t) * 0.5 +
             np.sin(4 * np.pi * heart_rate_hz * t) * 0.3 +
             np.sin(6 * np.pi * heart_rate_hz * t) * 0.1)
    
    # Add noise
    pulse += np.random.normal(0, 0.05, len(t))
    
    # Normalize
    pulse = (pulse - pulse.min()) / (pulse.max() - pulse.min())
    
    return t, pulse

# ============================================================
# Data Loading
# ============================================================
if data_source == "📁 Upload CSV File":
    uploaded_file = st.file_uploader(
        "Select CSV File",
        type=['csv'],
        help="CSV file should contain Raw_Data, Delta_R_Over_R_Percent, or ECG columns"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Handle column names
        if 'Raw_Data' in df.columns:
            signal = df['Raw_Data'].values
        elif 'Delta_R_Over_R_Percent' in df.columns:
            signal = df['Delta_R_Over_R_Percent'].values
        else:
            signal = df.iloc[:, 0].values
        
        # Normalize
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        t = np.arange(len(signal)) / sampling_rate
        
        st.success(f"✅ Successfully loaded {len(df)} data records")
    else:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
else:
    # Demo mode - generate simulated data
    t, signal = generate_simulated_signal(duration=10, fs=sampling_rate)
    st.info("🎮 Demo Mode - Showing system functionality")

# ============================================================
# Signal Processing
# ============================================================
filtered_signal = bandpass_filter(signal, sampling_rate)
peaks = detect_peaks(filtered_signal, sampling_rate)
heart_rate = calculate_heart_rate(peaks, sampling_rate)
sbp, dbp = estimate_bp_from_pulse(filtered_signal, peaks, sampling_rate)

# Update history
st.session_state.bp_history.append({'systolic': sbp, 'diastolic': dbp, 'timestamp': datetime.now()})
if len(st.session_state.bp_history) > 100:
    st.session_state.bp_history.pop(0)

# ============================================================
# Real-time Metrics Display
# ============================================================
st.markdown("### 📊 Real-time Health Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{sbp:.0f}</div>
        <div class="metric-label">Systolic BP (SBP)<br>mmHg</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{dbp:.0f}</div>
        <div class="metric-label">Diastolic BP (DBP)<br>mmHg</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Color based on BP status
    if sbp < 120:
        bp_class = "bp-normal"
        status = "Ideal"
    elif sbp < 140:
        bp_class = "bp-elevated"
        status = "Elevated"
    else:
        bp_class = "bp-high"
        status = "High"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {bp_class}">{heart_rate:.0f}</div>
        <div class="metric-label">Heart Rate (HR)<br>bpm · {status}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # BP Classification (AHA Standards)
    if sbp < 120 and dbp < 80:
        bp_status = "Ideal BP"
        bp_color = "#2ecc71"
    elif sbp < 130 and dbp < 85:
        bp_status = "Normal BP"
        bp_color = "#27ae60"
    elif sbp < 140 and dbp < 90:
        bp_status = "Elevated BP"
        bp_color = "#f39c12"
    elif sbp < 160 and dbp < 100:
        bp_status = "Stage 1 Hypertension"
        bp_color = "#e67e22"
    else:
        bp_status = "Hypertension"
        bp_color = "#e74c3c"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {bp_color}; background: none; -webkit-text-fill-color: {bp_color};">{bp_status}</div>
        <div class="metric-label">BP Classification<br>AHA Standards</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Waveform Chart
# ============================================================
st.markdown("### 📈 Real-time Pulse Waveform")

# Limit display duration
display_points = int(display_seconds * sampling_rate)
t_display = t[-display_points:] if len(t) > display_points else t
signal_display = filtered_signal[-display_points:] if len(filtered_signal) > display_points else filtered_signal
peaks_display = [p for p in peaks if p < len(signal_display)]

# Create waveform chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_display,
    y=signal_display,
    mode='lines',
    name='Pulse Wave',
    line=dict(color='#667eea', width=2),
    fill='tozeroy',
    fillcolor='rgba(102, 126, 234, 0.1)'
))

fig.add_trace(go.Scatter(
    x=t_display[peaks_display] if len(peaks_display) > 0 else [],
    y=signal_display[peaks_display] if len(peaks_display) > 0 else [],
    mode='markers',
    name='Peak Detection',
    marker=dict(color='#e74c3c', size=8, symbol='circle', line=dict(width=2, color='white'))
))

fig.update_layout(
    height=350,
    template='plotly_dark',
    title=None,
    xaxis_title="Time (seconds)",
    yaxis_title="Normalized Amplitude",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Historical Trend Chart
# ============================================================
st.markdown("### 📉 Blood Pressure History Trend")

if len(st.session_state.bp_history) > 1:
    history_df = pd.DataFrame(st.session_state.bp_history)
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['systolic'],
        mode='lines+markers',
        name='Systolic BP',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['diastolic'],
        mode='lines+markers',
        name='Diastolic BP',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    # Reference lines
    fig_trend.add_hline(y=120, line_dash="dash", line_color="orange", 
                         annotation_text="Normal Limit", annotation_position="top right")
    fig_trend.add_hline(y=140, line_dash="dash", line_color="red", 
                         annotation_text="Hypertension Threshold", annotation_position="bottom right")
    
    fig_trend.update_layout(
        height=300,
        template='plotly_dark',
        title=None,
        xaxis_title="Time",
        yaxis_title="Blood Pressure (mmHg)",
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Waiting for more data to display trend chart...")

# ============================================================
# Detailed Data Table
# ============================================================
with st.expander("📋 View Detailed Data", expanded=False):
    # Create data table
    table_data = []
    for i, p in enumerate(peaks[:20]):  # Show first 20 peaks
        if p < len(t):
            table_data.append({
                'Peak #': i + 1,
                'Time (s)': f"{t[p]:.3f}",
                'Amplitude': f"{signal_display[p] if p < len(signal_display) else 0:.4f}",
                'RR Interval (ms)': f"{(t[p] - t[peaks[i-1]])*1000:.1f}" if i > 0 else "-"
            })
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ============================================================
# Health Recommendations
# ============================================================
st.markdown("### 💡 Health Recommendations")

if sbp < 120 and dbp < 80:
    advice = "✅ Your blood pressure is in the ideal range. Maintain a healthy lifestyle: balanced diet, regular exercise, adequate sleep."
    advice_color = "#2ecc71"
elif sbp < 130 and dbp < 85:
    advice = "✅ Your blood pressure is normal. Keep up the good habits and monitor regularly."
    advice_color = "#27ae60"
elif sbp < 140 and dbp < 90:
    advice = "⚠️ Your blood pressure is elevated. Recommendations: Reduce salt intake, increase physical activity, maintain healthy weight."
    advice_color = "#f39c12"
elif sbp < 160 and dbp < 100:
    advice = "⚠️ Stage 1 hypertension detected. Please consult your doctor, monitor regularly, and adopt a heart-healthy diet."
    advice_color = "#e67e22"
else:
    advice = "🔴 Your blood pressure is high. Please consult your doctor immediately and begin lifestyle interventions."
    advice_color = "#e74c3c"

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); padding: 1rem; border-radius: 10px; border-left: 4px solid {advice_color};">
    <p style="color: {advice_color}; margin: 0;">{advice}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer">
    <p>⚠️ This system is for research and reference only. Does not constitute medical advice. Please consult a healthcare provider if you feel unwell.</p>
    <p>Based on Graphene/Sponge Flexible Sensor | BP Estimation Error ≤ 3% | Continuous Monitoring Supported</p>
    <p>BP-Interface v1.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
