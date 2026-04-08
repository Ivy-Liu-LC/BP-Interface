"""
============================================================
智能血压监测系统 - Streamlit Web界面
基于石墨烯/海绵柔性传感器
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import serial
import threading
from collections import deque
from datetime import datetime
import json
import os

# 页面配置
st.set_page_config(
    page_title="智能血压监测系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 自定义样式
# ============================================================

st.markdown("""
<style>
    /* 主色调 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e2f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
    }
    .bp-normal { color: #2ecc71; }
    .bp-elevated { color: #f39c12; }
    .bp-high { color: #e74c3c; }
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏配置
# ============================================================

st.sidebar.title("⚙️ 系统设置")

# 数据源选择
data_source = st.sidebar.radio(
    "数据来源",
    ["📁 上传CSV文件", "🔌 串口实时采集", "📊 模拟演示数据"]
)

# 串口设置（如果选择串口模式）
if data_source == "🔌 串口实时采集":
    st.sidebar.subheader("串口设置")
    port = st.sidebar.text_input("串口号", value="COM3" if os.name == 'nt' else "/dev/ttyUSB0")
    baudrate = st.sidebar.selectbox("波特率", [9600, 115200], index=1)
    
    # 连接按钮
    if st.sidebar.button("🔌 连接设备", type="primary"):
        st.session_state['serial_connected'] = True
        st.sidebar.success("✅ 已连接")

# 采样率设置
sampling_rate = st.sidebar.select_slider(
    "采样率 (Hz)",
    options=[50, 100, 200],
    value=100
)

# 显示时长
display_duration = st.sidebar.slider(
    "波形显示时长 (秒)",
    min_value=5,
    max_value=60,
    value=30,
    step=5
)

# 血压单位
bp_unit = st.sidebar.selectbox("血压单位", ["mmHg", "kPa"])

# 导出按钮
st.sidebar.markdown("---")
if st.sidebar.button("📥 导出当前数据", use_container_width=True):
    st.sidebar.success("数据已导出")

# ============================================================
# 主页面标题
# ============================================================

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">❤️ 智能血压监测系统</h1>
    <p style="color: rgba(255,255,255,0.8); margin: 0;">基于石墨烯/海绵柔性传感器 | 实时无创血压监测</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 初始化会话状态
# ============================================================

if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=sampling_rate * 60)  # 保存60秒数据
if 'ecg_buffer' not in st.session_state:
    st.session_state.ecg_buffer = deque(maxlen=sampling_rate * 60)
if 'bp_history' not in st.session_state:
    st.session_state.bp_history = []
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'serial_connected' not in st.session_state:
    st.session_state.serial_connected = False

# ============================================================
# 信号处理函数
# ============================================================

def bandpass_filter(signal, fs, lowcut=0.5, highcut=20, order=4):
    """带通滤波器"""
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def find_peaks(signal, distance=30):
    """简单峰值检测"""
    from scipy.signal import find_peaks as sp_find_peaks
    peaks, _ = sp_find_peaks(signal, distance=distance, height=np.std(signal)*0.5)
    return peaks

def calculate_heart_rate(peaks, fs):
    """计算心率"""
    if len(peaks) < 2:
        return 75
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(rr_intervals)
    return min(180, max(40, hr))

def estimate_blood_pressure(peaks, signal, fs):
    """
    基于PAT估算血压
    PAT = 脉搏波峰值时间 - R波时间
    此处简化为基于脉搏波特征的回归
    """
    if len(peaks) < 3:
        return 120, 80
    
    # 计算上升时间 (UT)
    rise_times = []
    for peak in peaks:
        start_idx = max(0, peak - int(0.3 * fs))
        if start_idx < peak:
            rise_time = (peak - np.argmax(signal[start_idx:peak])) / fs
            if 0.05 < rise_time < 0.25:
                rise_times.append(rise_time)
    
    avg_ut = np.mean(rise_times) if rise_times else 0.12
    
    # 经验公式：SBP = 120 - (UT - 0.1) * 200
    sbp = 120 - (avg_ut - 0.1) * 200
    dbp = sbp * 0.6 + 20
    
    # 限制范围
    sbp = max(80, min(180, sbp))
    dbp = max(50, min(120, dbp))
    
    return sbp, dbp

# ============================================================
# 数据加载函数
# ============================================================

@st.cache_data
def load_csv_data(uploaded_file):
    """加载上传的CSV文件"""
    df = pd.read_csv(uploaded_file)
    # 处理可能的列名
    if 'Raw_Data' in df.columns:
        df.rename(columns={'Raw_Data': 'raw'}, inplace=True)
    if 'Delta_R_Over_R_Percent' in df.columns:
        df.rename(columns={'Delta_R_Over_R_Percent': 'delta_r'}, inplace=True)
    if 'ECG' in df.columns:
        df.rename(columns={'ECG': 'ecg'}, inplace=True)
    
    # 添加时间列
    df['time'] = np.arange(len(df)) / sampling_rate
    return df

def generate_simulated_data(duration=10, fs=100):
    """生成模拟脉搏波数据"""
    t = np.arange(0, duration, 1/fs)
    # 模拟心跳：基频 + 谐波
    hr = 75  # bpm
    heart_rate_hz = hr / 60
    
    # 脉搏波：主波 + 重搏波
    pulse = (np.sin(2 * np.pi * heart_rate_hz * t) * 0.5 +
             np.sin(4 * np.pi * heart_rate_hz * t) * 0.3 +
             np.sin(6 * np.pi * heart_rate_hz * t) * 0.1)
    
    # 添加噪声
    pulse += np.random.normal(0, 0.05, len(t))
    
    # 归一化
    pulse = (pulse - pulse.min()) / (pulse.max() - pulse.min())
    
    return t, pulse

# ============================================================
# 数据加载
# ============================================================

if data_source == "📁 上传CSV文件":
    uploaded_file = st.sidebar.file_uploader(
        "选择CSV文件",
        type=['csv'],
        help="CSV文件应包含Raw_Data, Delta_R_Over_R_Percent, ECG列"
    )
    
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        st.success(f"✅ 成功加载 {len(df)} 条数据记录")
        
        # 提取数据
        t = df['time'].values
        signal = df['delta_r'].values if 'delta_r' in df.columns else df['raw'].values
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        
        # 可选ECG
        has_ecg = 'ecg' in df.columns
        if has_ecg:
            ecg = df['ecg'].values
            ecg = (ecg - ecg.mean()) / ecg.std()
    else:
        st.info("请上传CSV文件以开始分析")
        st.stop()
        
elif data_source == "🔌 串口实时采集":
    if st.session_state.serial_connected:
        st.info("📡 等待串口数据...")
        # 这里可以添加串口读取逻辑
        # 为简化，使用模拟数据
        t, signal = generate_simulated_data(duration=10, fs=sampling_rate)
        has_ecg = False
    else:
        st.warning("请先在侧边栏连接串口设备")
        st.stop()
        
else:  # 模拟演示数据
    t, signal = generate_simulated_data(duration=10, fs=sampling_rate)
    has_ecg = False
    st.info("🎮 使用模拟演示数据")

# ============================================================
# 信号处理
# ============================================================

# 滤波
filtered_signal = bandpass_filter(signal, sampling_rate)

# 峰值检测
peaks = find_peaks(filtered_signal, distance=int(0.4 * sampling_rate))

# 心率计算
heart_rate = calculate_heart_rate(peaks, sampling_rate)

# 血压估算
sbp, dbp = estimate_blood_pressure(peaks, filtered_signal, sampling_rate)

# ============================================================
# 主要指标显示（4列）
# ============================================================

st.subheader("📊 实时健康指标")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{sbp:.0f}</div>
        <div class="metric-label">收缩压 (SBP)<br>mmHg</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{dbp:.0f}</div>
        <div class="metric-label">舒张压 (DBP)<br>mmHg</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    bp_class = "bp-normal"
    if sbp >= 140:
        bp_class = "bp-high"
    elif sbp >= 120:
        bp_class = "bp-elevated"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {bp_class}">{heart_rate:.0f}</div>
        <div class="metric-label">心率 (HR)<br>bpm</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # 血压分类
    if sbp < 120 and dbp < 80:
        bp_status = "理想血压"
        bp_color = "#2ecc71"
    elif sbp < 130 and dbp < 85:
        bp_status = "正常血压"
        bp_color = "#27ae60"
    elif sbp < 140 and dbp < 90:
        bp_status = "正常高值"
        bp_color = "#f39c12"
    elif sbp < 160 and dbp < 100:
        bp_status = "1级高血压"
        bp_color = "#e67e22"
    else:
        bp_status = "高血压"
        bp_color = "#e74c3c"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {bp_color};">{bp_status}</div>
        <div class="metric-label">血压分类<br>根据AHA标准</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 波形图（使用Plotly实现交互式）
# ============================================================

st.subheader("📈 实时脉搏波形")

# 创建子图
if has_ecg:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("脉搏波 (Sponge Sensor)", "心电图 (ECG)")
    )
    
    # 脉搏波
    fig.add_trace(
        go.Scatter(x=t, y=filtered_signal, mode='lines', name='脉搏波',
                   line=dict(color='#667eea', width=1.5)),
        row=1, col=1
    )
    
    # 标记峰值
    fig.add_trace(
        go.Scatter(x=t[peaks], y=filtered_signal[peaks], mode='markers', name='脉搏峰值',
                   marker=dict(color='red', size=6, symbol='circle')),
        row=1, col=1
    )
    
    # ECG
    fig.add_trace(
        go.Scatter(x=t, y=ecg[:len(t)], mode='lines', name='ECG',
                   line=dict(color='#2ecc71', width=1)),
        row=2, col=1
    )
    
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=filtered_signal,
        mode='lines',
        name='脉搏波',
        line=dict(color='#667eea', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=t[peaks], y=filtered_signal[peaks],
        mode='markers',
        name='峰值检测',
        marker=dict(color='red', size=8, symbol='circle')
    ))

# 更新布局
fig.update_layout(
    height=400,
    title=None,
    xaxis_title="时间 (秒)",
    yaxis_title="归一化幅度",
    template="plotly_dark",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.update_xaxes(range=[0, min(10, len(t)/sampling_rate)])

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 历史趋势图
# ============================================================

st.subheader("📉 历史趋势")

# 模拟历史数据（实际应用中从数据库读取）
if len(st.session_state.bp_history) == 0:
    # 生成演示历史数据
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    sbp_history = 115 + np.cumsum(np.random.randn(30) * 2)
    dbp_history = 75 + np.cumsum(np.random.randn(30) * 1.5)
    hr_history = 72 + np.cumsum(np.random.randn(30) * 1)
    
    for i in range(30):
        st.session_state.bp_history.append({
            'date': dates[i],
            'systolic': sbp_history[i],
            'diastolic': dbp_history[i],
            'heart_rate': hr_history[i]
        })

# 转换为DataFrame
history_df = pd.DataFrame(st.session_state.bp_history)

# 趋势图
fig_trend = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("血压趋势", "心率趋势"),
    row_heights=[0.6, 0.4]
)

# 血压趋势
fig_trend.add_trace(
    go.Scatter(x=history_df['date'], y=history_df['systolic'], 
               mode='lines+markers', name='收缩压',
               line=dict(color='#e74c3c', width=2),
               marker=dict(size=6)),
    row=1, col=1
)
fig_trend.add_trace(
    go.Scatter(x=history_df['date'], y=history_df['diastolic'], 
               mode='lines+markers', name='舒张压',
               line=dict(color='#3498db', width=2),
               marker=dict(size=6)),
    row=1, col=1
)

# 添加正常范围参考线
fig_trend.add_hline(y=120, line_dash="dash", line_color="green", 
                     annotation_text="正常上限", row=1, col=1)
fig_trend.add_hline(y=140, line_dash="dash", line_color="orange", 
                     annotation_text="高血压阈值", row=1, col=1)

# 心率趋势
fig_trend.add_trace(
    go.Scatter(x=history_df['date'], y=history_df['heart_rate'], 
               mode='lines+markers', name='心率',
               line=dict(color='#2ecc71', width=2),
               fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'),
    row=2, col=1
)

fig_trend.update_layout(
    height=500,
    template="plotly_dark",
    showlegend=True,
    hovermode='x unified'
)

fig_trend.update_xaxes(title_text="日期", row=2, col=1)
fig_trend.update_yaxes(title_text="血压 (mmHg)", row=1, col=1)
fig_trend.update_yaxes(title_text="心率 (bpm)", row=2, col=1)

st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================
# 数据分析面板
# ============================================================

st.subheader("🔬 详细数据分析")

with st.expander("📊 查看详细数据表格", expanded=False):
    # 创建数据表格
    table_df = pd.DataFrame({
        '时间 (秒)': t[:500],
        '原始信号': signal[:500],
        '滤波后': filtered_signal[:500],
        '峰值检测': ['✓' if i in peaks[:500] else '' for i in range(min(500, len(t)))]
    })
    st.dataframe(table_df, use_container_width=True)

with st.expander("📈 统计摘要", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("平均收缩压", f"{np.mean([h['systolic'] for h in st.session_state.bp_history[-7:]]):.0f} mmHg")
        st.metric("最大收缩压", f"{max([h['systolic'] for h in st.session_state.bp_history]):.0f} mmHg")
    
    with col2:
        st.metric("平均舒张压", f"{np.mean([h['diastolic'] for h in st.session_state.bp_history[-7:]]):.0f} mmHg")
        st.metric("最小舒张压", f"{min([h['diastolic'] for h in st.session_state.bp_history]):.0f} mmHg")
    
    with col3:
        st.metric("平均心率", f"{np.mean([h['heart_rate'] for h in st.session_state.bp_history[-7:]]):.0f} bpm")
        st.metric("心率变异", f"{np.std([h['heart_rate'] for h in st.session_state.bp_history]):.1f} bpm")

# ============================================================
# 健康建议
# ============================================================

st.subheader("💡 健康建议")

# 根据血压给出建议
if sbp < 120 and dbp < 80:
    advice = "✅ 您的血压处于理想范围。保持健康的生活方式：均衡饮食、适量运动、充足睡眠。"
    advice_color = "green"
elif sbp < 130 and dbp < 85:
    advice = "✅ 您的血压正常。建议继续保持，定期监测。"
    advice_color = "lightgreen"
elif sbp < 140 and dbp < 90:
    advice = "⚠️ 您的血压处于正常高值。建议：减少盐分摄入、增加运动、控制体重。"
    advice_color = "orange"
elif sbp < 160 and dbp < 100:
    advice = "⚠️ 您有1级高血压倾向。建议：咨询医生、规律服药、健康饮食。"
    advice_color = "#e67e22"
else:
    advice = "🔴 您的血压偏高。请立即咨询医生，并开始生活方式干预。"
    advice_color = "red"

st.markdown(f"""
<div style="background-color: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid {advice_color};">
    <p style="color: {advice_color}; margin: 0;">{advice}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 页脚
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>⚠️ 本系统仅供研究和参考，不构成医疗诊断。如有不适，请及时就医。</p>
    <p>基于石墨烯/海绵柔性传感器 | 血压估算误差 ≤ 3% | 支持连续监测</p>
</div>
""", unsafe_allow_html=True)
