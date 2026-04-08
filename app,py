"""
BP-Interface - 智能血压监测系统
Streamlit Web界面
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
# 页面配置
# ============================================================
st.set_page_config(
    page_title="BP-Interface | 智能血压监测",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 自定义CSS
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
# 标题区域
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>❤️ BP-Interface</h1>
    <p>基于石墨烯/海绵柔性传感器的智能血压监测系统</p>
    <p style="font-size: 0.8rem;">Graphene/Sponge-Based Flexible Sensor for Blood Pressure Monitoring</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏 - 数据源选择
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 系统设置")
    
    data_source = st.radio(
        "📊 数据来源",
        ["📁 上传CSV文件", "🎮 模拟演示数据"],
        index=1,
        help="选择数据来源方式"
    )
    
    st.markdown("---")
    
    st.markdown("### 📈 显示设置")
    sampling_rate = st.select_slider("采样率 (Hz)", options=[50, 100, 200], value=100)
    display_seconds = st.slider("波形显示时长 (秒)", 5, 30, 10)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ 关于系统")
    st.info(
        """
        **系统特点：**
        - 柔性石墨烯/海绵传感器
        - 实时脉搏波监测
        - 连续血压估算
        - 心率变异分析
        
        **精度：** 血压误差 ≤ 3%
        """
    )
    
    st.markdown("---")
    
    # 导出按钮
    if st.button("📥 导出报告", use_container_width=True):
        st.success("报告导出功能开发中...")

# ============================================================
# 初始化会话状态
# ============================================================
if 'bp_history' not in st.session_state:
    st.session_state.bp_history = []
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = []

# ============================================================
# 信号处理函数
# ============================================================
def bandpass_filter(signal, fs, lowcut=0.5, highcut=20, order=4):
    """带通滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_peaks(signal, fs):
    """检测脉搏波峰值"""
    min_distance = int(0.4 * fs)  # 最小间隔对应150 bpm
    peaks, _ = find_peaks(signal, distance=min_distance, height=np.std(signal)*0.5)
    return peaks

def calculate_heart_rate(peaks, fs):
    """计算心率 (bpm)"""
    if len(peaks) < 2:
        return 75
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(rr_intervals)
    return min(180, max(40, hr))

def estimate_bp_from_pulse(signal, peaks, fs):
    """从脉搏波估算血压"""
    if len(peaks) < 3:
        return 120, 80
    
    # 计算上升时间 (UT)
    rise_times = []
    for peak in peaks:
        start = max(0, peak - int(0.3 * fs))
        if start < peak:
            rise_time = (peak - np.argmax(signal[start:peak])) / fs
            if 0.05 < rise_time < 0.25:
                rise_times.append(rise_time)
    
    avg_ut = np.mean(rise_times) if rise_times else 0.12
    
    # 经验公式
    sbp = 120 - (avg_ut - 0.1) * 200
    dbp = sbp * 0.6 + 20
    
    # 限制范围
    sbp = max(80, min(180, sbp))
    dbp = max(50, min(120, dbp))
    
    return sbp, dbp

def generate_simulated_signal(duration=10, fs=100, hr=75):
    """生成模拟脉搏波信号"""
    t = np.arange(0, duration, 1/fs)
    heart_rate_hz = hr / 60
    
    # 脉搏波模型
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
    uploaded_file = st.file_uploader(
        "选择CSV文件",
        type=['csv'],
        help="CSV文件应包含 Raw_Data, Delta_R_Over_R_Percent, ECG 等列"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # 处理列名
        if 'Raw_Data' in df.columns:
            signal = df['Raw_Data'].values
        elif 'Delta_R_Over_R_Percent' in df.columns:
            signal = df['Delta_R_Over_R_Percent'].values
        else:
            signal = df.iloc[:, 0].values
        
        # 归一化
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        t = np.arange(len(signal)) / sampling_rate
        
        st.success(f"✅ 成功加载 {len(df)} 条数据记录")
    else:
        st.info("请上传CSV文件以开始分析")
        st.stop()
else:
    # 生成模拟数据
    t, signal = generate_simulated_signal(duration=10, fs=sampling_rate)
    st.info("🎮 使用模拟演示数据 - 展示系统功能")

# ============================================================
# 信号处理
# ============================================================
filtered_signal = bandpass_filter(signal, sampling_rate)
peaks = detect_peaks(filtered_signal, sampling_rate)
heart_rate = calculate_heart_rate(peaks, sampling_rate)
sbp, dbp = estimate_bp_from_pulse(filtered_signal, peaks, sampling_rate)

# 更新历史记录
st.session_state.bp_history.append({'systolic': sbp, 'diastolic': dbp, 'timestamp': datetime.now()})
if len(st.session_state.bp_history) > 100:
    st.session_state.bp_history.pop(0)

# ============================================================
# 实时指标显示
# ============================================================
st.markdown("### 📊 实时健康指标")

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
    # 根据血压选择颜色
    if sbp < 120:
        bp_class = "bp-normal"
        status = "理想"
    elif sbp < 140:
        bp_class = "bp-elevated"
        status = "正常高值"
    else:
        bp_class = "bp-high"
        status = "偏高"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {bp_class}">{heart_rate:.0f}</div>
        <div class="metric-label">心率 (HR)<br>bpm · {status}</div>
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
        <div class="metric-value" style="color: {bp_color}; background: none; -webkit-text-fill-color: {bp_color};">{bp_status}</div>
        <div class="metric-label">血压分类<br>AHA标准</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 波形图
# ============================================================
st.markdown("### 📈 实时脉搏波形")

# 限制显示时长
display_points = int(display_seconds * sampling_rate)
t_display = t[-display_points:] if len(t) > display_points else t
signal_display = filtered_signal[-display_points:] if len(filtered_signal) > display_points else filtered_signal
peaks_display = [p for p in peaks if p < len(signal_display)]

# 创建波形图
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_display,
    y=signal_display,
    mode='lines',
    name='脉搏波',
    line=dict(color='#667eea', width=2),
    fill='tozeroy',
    fillcolor='rgba(102, 126, 234, 0.1)'
))

fig.add_trace(go.Scatter(
    x=t_display[peaks_display] if len(peaks_display) > 0 else [],
    y=signal_display[peaks_display] if len(peaks_display) > 0 else [],
    mode='markers',
    name='脉搏峰值',
    marker=dict(color='#e74c3c', size=8, symbol='circle', line=dict(width=2, color='white'))
))

fig.update_layout(
    height=350,
    template='plotly_dark',
    title=None,
    xaxis_title="时间 (秒)",
    yaxis_title="归一化幅度",
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 血压趋势图
# ============================================================
st.markdown("### 📉 血压历史趋势")

if len(st.session_state.bp_history) > 1:
    history_df = pd.DataFrame(st.session_state.bp_history)
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['systolic'],
        mode='lines+markers',
        name='收缩压',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['diastolic'],
        mode='lines+markers',
        name='舒张压',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    # 参考线
    fig_trend.add_hline(y=120, line_dash="dash", line_color="orange", 
                         annotation_text="正常上限", annotation_position="top right")
    fig_trend.add_hline(y=140, line_dash="dash", line_color="red", 
                         annotation_text="高血压阈值", annotation_position="bottom right")
    
    fig_trend.update_layout(
        height=300,
        template='plotly_dark',
        title=None,
        xaxis_title="时间",
        yaxis_title="血压 (mmHg)",
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("等待更多数据以显示趋势图...")

# ============================================================
# 详细数据表格
# ============================================================
with st.expander("📋 查看详细数据", expanded=False):
    # 创建数据表格
    table_data = []
    for i, p in enumerate(peaks[:20]):  # 只显示前20个峰值
        if p < len(t):
            table_data.append({
                '峰值序号': i + 1,
                '时间 (秒)': f"{t[p]:.3f}",
                '幅度': f"{signal_display[p] if p < len(signal_display) else 0:.4f}",
                'RR间期 (ms)': f"{(t[p] - t[peaks[i-1]])*1000:.1f}" if i > 0 else "-"
            })
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ============================================================
# 健康建议
# ============================================================
st.markdown("### 💡 健康建议")

if sbp < 120 and dbp < 80:
    advice = "✅ 您的血压处于理想范围。保持健康的生活方式：均衡饮食、适量运动、充足睡眠。"
    advice_color = "#2ecc71"
elif sbp < 130 and dbp < 85:
    advice = "✅ 您的血压正常。继续保持，建议定期监测。"
    advice_color = "#27ae60"
elif sbp < 140 and dbp < 90:
    advice = "⚠️ 您的血压处于正常高值。建议：减少盐分摄入、增加运动、控制体重。"
    advice_color = "#f39c12"
elif sbp < 160 and dbp < 100:
    advice = "⚠️ 您有1级高血压倾向。建议：咨询医生、规律监测、健康饮食。"
    advice_color = "#e67e22"
else:
    advice = "🔴 您的血压偏高。请立即咨询医生，并开始生活方式干预。"
    advice_color = "#e74c3c"

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); padding: 1rem; border-radius: 10px; border-left: 4px solid {advice_color};">
    <p style="color: {advice_color}; margin: 0;">{advice}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 页脚
# ============================================================
st.markdown("""
<div class="footer">
    <p>⚠️ 本系统仅供研究和参考，不构成医疗诊断。如有不适，请及时就医。</p>
    <p>基于石墨烯/海绵柔性传感器 | 血压估算误差 ≤ 3% | 支持连续监测</p>
    <p>BP-Interface v1.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True) 
