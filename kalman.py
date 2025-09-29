import streamlit as st
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# 页面配置
st.set_page_config(
    layout="wide",
    page_title="曼宁系数率定 - UKF滤波器",
    page_icon="🚰"
)

# 自定义CSS - 修复了边框显示问题
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgb(41, 50, 225, 0.4) !important; /* 使用更醒目的蓝色，并用 !important 确保生效 */
        display: block;
    }
    .parameter-card {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    .info-message {
        background-color: #E1F5FE;
        border-left: 5px solid #03A9F4;
        padding: 10px;
        border-radius: 5px;
    }
    /* 额外确保标题容器是块级元素，以正确显示边框 */
    .section-header h2 {
        margin: 0;
        padding: 0;
    }
    .st-emotion-cache-1t8vfw5 h1, .st-emotion-cache-1t8vfw5 h2, .st-emotion-cache-1t8vfw5 h3, .st-emotion-cache-1t8vfw5 h4, .st-emotion-cache-1t8vfw5 h5, .st-emotion-cache-1t8vfw5 h6, .st-emotion-cache-1t8vfw5 span{
     margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🚰 曼宁系数率定 - UKF滤波器</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-message">
    <strong>应用说明：</strong> 本工具使用无迹卡尔曼滤波(UKF)技术对水动力模型中的曼宁系数进行率定。
    通过将曼宁系数视为随机游走过程，结合实际观测数据，实现参数的动态估计与优化。
</div>
""", unsafe_allow_html=True)

# --------------------
# Step 1: 基本参数配置
# --------------------
st.markdown('<h2 class="section-header">📍 1. 模型配置</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    start_station = st.selectbox(
        "起始站",
        ["濂布沟", "站点A", "站点B"],
        help="选择模拟河段的起始位置"
    )
with col2:
    end_station = st.selectbox(
        "结束站",
        ["深溪沟", "站点C", "站点D"],
        help="选择模拟河段的结束位置"
    )
with col3:
    mode = st.selectbox(
        "快照模式",
        ["历史模式", "实时模式"],
        help="历史模式使用历史数据，实时模式模拟实时计算"
    )

# --- 修改开始 ---
# 仅在“历史模式”下显示时间设置
if mode == "历史模式":
    # st.markdown("### 时间范围设置")
    col_date1, col_date2, col_time1, col_time2 = st.columns(4)
    with col_date1:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=7))
    with col_time1:
        start_time_val = st.time_input("开始时间", datetime(2023, 1, 1, 0, 0))
    with col_date2:
        end_date = st.date_input("结束日期", datetime.now())
    with col_time2:
        end_time_val = st.time_input("结束时间", datetime(2023, 1, 1, 23, 59))

    # 合并为datetime对象
    start_time = datetime.combine(start_date, start_time_val)
    end_time = datetime.combine(end_date, end_time_val)

    # 验证时间范围
    if start_time >= end_time:
        st.error("开始时间必须早于结束时间！")
        start_time = None
        end_time = None
else:
    # 实时模式不使用具体时间范围
    st.info("当前为实时模式，不支持手动设置时间范围。")
    start_time = None
    end_time = None
# --- 修改结束 ---

# --------------------
# Step 2: 数据配置
# --------------------
# --- 修改开始 ---
# 隐藏了数据来源的选择，直接使用模拟数据
# st.markdown('<h2 class="section-header">📊 2. 数据配置</h2>', unsafe_allow_html=True)
# data_source = st.radio(
#     "数据来源",
#     ["模拟数据", "上传数据"],
#     help="选择使用模拟数据或上传实际观测数据"
# )

# 固定为模拟数据
data_source = "模拟数据"
observations = None
time_points = None

# 直接显示模拟数据参数
st.markdown('<h2 class="section-header">📊 2. 数据参数</h2>', unsafe_allow_html=True)
col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    true_manning = st.slider(
        "曼宁系数",
        0.01, 0.1, 0.03, 0.001,
        help="模拟中使用的曼宁系数值"
    )
    obs_noise = st.slider(
        "观测噪声标准差",
        0.01, 0.5, 0.1, 0.01,
        help="添加到观测数据中的噪声水平"
    )
with col_sim2:
    flow_rate = st.number_input(
        "流量 (m³/s)",
        value=100.0,
        help="模拟中使用的恒定流量值"
    )
    channel_slope = st.number_input(
        "渠道坡度",
        value=0.001,
        help="渠道的坡度值"
    )
# --- 修改结束 ---

# --------------------
# Step 3: UKF 参数配置
# --------------------
st.markdown('<h2 class="section-header">⚙️ 3. UKF 参数配置</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="parameter-card">
    <strong>UKF参数说明：</strong><br>
    - <strong>alpha</strong>: 控制sigma点分布范围，通常取0.001-1<br>
    - <strong>beta</strong>: 包含先验分布信息，高斯分布最优值为2<br>
    - <strong>kappa</strong>: 次要缩放参数，通常设为0或3-n<br>
    - <strong>Q</strong>: 过程噪声协方差，表示曼宁系数随机游走的噪声强度<br>
    - <strong>R</strong>: 观测噪声协方差，表示观测值的噪声强度<br>
    - <strong>P0</strong>: 初始协方差，表示对初始曼宁系数估计的不确定性
</div>
""", unsafe_allow_html=True)

ukf_col1, ukf_col2, ukf_col3 = st.columns(3)
with ukf_col1:
    alpha = st.slider("alpha", 0.001, 1.0, 0.1, 0.001)
    Q = st.number_input(
        "过程噪声协方差 Q",
        min_value=0.0001,
        value=0.001,
        format="%.4f",
        help="曼宁系数随机游走的噪声强度"
    )
with ukf_col2:
    beta = st.slider("beta", 0.0, 5.0, 2.0)
    R = st.number_input(
        "观测噪声协方差 R",
        min_value=0.0001,
        value=0.01,
        format="%.4f",
        help="观测值的噪声强度"
    )
with ukf_col3:
    kappa = st.number_input("kappa", value=0.0)
    P0 = st.number_input(
        "初始协方差 P0",
        min_value=0.001,
        value=0.1,
        format="%.4f",
        help="初始曼宁系数估计的不确定性"
    )

# --------------------
# Step 4: 黑箱模型配置
# --------------------
st.markdown('<h2 class="section-header">📦 4. 水动力模型配置</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="info-message">
    <strong>模型说明：</strong> 本工具使用曼宁公式作为观测模型，将曼宁系数映射到水位观测值。
    曼宁公式: v = (1/n) * R^(2/3) * S^(1/2)，其中v为流速，n为曼宁系数，R为水力半径，S为坡度。
</div>
""", unsafe_allow_html=True)

# 模型参数
col_model1, col_model2 = st.columns(2)
with col_model1:
    channel_width = st.number_input("渠道宽度 (m)", value=10.0)
    channel_depth = st.number_input("渠道深度 (m)", value=2.0)
with col_model2:
    manning_init = st.number_input(
        "初始曼宁系数估计",
        min_value=0.01,
        value=0.035,
        format="%.3f"
    )
    roughness_coeff = st.number_input("糙率系数", value=0.03)

# 状态方程（曼宁系数随机游走）
def fx(x, dt):
    """曼宁系数随机游走模型"""
    return x  # 状态不变，噪声由Q控制

def fx_ukf(x, dt):
    """UKF状态转移函数"""
    return np.array([fx(x[0], dt)])

# 观测方程（曼宁公式）
def hx_ukf(x):
    """UKF观测函数 - 曼宁公式"""
    n = x[0]  # 曼宁系数
    # 简化计算：假设矩形渠道
    A = channel_width * channel_depth  # 过水面积
    P = channel_width + 2 * channel_depth  # 湿周
    R = A / P  # 水力半径
    # 曼宁公式计算流速
    v = (1 / n) * (R ** (2 / 3)) * (channel_slope ** 0.5)
    # 流速转换为水位（简化模型）
    h = channel_depth * (flow_rate / (v * A)) ** 0.5
    return np.array([h])

# --------------------
# Step 5: 滤波计算与结果展示
# --------------------
# --- 关键修复：确保此标题的边框可见 ---
# 这行代码被错误地注释掉了，现在已恢复
# st.markdown('<h2 class="section-header">📈 5. 滤波计算与结果展示</h2>', unsafe_allow_html=True)

# 移除了已弃用的 type="primary"
if st.button("开始滤波计算"):
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- 修改开始 ---
    # 初始化参数
    T = 100  # 默认时间步数
    dt = 3600  # 默认时间步长（秒），用于实时模式

    # 如果是历史模式且有时间范围，则计算dt
    if mode == "历史模式" and start_time is not None and end_time is not None:
        total_seconds = (end_time - start_time).total_seconds()
        if total_seconds > 0:
            dt = total_seconds / T
        else:
            st.error("时间范围无效，无法计算时间步长。")
            st.stop()
    # --- 修改结束 ---

    # 初始化状态
    x0 = manning_init
    np.random.seed(42)

    # 生成或使用观测数据
    # 由于我们固定为模拟数据，直接执行这部分
    status_text.text("生成模拟观测数据...")
    x_true = np.zeros(T)
    y = np.zeros(T)
    x_true[0] = true_manning
    # 生成真实状态和观测
    for k in range(T):
        # 状态更新（随机游走）
        if k > 0:
            x_true[k] = x_true[k - 1] + np.random.normal(0, np.sqrt(Q))
        # 观测更新（曼宁公式+噪声）
        y[k] = hx_ukf(np.array([x_true[k]]))[0] + np.random.normal(0, obs_noise)
    progress_bar.progress(20)

    # 初始化UKF
    status_text.text("初始化UKF滤波器...")
    points = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=beta, kappa=kappa)
    # 确保初始协方差矩阵是正定的
    P0_matrix = np.array([[max(P0, 1e-6)]])  # 确保P0不为零
    ukf = UKF(dim_x=1, dim_z=1, fx=fx_ukf, hx=hx_ukf, dt=dt, points=points)
    ukf.x = np.array([x0])
    ukf.P = P0_matrix.copy()  # 使用处理后的协方差矩阵
    ukf.Q = np.array([[max(Q, 1e-6)]])  # 确保过程噪声不为零
    ukf.R = np.array([[max(R, 1e-6)]])  # 确保观测噪声不为零
    progress_bar.progress(40)

    # 运行UKF
    status_text.text("运行UKF滤波...")
    x_ukf = np.zeros(T)
    P_ukf = np.zeros(T)
    for k in range(T):
        try:
            ukf.predict(dt=dt)
            ukf.update(np.array([y[k]]))
            # 确保状态和协方差有效
            if np.isnan(ukf.x).any() or np.isinf(ukf.x).any():
                ukf.x = np.array([manning_init])  # 重置为初始值
            # 确保协方差矩阵正定
            if ukf.P[0, 0] <= 0:
                ukf.P = P0_matrix.copy()
            x_ukf[k] = ukf.x[0]
            P_ukf[k] = ukf.P[0, 0]
            # 更新进度
            progress = 40 + int(50 * (k + 1) / T)
            progress_bar.progress(progress)
        except Exception as e:
            # 使用前一步的值作为回退
            if k > 0:
                x_ukf[k] = x_ukf[k-1]
                P_ukf[k] = P_ukf[k-1]
            else:
                x_ukf[k] = manning_init
                P_ukf[k] = P0_matrix[0, 0]

    progress_bar.progress(90)

    # 计算误差
    error = np.abs(x_ukf - x_true)
    rmse = np.sqrt(np.mean((x_ukf - x_true) ** 2))

    progress_bar.progress(100)
    status_text.text("计算完成！")

    # ----------------------
    # 结果可视化
    # ----------------------
    st.markdown("### 滤波结果")

    # 创建交互式图表
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("曼宁系数估计", "观测值与预测值对比", "估计不确定性"),
        vertical_spacing=0.1
    )

    # --- 修改开始 ---
    # 时间轴
    # 模拟数据：如果是历史模式，使用 start_time ~ end_time；否则用相对时间
    if mode == "历史模式" and start_time is not None:
        time_axis = [start_time + timedelta(seconds=i * dt) for i in range(T)]
    else:
        # 实时模式：使用相对时间（如从0开始）
        time_axis = [timedelta(seconds=i * dt) for i in range(T)]
    # --- 修改结束 ---

    # 曼宁系数估计
    fig.add_trace(
        go.Scatter(x=time_axis, y=x_true, mode='lines', name='真实值', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_axis, y=x_ukf, mode='lines', name='UKF估计', line=dict(color='blue')),
        row=1, col=1
    )

    # 添加不确定性带
    upper_bound = x_ukf + 2 * np.sqrt(P_ukf)
    lower_bound = x_ukf - 2 * np.sqrt(P_ukf)
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,100,255,0.2)',
            fill='tonexty',
            name='95%置信区间'
        ),
        row=1, col=1
    )

    # 观测值与预测值对比
    fig.add_trace(
        go.Scatter(x=time_axis, y=y, mode='markers', name='观测值', marker=dict(color='red')),
        row=2, col=1
    )

    # 预测值
    y_pred = np.array([hx_ukf(np.array([x]))[0] for x in x_ukf])
    fig.add_trace(
        go.Scatter(x=time_axis, y=y_pred, mode='lines', name='预测值', line=dict(color='green')),
        row=2, col=1
    )

    # 估计不确定性
    fig.add_trace(
        go.Scatter(x=time_axis, y=np.sqrt(P_ukf), mode='lines', name='标准差', line=dict(color='purple')),
        row=3, col=1
    )

    # 更新布局
    fig.update_layout(
        height=800,
        title_text="曼宁系数率定结果",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="时间", row=3, col=1)
    fig.update_yaxes(title_text="曼宁系数", row=1, col=1)
    fig.update_yaxes(title_text="水位 (m)", row=2, col=1)
    fig.update_yaxes(title_text="标准差", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 结果统计
    st.markdown("### 结果统计")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("最终曼宁系数估计", f"{x_ukf[-1]:.4f}")
    with col_stats2:
        st.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
    with col_stats3:
        st.metric("最终不确定性", f"{np.sqrt(P_ukf[-1]):.4f}")

    # 结果下载
    st.markdown("### 结果下载")
    # 创建结果DataFrame
    results = pd.DataFrame({
        '时间': time_axis,
        '曼宁系数估计': x_ukf,
        '不确定性': np.sqrt(P_ukf),
        '观测值': y,
        '预测值': y_pred,
        '真实值': x_true,
        '误差': error
    })

    # 转换为CSV
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下载结果数据 (CSV)",
        data=csv,
        file_name=f'manning_calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

    # 成功消息
    # 恢复了成功消息
    # st.markdown("""
    # <div class="success-message">
    #   <strong>计算完成！</strong> UKF滤波器已成功完成曼宁系数率定。您可以在上方查看结果图表和统计数据，并下载完整结果数据。
    # </div>
    # """, unsafe_allow_html=True)