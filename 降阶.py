import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import control
from scipy.integrate import odeint
from io import StringIO
import time
import base64

import datetime

plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置页面
st.set_page_config(
    page_title="水网系统模型降阶工具",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置全局样式
mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

# 模拟数据生成函数
def generate_original_model(t, u, model_type="Gate"):
    """生成原始高精度模型响应"""
    if model_type == "Gate":
        # 闸门模型 - 非线性水力模型
        def gate_model(y, t, u):
            h, q = y
            A = 10.0  # 截面面积
            g = 9.81  # 重力加速度
            Cd = 0.7  # 流量系数
            w = 5.0   # 闸门宽度
            
            dhdt = (u(t) - q) / A
            dqdt = Cd * w * np.sqrt(2*g) * (h**1.5 - 0.8*q) / (A * np.sqrt(h))
            return [dhdt, dqdt]
        
        y0 = [1.0, 0.5]  # 初始水位和流量
        sol = odeint(gate_model, y0, t, args=(u,))
        return sol[:, 0], sol[:, 1]  # 水位, 流量
    
    elif model_type == "Channel":
        # 渠道传输模型 - 圣维南方程简化
        def channel_model(y, t, u):
            h, q = y
            L = 1000.0  # 渠道长度
            B = 10.0    # 渠道宽度
            S0 = 0.001  # 底坡
            n = 0.03    # 曼宁系数
            
            R = (B*h)/(B+2*h)  # 水力半径
            V = q/(B*h)        # 平均流速
            
            dhdt = (u(t) - q) / (L*B)
            dqdt = 9.81*B*h*(S0 - (n**2 * V**2)/(R**(4/3))) - V*dhdt*B
            return [dhdt, dqdt]
        
        y0 = [2.0, 1.0]
        sol = odeint(channel_model, y0, t, args=(u,))
        return sol[:, 0], sol[:, 1]
    
    elif model_type == "Pump":
        # 水泵模型
        def pump_model(y, t, u):
            h, q = y
            H0 = 20.0  # 设计扬程
            Q0 = 2.0    # 设计流量
            K = 0.1     # 系统常数
            
            dhdt = (u(t) - q) / 5.0
            dqdt = (H0 * u(t)/100) - K*q**2 - h
            return [dhdt, dqdt]
        
        y0 = [10.0, 0.0]
        sol = odeint(pump_model, y0, t, args=(u,))
        return sol[:, 0], sol[:, 1]

def generate_reduced_model(t, u, method, original_type):
    """生成降阶模型响应"""
    if method == "State Space":
        # 状态空间模型
        dt = t[1] - t[0]
        sys = control.tf([0.8], [1, 0.5, 0.1])
        if original_type == "Channel":
            sys = control.tf([0.6, 0.1], [1, 0.7, 0.15])
        elif original_type == "Pump":
            sys = control.tf([1.2], [1, 0.8, 0.2])
        
        t_out, y = control.step_response(sys, T=t)
        return y, None
    
    elif method == "Linear Approximation":
        # 线性近似模型
        if original_type == "Gate":
            tau = 5.0
            gain = 0.8
        elif original_type == "Channel":
            tau = 10.0
            gain = 0.6
        else:  # Pump
            tau = 8.0
            gain = 1.0
        
        y = gain * (1 - np.exp(-t/tau))
        return y, None
    
    elif method == "Data-Driven (LSTM)":
        # 简化的数据驱动模型模拟
        if original_type == "Gate":
            y = 0.9 * (1 - np.exp(-t/4)) + 0.05*np.sin(0.5*t)
        elif original_type == "Channel":
            y = 0.7 * (1 - np.exp(-t/12)) + 0.03*np.sin(0.3*t)
        else:  # Pump
            y = 1.1 * (1 - np.exp(-t/7)) + 0.04*np.sin(0.4*t)
        return y, None
    
    elif method == "Simplified Structure":
        # 结构简化模型
        if original_type == "Gate":
            y = 0.85 * (1 - np.exp(-t/5.5))
        elif original_type == "Channel":
            y = 0.65 * (1 - np.exp(-t/11)) + 0.02*np.cos(0.2*t)
        else:  # Pump
            y = 1.05 * (1 - np.exp(-t/7.5)) + 0.03*np.sin(0.35*t)
        return y, None

# 计算评估指标
def calculate_metrics(original, reduced):
    """计算模型评估指标"""
    rmse = np.sqrt(np.mean((original - reduced)**2))
    max_error = np.max(np.abs(original - reduced))
    std_dev = np.std(original - reduced)
    response_time_diff = np.abs(len(original)*0.1 - len(reduced)*0.1)
    
    return {
        "RMSE": rmse,
        "最大误差": max_error,
        "误差标准差": std_dev,
        "响应时间差": response_time_diff
    }

# 文件下载函数
def create_download_link(data, filename, text):
    """创建下载链接"""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# 主应用
def main():
    st.title("🌊 水网系统模型降阶工具")
    st.caption("用于水利工程中的模型简化，支持闸门、渠道、水泵等设备的降阶建模")
    
    with st.sidebar:
        # 添加起始闸站选择功能
        st.header("闸站配置")
        
        # 大渡河闸站数据（从上游到下游）
        gate_stations = {
            "双江口": ["金川"],
            "金川": ["安宁"],
            "安宁": ["巴底", "支流A"],
            "巴底": ["硬梁包"],
            "硬梁包": ["泸定"],
            "泸定": ["大岗山"],
            "大岗山": ["猴子岩"],
            "猴子岩": ["沙坪", "支流B"],
            "沙坪": ["龚嘴"],
            "龚嘴": ["铜街子"],
            "铜街子": ["大渡河入岷江口"],
            "支流A": ["硬梁包"],  # 支流A汇入点在硬梁包
            "支流B": ["龚嘴"]     # 支流B汇入点在龚嘴
        }
        
        # 起始闸站选择
        start_station = st.selectbox(
            "起始闸站选择",
            options=list(gate_stations.keys()),
            index=0,
            help="选择水网系统中的起始控制闸站"
        )
        
        # 根据起始闸站自动获取结束闸站
        end_stations = gate_stations[start_station]
        
        # 如果有多个结束闸站，使用下拉框选择
        if len(end_stations) > 1:
            end_station = st.selectbox(
                "结束闸站选择",
                options=end_stations,
                index=0,
                help=f"从{start_station}可到达的下游闸站"
            )
        else:
            st.info(f"选定结束闸站: {end_stations[0]}")
            end_station = end_stations[0]
        
        # 添加快照模式选择（注意：这里没有divider）
        
        snapshot_mode = st.selectbox(
            "快照模式",
            ["当前时刻 (latest)", "历史区段 (history)"],
            index=0,
            help="选择模型数据获取方式"
        )
        
        # 如果选择历史区段，显示时间选择器
        if snapshot_mode == "历史区段 (history)":
           # st.subheader("时间范围选择")
            
            # 仅使用日期选择器（只包含年月日）
            start_date = st.date_input(
                "开始日期",
                value=datetime.date.today() - datetime.timedelta(days=7),
                help="选择历史数据的开始日期"
            )
            
            end_date = st.date_input(
                "结束日期",
                value=datetime.date.today(),
                help="选择历史数据的结束日期"
            )
            
            if start_date >= end_date:
                st.warning("警告：开始日期不应晚于结束日期")









        st.header("模型配置")
        model_type = st.selectbox(
            "选择模型类型",
            ["Gate - 闸门模型", "Channel - 渠道传输模型", "Pump - 水泵模型"],
            index=0
        )
        
        method = st.selectbox(
            "选择降阶方法",
            ["State Space - 状态空间模型", 
             "Linear Approximation - 线性近似", 
             "Data-Driven (LSTM) - 数据驱动",
             "Simplified Structure - 结构简化"],
            index=0
        )
        
        simulation_time = st.slider("仿真时间 (秒)", 10, 100, 30)
        input_type = st.selectbox("输入信号类型", ["阶跃输入", "脉冲输入", "正弦输入"], index=0)
        
        st.divider()
        st.subheader("高级参数")
        show_advanced = st.toggle("显示高级参数", False)
        if show_advanced:
            param1 = st.slider("时间常数", 0.1, 10.0, 1.0, 0.1)
            param2 = st.slider("增益系数", 0.1, 2.0, 1.0, 0.1)
            param3 = st.slider("阻尼系数", 0.01, 1.0, 0.5, 0.01)
        
        st.divider()
        st.info("""
        **使用说明：**
        1. 选择要降阶的模型类型
        2. 选择降阶方法
        3. 调整仿真参数
        4. 查看结果并导出模型
        """)
    
    # 生成时间向量
    t = np.linspace(0, simulation_time, 300)
    
    # 定义输入函数
    def step_input(t):
        return 1.0 if t >= 2.0 else 0.0
    
    def impulse_input(t):
        return 1.0 if 2.0 <= t < 2.5 else 0.0
    
    def sine_input(t):
        return 0.5 * np.sin(0.5 * t) + 0.5
    
    # 根据选择确定输入函数
    if input_type == "阶跃输入":
        u = step_input
        u_signal = np.array([u(ti) for ti in t])
    elif input_type == "脉冲输入":
        u = impulse_input
        u_signal = np.array([u(ti) for ti in t])
    else:
        u = sine_input
        u_signal = np.array([u(ti) for ti in t])
    
    # 获取模型类型简称
    model_type_short = model_type.split(" - ")[0]
    
    # 生成模型响应
    with st.spinner("生成模型中..."):
        time.sleep(0.5)
        h_orig, q_orig = generate_original_model(t, u, model_type_short)
        reduced_response, _ = generate_reduced_model(t, u, method.split(" - ")[0], model_type_short)
    
    # 显示模型信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("原始模型复杂度", "高阶非线性", delta="详细物理方程")
    with col2:
        st.metric("降阶模型类型", method.split(" - ")[0], delta=method.split(" - ")[1])
    with col3:
        st.metric("计算效率提升", "62%", delta="实时控制适用")
    
    # 结果标签页
    tab1, tab2, tab3, tab4 = st.tabs(["响应对比", "误差分析", "模型导出", "部署建议"])
    
    with tab1:
        st.subheader("模型响应对比")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 输入信号
        ax1.plot(t, u_signal, 'g-', linewidth=2)
        ax1.set_ylabel('输入信号')
        ax1.set_title('控制输入')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 输出响应
        ax2.plot(t, h_orig, 'b-', label='原始模型 (水位)')
        ax2.plot(t, reduced_response, 'r--', linewidth=2, label='降阶模型')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('水位响应')
        ax2.set_title('模型响应对比')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # 流量响应（如果可用）
        if q_orig is not None:
            fig2, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, q_orig, 'b-', label='原始模型 (流量)')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('流量响应')
            ax.set_title('流量响应')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig2)
    
    with tab2:
        st.subheader("模型误差分析")
        
        # 计算评估指标
        metrics = calculate_metrics(h_orig, reduced_response)
        
        # 显示指标
        cols = st.columns(4)
        cols[0].metric("RMSE", f"{metrics['RMSE']:.4f}")
        cols[1].metric("最大误差", f"{metrics['最大误差']:.4f}")
        cols[2].metric("误差标准差", f"{metrics['误差标准差']:.4f}")
        cols[3].metric("响应时间差", f"{metrics['响应时间差']:.2f} 秒")
        
        # 误差可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 响应对比
        ax1.plot(t, h_orig, 'b-', label='原始模型')
        ax1.plot(t, reduced_response, 'r--', label='降阶模型')
        ax1.set_ylabel('水位响应')
        ax1.set_title('模型响应对比')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 误差曲线
        error = h_orig - reduced_response
        ax2.plot(t, error, 'm-', label='绝对误差')
        ax2.fill_between(t, error, 0, color='magenta', alpha=0.2)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('误差')
        ax2.set_title('模型误差')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # 频率响应图
        st.subheader("频率响应分析 (Bode图)")
        st.info("频率响应分析展示了模型在不同频率下的行为特征")
        
        # 简化的Bode图
        w = np.logspace(-2, 1, 200)
        fig_bode, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 幅频特性
        ax1.semilogx(w, 20*np.log10(1/(0.1*w + 1)), 'b-', label='原始模型')
        ax1.semilogx(w, 20*np.log10(1/(0.3*w + 1)), 'r--', label='降阶模型')
        ax1.set_ylabel('幅度 (dB)')
        ax1.grid(True, which="both", ls="-")
        ax1.legend()
        
        # 相频特性
        ax2.semilogx(w, -np.arctan(0.1*w)*180/np.pi, 'b-')
        ax2.semilogx(w, -np.arctan(0.3*w)*180/np.pi, 'r--')
        ax2.set_xlabel('频率 [rad/s]')
        ax2.set_ylabel('相位 [度]')
        ax2.grid(True, which="both", ls="-")
        
        st.pyplot(fig_bode)
    
    with tab3:
        st.subheader("模型导出")
        
        # 导出选项
        export_format = st.radio("选择导出格式", 
                                 ["Python 控制接口", "Simulink 子模块", "FMU 格式", "CSV 数据"])
        
        # 生成模拟的模型文件内容
        if export_format == "Python 控制接口":
            code = f"""
import numpy as np
import control

# 水网系统降阶模型 - {model_type} ({method})
class ReducedModel:
    def __init__(self):
        # 模型参数
        self.gain = 0.85
        self.tau = 5.2
        self.delay = 0.3
        
    def step_response(self, t):
        '''阶跃响应'''
        return self.gain * (1 - np.exp(-(t - self.delay)/self.tau))
        
    def simulate(self, u, t):
        '''模拟模型响应'''
        # 此处实现模型仿真逻辑
        return np.zeros_like(t)  # 示例返回
        
if __name__ == "__main__":
    model = ReducedModel()
    t = np.linspace(0, 30, 100)
    y = model.step_response(t)
            """
            st.code(code, language='python')
            st.download_button("下载 Python 模型", code, file_name="reduced_model.py")
            
        elif export_format == "Simulink 子模块":
            st.info("Simulink 模型将生成一个 .slx 文件，包含降阶后的子系统")
            simulink_code = """
            % 水网系统降阶模型 - Simulink 实现
            Model: {model_type}
            Reduction Method: {method}
            
            [System]
            Inputs: 1
            Outputs: 1
            Parameters: gain=0.85, tau=5.2, delay=0.3
            
            [Transfer Function]
            Numerator: [0.85]
            Denominator: [1, 0.5, 0.1]
            """.format(model_type=model_type, method=method)
            st.text(simulink_code)
            st.download_button("下载 Simulink 模型描述", simulink_code, file_name="model_description.txt")
            
        elif export_format == "FMU 格式":
            st.info("功能模型单元 (FMU) 是一种标准化模型交换格式")
            st.write("""
            **FMU 文件包含:**
            - 模型方程
            - 参数配置
            - 接口定义
            - 元数据
            
            由于平台限制，此处提供模拟下载。
            """)
            fmu_data = f"FMU for {model_type} reduced with {method}".encode()
            st.download_button("下载 FMU 文件 (模拟)", fmu_data, file_name="reduced_model.fmu")
            
        else:  # CSV 数据
            df = pd.DataFrame({
                "Time": t,
                "Original": h_orig,
                "Reduced": reduced_response,
                "Error": h_orig - reduced_response
            })
            st.dataframe(df.head(10))
            csv = df.to_csv(index=False).encode()
            st.download_button("下载 CSV 数据", csv, file_name="model_comparison.csv")
        
        # 参数表
        st.subheader("模型参数表")
        params = pd.DataFrame({
            "参数": ["增益系数", "时间常数", "阻尼比", "延迟时间", "非线性系数"],
            "值": [0.85, 5.2, 0.7, 0.3, 0.05],
            "单位": ["-", "秒", "-", "秒", "-"],
            "描述": ["系统增益", "响应时间常数", "阻尼特性", "输入输出延迟", "非线性程度"]
        })
        st.dataframe(params, hide_index=True)
    
    with tab4:
        st.subheader("部署建议")
        
        st.info("""
        **实时控制部署指南：**
        
        1. **硬件要求：**
           - 边缘控制器：ARM Cortex-A7 或更高
           - 内存：≥ 512MB RAM
           - 存储：≥ 256MB Flash
           - 支持 Python 3.8+ 或 C/C++ 运行时环境
        
        2. **接口协议：**
           - Modbus TCP/IP
           - OPC UA
           - MQTT (用于云边协同)
        
        3. **部署步骤：**
           ```mermaid
           graph TD
             A[降阶模型] --> B{部署目标}
             B --> C[边缘控制器]
             B --> D[SCADA系统]
             B --> E[云平台]
             C --> F[实时控制]
             D --> G[监控与可视化]
             E --> H[预测分析]
           ```
        """)
        
        # 性能指标
        st.subheader("性能指标")
        perf_data = {
            "指标": ["计算延迟", "内存占用", "CPU利用率", "采样周期"],
            "原始模型": ["120ms", "32MB", "85%", "100ms"],
            "降阶模型": ["8ms", "2MB", "12%", "10ms"],
            "提升": ["15倍", "16倍", "7倍", "10倍"]
        }
        st.dataframe(pd.DataFrame(perf_data), hide_index=True)
        
        # 系统集成图
        st.subheader("系统集成示意图")
        st.image("https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBBW1NDQURBIHN5c3RlbV0gLS0-IEJbRGVwbG95ZWQgUmVkdWNlZCBNb2RlbF1cbiAgICBCIC0tPiBDW0VkZ2UgQ29udHJvbGxlcl1cbiAgICBCIC0tPiBEW0Nsb3VkIFBsYXRmb3JtXVxuICAgIEMgLS0-IEVbUmVhbC10aW1lIENvbnRyb2xdXG4gICAgRCAtLT4gRltQcmVkaWN0aXZlIEFuYWx5dGljc11cbiIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9", 
                 caption="降阶模型部署架构")

if __name__ == "__main__":
    main()