import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========================
#设置图示的字体
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
 
# 设置字体路径（以 Windows 系统为例，使用微软雅黑）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
 
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('示例')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.show()
# ========================



st.set_page_config(page_title="通用分段线性化工具", layout="wide")

st.title("🔧 分段线性化")

# ========================
# 新增控件：站点和快照模式选择（单个站点选择）
# ========================
st.sidebar.subheader("📍 闸站与快照模式设置")

# 大渡河闸站数据
stations = [
    "龚嘴水库", "铜街子电站", "沙坪枢纽", "安谷电站", 
    "犍为航电枢纽", "沐溪河口", "五通桥", "乐山港", "长江口"
]

selected_station = st.sidebar.selectbox("选择闸站", options=stations)

# 快照模式下拉框，显示中文和英文
snapshot_mode_options = {
    "当前时刻 (latest)": "latest",
    "历史区段 (history)": "history"
}

snapshot_mode = st.sidebar.selectbox(
    "快照模式", 
    options=list(snapshot_mode_options.keys()),
    format_func=lambda x: x  # 直接显示键值（包含中英文）
)

# 获取实际的模式值
selected_mode = snapshot_mode_options[snapshot_mode]

# 当选择历史区段时，显示开始时间和结束时间选择器
if selected_mode == "history":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.sidebar.date_input("开始时间")
    with col2:
        end_time = st.sidebar.date_input("结束时间")

# ========================
# 函数维度选择
# ========================
dim = st.sidebar.radio("选择函数维度", ["一元函数 y=f(x)", "二元函数 z=f(x, y)"])

# ========================
# 选择对象（根据函数维度设置不同的选择限制）
# ========================
object_options = ["流量", "水位", "闸门开度", "水头"]

# 根据函数维度设置最大选择数量和提示信息
if dim == "一元函数 y=f(x)":
    max_selections = 2
    hint_text = "💡 一元函数：请选择2个对象（1个自变量，1个因变量）"
else:
    max_selections = 3
    hint_text = "💡 二元函数：请选择3个对象（2个自变量，1个因变量）"

st.sidebar.markdown(hint_text)
selected_objects = st.sidebar.multiselect("选择对象", options=object_options, max_selections=max_selections)

# 检查选择数量是否符合要求
if dim == "一元函数 y=f(x)" and len(selected_objects) != 2 and len(selected_objects) > 0:
    st.sidebar.warning(f"一元函数需要选择2个对象，当前选择了{len(selected_objects)}个")
elif dim == "二元函数 z=f(x, y)" and len(selected_objects) != 3 and len(selected_objects) > 0:
    st.sidebar.warning(f"二元函数需要选择3个对象，当前选择了{len(selected_objects)}个")

# =============================================
# 一元函数处理
# =============================================
if dim == "一元函数 y=f(x)":
    st.header("📈 一元函数分段线性化")

    # 只有当选择了正确数量的对象时才显示处理界面
    if len(selected_objects) == 2:
        st.subheader(f"📊 基于 {selected_objects[0]} → {selected_objects[1]} 的一元函数分段线性化")
        
        #expr = st.text_input("输入一元函数表达式（变量使用 x）：", value="0.1*x**2 + 3*x + 5")   #隐藏这个公式
        expr = "0.1*x**2 + 3*x + 5"  # 默认表达式，不显示输入框

        x_min, x_max = st.slider("选择 x 的范围", 0.0, 100.0, (0.0, 50.0), key="x_range_slider")

        num_points = st.number_input("采样点数", min_value=10, max_value=1000, value=200)
        num_segments = st.number_input("分段数量", min_value=2, max_value=50, value=5)

        if st.button("生成并线性化"):
            x_vals = np.linspace(x_min, x_max, int(num_points))
            y_vals = np.array([eval(expr.replace("x", str(x))) for x in x_vals])

            segment_size = len(x_vals) // num_segments
            y_fit = np.zeros_like(y_vals)

            st.subheader("🔍 分段线性回归结果")
            st.markdown(f"**函数关系**: {selected_objects[1]} = f({selected_objects[0]})")
            
            for i in range(num_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < num_segments - 1 else len(x_vals)
                x_seg = x_vals[start:end]
                y_seg = y_vals[start:end]

                A = np.vstack([x_seg, np.ones(len(x_seg))]).T
                m, c = np.linalg.lstsq(A, y_seg, rcond=None)[0]
                y_fit[start:end] = m * x_seg + c

                st.markdown(f"段 {i+1}: ${selected_objects[1]} = {m:.3f} × {selected_objects[0]} + {c:.3f}$")

            # 误差评估
            mae = mean_absolute_error(y_vals, y_fit)
            rmse = np.sqrt(mean_squared_error(y_vals, y_fit))
            st.markdown(f"**误差评估** - MAE: {mae:.3f}, RMSE: {rmse:.3f}")

            # 图像展示
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label="原始函数", color='blue')
            ax.plot(x_vals, y_fit, label="线性化结果", linestyle='--', color='red')
            ax.set_xlabel(selected_objects[0])
            ax.set_ylabel(selected_objects[1])
            ax.set_title(f"{selected_objects[1]} 与 {selected_objects[0]} 的分段线性化对比")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("请先在侧边栏中选择2个对象以进行一元函数分析")

# =============================================
# 二元函数处理
# =============================================
else:
    st.header("📊 二元函数分段线性化")

    # 只有当选择了正确数量的对象时才显示处理界面
    if len(selected_objects) == 3:
        st.subheader(f"📊 基于 {selected_objects[0]}, {selected_objects[1]} → {selected_objects[2]} 的二元函数分段线性化")
        
        #expr2 = st.text_input("输入二元函数表达式（变量使用 x, y）：", value="np.sin(x/5)*np.cos(y/5)*10")    #隐藏这个公式
        expr2 = "np.sin(x/5)*np.cos(y/5)*10"  # 内置默认表达式，不显示输入框

        x_range = st.slider("x 轴范围", 0.0, 100.0, (0.0, 50.0))
        y_range = st.slider("y 轴范围", 0.0, 100.0, (0.0, 50.0))
        resolution = st.slider("网格分辨率", 10, 200, 100, step=10)
        seg_x = st.slider("x 分段数", 1, 20, 5)
        seg_y = st.slider("y 分段数", 1, 20, 5)

        if st.button("生成并线性化"):
            x_vals = np.linspace(x_range[0], x_range[1], resolution)
            y_vals = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = eval(expr2, {"np": np}, {"x": X, "y": Y})

            Z_fit = np.zeros_like(Z)
            dx = resolution // seg_x
            dy = resolution // seg_y

            st.subheader("🔍 分段线性系数")   #分段线性系数（均为平面 z=ax+by+c）
            st.markdown(f"**函数关系**: {selected_objects[2]} = f({selected_objects[0]}, {selected_objects[1]})")
            
            coeff_data = []
            for i in range(seg_x):
                for j in range(seg_y):
                    xi = slice(i * dx, (i + 1) * dx)
                    yj = slice(j * dy, (j + 1) * dy)

                    x_block = X[yj, xi].flatten()
                    y_block = Y[yj, xi].flatten()
                    z_block = Z[yj, xi].flatten()

                    A = np.c_[x_block, y_block, np.ones_like(x_block)]
                    coef, *_ = np.linalg.lstsq(A, z_block, rcond=None)
                    z_fit_block = A @ coef
                    Z_fit[yj, xi] = z_fit_block.reshape(dy, dx)
                    
                    # 保存系数数据
                    coeff_data.append({
                        "段号": f"({i+1},{j+1})",
                        f"{selected_objects[0]}系数(a)": f"{coef[0]:.3f}",
                        f"{selected_objects[1]}系数(b)": f"{coef[1]:.3f}",
                        "常数项(c)": f"{coef[2]:.3f}"
                    })

            # 显示系数表格
            coeff_df = pd.DataFrame(coeff_data)
            st.dataframe(coeff_df, use_container_width=True)

            # 误差评估图
            diff = np.abs(Z - Z_fit)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("✅ 原始函数热力图")
                fig1, ax1 = plt.subplots()
                sns.heatmap(Z, ax=ax1, cmap="viridis", cbar=True)
                ax1.set_title(f"{selected_objects[2]} 原始分布")
                st.pyplot(fig1)

            with col2:
                st.markdown("📉 线性拟合误差热力图")
                fig2, ax2 = plt.subplots()
                sns.heatmap(diff, ax=ax2, cmap="Reds", cbar=True)
                ax2.set_title("拟合误差分布")
                st.pyplot(fig2)

            st.markdown(f"**误差评估** - MAE: {mean_absolute_error(Z.flatten(), Z_fit.flatten()):.3f} | RMSE: {np.sqrt(mean_squared_error(Z.flatten(), Z_fit.flatten())):.3f}")
    else:
        st.info("请先在侧边栏中选择3个对象以进行二元函数分析")