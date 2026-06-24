"""
app.py - 黄河污染溯源智能助手 Streamlit 网页应用
实现：数据上传 → 预处理 → 参数设置 → 一键溯源 → 结果展示 → 报告下载
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# 导入核心函数
from utils import preprocess_data, run_tv_inversion, generate_brief

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="黄河污染溯源智能助手",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 标题与简介 ====================
st.title("🌊 黄河下游氨氮污染精准溯源智能助手")
st.markdown("""
> 基于**分数阶建模**与**全变分正则化反演**技术，上传监测数据即可快速定位污染嫌疑河段。
> 无需编程背景，三步完成溯源：**上传数据 → 选择时段 → 一键溯源**。
""")

# ==================== 侧边栏：文件上传与参数设置 ====================
with st.sidebar:
    st.header("📤 数据上传")
    uploaded_file = st.file_uploader(
        "上传监测数据 CSV 文件",
        type=['csv'],
        help="要求包含：日期/时间列、断面/站点列、氨氮浓度列"
    )
    
    st.divider()
    st.header("⚙️ 溯源参数")
    
    # 参数说明
    st.caption("""
    **α (分数阶阶数)**：反映污染物扩散的记忆效应，黄河下游推荐 1.9  
    **λ (正则化参数)**：控制反演结果的平滑程度
    """)
    
    alpha = st.slider(
        "分数阶阶数 α",
        min_value=1.0,
        max_value=2.0,
        value=1.9,
        step=0.05,
        help="α 越接近 2 扩散越快，越接近 1 扩散越慢"
    )
    
    lam = st.number_input(
        "正则化参数 λ",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
    
    st.divider()
    st.caption("💡 提示：点击下方按钮开始溯源")
    run_button = st.button("🚀 开始溯源", use_container_width=True, type="primary")

# ==================== 主区域：数据加载与处理 ====================
# 初始化 session state
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'brief' not in st.session_state:
    st.session_state.brief = None

# 主区域布局：两列
col1, col2 = st.columns([2, 1])

with col1:
    # ---------- 页面1：数据上传与预览 ----------
    if uploaded_file is not None:
        try:
            # 读取原始数据
            raw_df = pd.read_csv(uploaded_file)
            
            # 执行预处理
            with st.spinner("🔄 正在清洗数据..."):
                df_clean = preprocess_data(raw_df)
                st.session_state.df_clean = df_clean
            
            st.success("✅ 数据预处理完成！")
            
            # 显示数据概览
            st.subheader("📊 数据预览")
            
            # 统计摘要
            col_meta, col_stats = st.columns(2)
            with col_meta:
                st.metric("断面/站点数", df_clean['station'].nunique())
                st.metric("数据记录数", len(df_clean))
            with col_stats:
                if 'nh3n' in df_clean.columns:
                    st.metric("氨氮均值 (mg/L)", f"{df_clean['nh3n'].mean():.3f}")
                    st.metric("氨氮最大值 (mg/L)", f"{df_clean['nh3n'].max():.3f}")
            
            # 数据表格
            st.dataframe(
                df_clean.head(20),
                use_container_width=True,
                height=300
            )
            
            # 时间范围信息
            if 'date' in df_clean.columns:
                date_min = df_clean['date'].min()
                date_max = df_clean['date'].max()
                st.info(f"📅 数据时间范围：{date_min.strftime('%Y-%m-%d')} 至 {date_max.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.error(f"❌ 数据读取失败：{str(e)}")
            st.stop()
    
    else:
        # 未上传文件时的占位提示
        st.info("👈 请在左侧侧边栏上传监测数据 CSV 文件")
        st.markdown("""
        **数据格式要求：**
        - 必须包含：日期列、断面/站点列、氨氮浓度列
        - 支持列名：`日期`/`date`/`时间`、`断面`/`station`/`站点`、`氨氮`/`nh3n`/`浓度`
        - 编码建议：UTF-8
        """)
        
        # 显示示例数据格式
        sample_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'station': ['断面A', '断面B', '断面A', '断面B'],
            'nh3n': [0.52, 0.38, 0.61, 0.42]
        })
        st.caption("📋 示例数据格式：")
        st.dataframe(sample_data, use_container_width=True)

with col2:
    # ---------- 页面2：时段选择与观测浓度图 ----------
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        st.subheader("📅 选择溯源时段")
        
        # 日期选择器
        if 'date' in df.columns:
            dates = df['date'].dt.date
            date_min = dates.min()
            date_max = dates.max()
            
            start_date = st.date_input(
                "起始日期",
                value=date_min,
                min_value=date_min,
                max_value=date_max
            )
            end_date = st.date_input(
                "结束日期",
                value=date_max,
                min_value=date_min,
                max_value=date_max
            )
            
            if start_date > end_date:
                st.error("⚠️ 起始日期不能晚于结束日期")
                st.stop()
            
            # 筛选时段数据
            mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
            period_df = df[mask]
            
            if len(period_df) == 0:
                st.warning("⚠️ 所选时段无数据，请调整日期范围")
            else:
                st.success(f"✅ 共 {len(period_df)} 条记录")
                
                # 绘制观测浓度折线图
                st.subheader("📈 各断面氨氮浓度趋势")
                
                if 'station' in period_df.columns and 'nh3n' in period_df.columns:
                    fig = px.line(
                        period_df,
                        x='date',
                        y='nh3n',
                        color='station',
                        title=f"氨氮浓度时序图 ({start_date} ~ {end_date})",
                        labels={'nh3n': '氨氮浓度 (mg/L)', 'date': '日期'}
                    )
                    fig.update_layout(
                        height=300,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 按断面聚合均值
                    obs_series = period_df.groupby('station')['nh3n'].mean()
                    
                    # 存储观测数据供溯源使用
                    st.session_state.obs_series = obs_series
                    st.session_state.start_date = start_date
                    st.session_state.end_date = end_date
                else:
                    st.warning("数据缺少 'station' 或 'nh3n' 列")
    else:
        st.info("请先上传数据")

# ==================== 页面3：溯源结果展示 ====================
st.divider()
st.header("🔍 溯源结果")

if run_button:
    if st.session_state.df_clean is None:
        st.warning("⚠️ 请先上传数据")
    elif 'obs_series' not in st.session_state or st.session_state.obs_series is None:
        st.warning("⚠️ 请先在右侧选择有效的溯源时段")
    else:
        with st.spinner("🧮 正在执行全变分反演溯源计算..."):
            try:
                obs_series = st.session_state.obs_series
                result = run_tv_inversion(obs_series, alpha=alpha, lam=lam)
                st.session_state.result = result
                
                # 生成简报
                brief = generate_brief(
                    result,
                    st.session_state.start_date,
                    st.session_state.end_date
                )
                st.session_state.brief = brief
                
                st.success("✅ 溯源计算完成！")
                
            except Exception as e:
                st.error(f"❌ 溯源计算失败：{str(e)}")
                st.stop()

# 显示结果
if st.session_state.result is not None:
    result = st.session_state.result
    stations = result['source_loc']
    strengths = result['source_strength']
    
    # 结果布局：两列
    res_col1, res_col2 = st.columns([3, 2])
    
    with res_col1:
        # 污染源强度空间分布图（柱状图）
        st.subheader("📊 污染源强度空间分布")
        
        # 创建 DataFrame 用于绘图
        df_result = pd.DataFrame({
            '河段': stations,
            '源强估算值': strengths
        })
        df_result = df_result.sort_values('源强估算值', ascending=False)
        
        # 设置颜色：高风险标红
        colors = ['#FF6B6B' if x > 0.2 else '#4ECDC4' for x in df_result['源强估算值']]
        
        fig = px.bar(
            df_result,
            x='河段',
            y='源强估算值',
            title='各河段污染源强度估算',
            labels={'源强估算值': '源强 (相对值)', '河段': '断面/河段'},
            color=df_result['源强估算值'],
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with res_col2:
        # 嫌疑河段排序表
        st.subheader("📋 嫌疑河段排序")
        
        df_table = pd.DataFrame({
            '排名': range(1, len(stations) + 1),
            '河段': stations,
            '源强': strengths,
            '风险等级': ['🔴 高' if s > 0.2 else '🟡 中' if s > 0.05 else '🟢 低' for s in strengths]
        })
        df_table = df_table.sort_values('源强', ascending=False)
        
        st.dataframe(
            df_table.head(10),
            use_container_width=True,
            hide_index=True,
            height=300
        )
    
    # 收敛状态
    if result.get('converged', False):
        st.success(f"✅ 反演收敛，目标函数值：{result.get('fun_value', 0):.6f}")
    else:
        st.warning(f"⚠️ 反演未完全收敛：{result.get('message', '未知原因')}")

# ==================== 页面4：报告下载 ====================
if st.session_state.brief is not None:
    st.divider()
    st.header("📄 溯源简报")
    
    brief_text = st.session_state.brief
    st.text_area("简报内容", brief_text, height=300)
    
    # 下载按钮
    st.download_button(
        label="📥 下载溯源报告 (TXT)",
        data=brief_text,
        file_name=f"溯源报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ==================== 页脚 ====================
st.divider()
st.caption("🌊 黄河下游氨氮污染精准溯源智能助手 | 基于分数阶建模 + 全变分反演")
