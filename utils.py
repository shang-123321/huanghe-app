"""
utils.py - 黄河污染溯源智能助手核心函数模块
包含数据预处理、全变分反演溯源、简报生成三个核心功能
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理函数
    输入：用户上传的原始CSV（DataFrame格式）
    处理：统一列名、时间格式化、缺失值前向填充、异常值替换
    返回：清洗后的DataFrame
    """
    df = df.copy()
    
    # 1. 统一列名：自动识别常见列名变体
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if '日期' in col_lower or '时间' in col_lower or 'date' in col_lower or 'time' in col_lower:
            column_mapping[col] = 'date'
        elif '断面' in col_lower or '站点' in col_lower or 'station' in col_lower or 'site' in col_lower:
            column_mapping[col] = 'station'
        elif '氨氮' in col_lower or 'nh3' in col_lower or 'nh3n' in col_lower or '浓度' in col_lower:
            column_mapping[col] = 'nh3n'
    df = df.rename(columns=column_mapping)
    
    # 2. 时间格式化
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise ValueError("未找到日期列，请确保数据包含日期/时间字段")
    
    # 3. 缺失值处理：前向填充 + 线性插值（修正为兼容新版pandas）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 使用 ffill() 和 bfill() 代替 fillna(method=...)
        df[col] = df[col].ffill().bfill()
        # 若仍有缺失，用均值填充
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # 4. 异常值处理：3σ原则替换
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 5. 确保断面列为字符串类型
    if 'station' in df.columns:
        df['station'] = df['station'].astype(str)
    
    return df


def build_forward_matrix(stations: list, alpha: float = 1.9) -> np.ndarray:
    """
    构建正演矩阵（分数阶反常扩散模型）
    模拟污染物从各断面到观测点的传递系数
    """
    n = len(stations)
    A = np.zeros((n, n))
    
    # 基于距离的分数阶核函数：K(r) ∝ r^(-alpha)
    # 假设断面等间距排列，距离为 |i-j|
    for i in range(n):
        for j in range(n):
            dist = abs(i - j) + 1  # 避免除零
            A[i, j] = dist ** (-alpha)
    
    # 行归一化
    row_sums = A.sum(axis=1, keepdims=True)
    A = A / row_sums
    
    return A


def tv_regularization(A: np.ndarray, obs: np.ndarray, lam: float = 0.01) -> dict:
    """
    全变分（TV）正则化反演求解器
    目标函数：||A*s - obs||^2 + lam * ||D*s||_1
    其中D为差分矩阵，促进解的分段平滑（保留尖锐边界）
    """
    n = A.shape[1]
    
    # 构造差分矩阵 D (n-1) x n
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = 1
        D[i, i+1] = -1
    
    def objective(s):
        residual = A @ s - obs
        tv_term = lam * np.sum(np.abs(D @ s))
        return 0.5 * np.sum(residual ** 2) + tv_term
    
    # 非负约束：源强不能为负
    bounds = [(0, None)] * n
    
    # 初始猜测：均匀分布
    s0 = np.ones(n) * np.mean(obs) / n
    
    # 优化求解
    result = minimize(objective, s0, method='L-BFGS-B', bounds=bounds)
    
    return {
        'source_strength': result.x,
        'converged': result.success,
        'message': result.message,
        'fun_value': result.fun
    }


def run_tv_inversion(obs_series: pd.Series, alpha: float = 1.9, lam: float = 0.01) -> dict:
    """
    全变分溯源引擎
    输入：指定时段的观测数据（Series，index为断面名，value为氨氮均值）
         分数阶阶数 alpha（默认1.9），正则化参数 lam（默认0.01）
    输出：字典，包含污染源位置（断面区间）、源强向量、收敛状态
    """
    stations = obs_series.index.tolist()
    obs_values = obs_series.values.astype(float)
    
    # 处理全零或无效观测
    if np.all(obs_values == 0) or np.any(np.isnan(obs_values)):
        return {
            'source_loc': stations,
            'source_strength': np.zeros(len(stations)),
            'converged': False,
            'message': '观测数据无效或全为零'
        }
    
    # 构建正演矩阵
    A = build_forward_matrix(stations, alpha)
    
    # TV反演求解
    result = tv_regularization(A, obs_values, lam)
    
    return {
        'source_loc': stations,
        'source_strength': result['source_strength'],
        'converged': result['converged'],
        'message': result['message'],
        'fun_value': result['fun_value']
    }


def generate_brief(result: dict, start_date, end_date, top_n: int = 3) -> str:
    """
    结果解释与简报生成
    根据源强排序，找出Top N嫌疑河段，生成决策简报文本
    """
    stations = result['source_loc']
    strengths = result['source_strength']
    
    # 按源强降序排序
    sorted_indices = np.argsort(strengths)[::-1]
    
    # 格式化日期
    if isinstance(start_date, (pd.Timestamp, datetime)):
        start_str = start_date.strftime('%Y-%m-%d')
    else:
        start_str = str(start_date)
    if isinstance(end_date, (pd.Timestamp, datetime)):
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        end_str = str(end_date)
    
    # 构建简报
    brief_lines = []
    brief_lines.append("=" * 50)
    brief_lines.append(f"     黄河下游氨氮污染溯源简报")
    brief_lines.append("=" * 50)
    brief_lines.append(f"溯源时段：{start_str} 至 {end_str}")
    brief_lines.append(f"模型参数：α = 1.9（分数阶阶数），λ = 0.01（正则化参数）")
    brief_lines.append(f"收敛状态：{'✅ 收敛' if result.get('converged', False) else '⚠️ 未完全收敛'}")
    brief_lines.append("")
    brief_lines.append(f"【嫌疑河段 Top {top_n}】")
    brief_lines.append("-" * 50)
    
    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i]
        loc = stations[idx]
        strength = strengths[idx]
        # 根据强度给出建议措施
        if strength > 0.5:
            suggestion = "🚨 紧急排查：建议立即安排执法人员现场取样"
        elif strength > 0.2:
            suggestion = "⚠️ 重点关注：建议24小时内完成现场核查"
        else:
            suggestion = "📋 常规关注：纳入日常巡查计划"
        brief_lines.append(f"  {i+1}. {loc}")
        brief_lines.append(f"     源强估算值：{strength:.4f}")
        brief_lines.append(f"     建议措施：{suggestion}")
        brief_lines.append("")
    
    # 综合建议
    if len(sorted_indices) > 0:
        top_loc = stations[sorted_indices[0]]
        brief_lines.append("-" * 50)
        brief_lines.append(f"📌 综合建议：优先排查 {top_loc} 河段沿岸工业排放口、")
        brief_lines.append("   生活污水直排口及农业面源污染，建议结合现场")
        brief_lines.append("   巡查与水质快速检测验证溯源结果。")
    
    brief_lines.append("=" * 50)
    brief_lines.append("报告生成时间：" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    brief_lines.append("=" * 50)
    
    return "\n".join(brief_lines)
