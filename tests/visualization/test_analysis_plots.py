import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.analysis_plots import AnalysisPlotter
from typing import List, Dict, Tuple

@pytest.fixture
def test_convergence_times() -> List[int]:
    """创建测试用收敛时间数据"""
    return [100, 200, 500, 1000, 2000, 5000, 10000]

@pytest.fixture
def test_belief_history() -> List[List[float]]:
    """创建测试用信念演化历史"""
    time_steps = 10
    n_agents = 5
    return [
        [0.2 + 0.1*t + 0.1*i for i in range(n_agents)]
        for t in range(time_steps)
    ]

@pytest.fixture
def test_metrics_history() -> List[Dict[str, float]]:
    """创建测试用网络指标历史"""
    return [
        {
            "mean_degree": 4.0 + 0.1*t,
            "clustering": 0.3 + 0.05*t,
            "path_length": 2.0 - 0.02*t
        }
        for t in range(5)
    ]

@pytest.fixture
def test_convergence_data() -> Dict[str, List[int]]:
    """创建测试用收敛结果数据"""
    return {
        "contribution": [50, 40, 30, 20],
        "defection": [30, 40, 60, 75],
        "not_converge": [20, 20, 10, 5]
    }

@pytest.fixture
def test_catastrophe_ratios() -> List[Tuple[float, float]]:
    """创建测试用灾难原理比率数据"""
    return [
        (0.848, 0.824),
        (0.896, 0.900),
        (0.196, 0.192),
        (0.020, 0.028)
    ]

@pytest.fixture
def plotter() -> AnalysisPlotter:
    """创建分析绘图器实例"""
    return AnalysisPlotter()

def test_init():
    """测试初始化"""
    plotter = AnalysisPlotter(style='seaborn')
    assert isinstance(plotter, AnalysisPlotter)

def test_plot_convergence_tail_probability(plotter, test_convergence_times):
    """测试收敛时间尾概率分布图绘制"""
    # 基本绘制
    ax = plotter.plot_convergence_tail_probability(test_convergence_times)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 带标签的绘制
    ax = plotter.plot_convergence_tail_probability(
        convergence_times=test_convergence_times,
        labels=['Test'],
        log_scale=True
    )
    assert len(ax.get_legend().get_texts()) > 0
    plt.close()
    
    # 线性坐标轴
    ax = plotter.plot_convergence_tail_probability(
        convergence_times=test_convergence_times,
        log_scale=False
    )
    assert ax.get_xscale() == 'linear'
    assert ax.get_yscale() == 'linear'
    plt.close()

def test_plot_belief_evolution_heatmap(plotter, test_belief_history):
    """测试信念演化热力图绘制"""
    # 基本绘制
    ax = plotter.plot_belief_evolution_heatmap(test_belief_history)
    assert isinstance(ax, plt.Axes)
    
    # 验证轴标签
    assert ax.get_xlabel() == "Time Step"
    assert ax.get_ylabel() == "Agent ID"
    plt.close()
    
    # 自定义标题
    ax = plotter.plot_belief_evolution_heatmap(
        belief_history=test_belief_history,
        title="Custom Title"
    )
    assert ax.get_title() == "Custom Title"
    plt.close()

def test_plot_network_metrics(plotter, test_metrics_history):
    """测试网络指标变化图绘制"""
    metric_names = ["mean_degree", "clustering", "path_length"]
    
    # 基本绘制
    ax = plotter.plot_network_metrics(
        metrics_history=test_metrics_history,
        metric_names=metric_names
    )
    assert isinstance(ax, plt.Axes)
    
    # 验证图例
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(legend_texts) == set(metric_names)
    plt.close()

def test_plot_convergence_analysis(plotter, test_convergence_data):
    """测试收敛分析图绘制"""
    network_params = [0.15, 0.20, 0.25, 0.30]
    
    # 基本绘制
    fig, ax = plotter.plot_convergence_analysis(
        convergence_data=test_convergence_data,
        network_params=network_params
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    
    # 验证x轴刻度
    assert len(ax.get_xticks()) == len(network_params)
    plt.close()

def test_plot_catastrophe_ratio(plotter, test_catastrophe_ratios):
    """测试灾难原理比率图绘制"""
    network_params = [0.15, 0.20, 0.25, 0.30]
    
    # 基本绘制
    ax = plotter.plot_catastrophe_ratio(
        ratios=test_catastrophe_ratios,
        network_params=network_params
    )
    assert isinstance(ax, plt.Axes)
    
    # 验证图例
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "No Replacement" in legend_texts
    assert "With Replacement" in legend_texts
    plt.close()

def test_edge_cases(plotter):
    """测试边界情况"""
    # 空数据
    empty_times = []
    empty_history = []
    empty_metrics = []
    
    ax = plotter.plot_convergence_tail_probability(empty_times)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    ax = plotter.plot_belief_evolution_heatmap(empty_history)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    ax = plotter.plot_network_metrics(empty_metrics, [])
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_numerical_stability(plotter):
    """测试数值稳定性"""
    # 极端值测试
    extreme_times = [1, int(1e6), int(1e9)]
    ax = plotter.plot_convergence_tail_probability(extreme_times)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 大规模数据测试
    large_history = np.random.random((100, 50))  # 100个时间步，50个智能体
    ax = plotter.plot_belief_evolution_heatmap(large_history)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_custom_axes(plotter, test_convergence_times, test_belief_history):
    """测试自定义轴对象"""
    # 创建自定义轴对象
    fig, ax = plt.subplots()
    
    # 在自定义轴上绘制
    ax = plotter.plot_convergence_tail_probability(
        convergence_times=test_convergence_times,
        ax=ax
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1 = plotter.plot_belief_evolution_heatmap(
        belief_history=test_belief_history,
        ax=ax1
    )
    ax2 = plotter.plot_convergence_tail_probability(
        convergence_times=test_convergence_times,
        ax=ax2
    )
    assert isinstance(ax1, plt.Axes)
    assert isinstance(ax2, plt.Axes)
    plt.close()