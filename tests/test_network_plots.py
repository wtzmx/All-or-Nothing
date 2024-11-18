import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.network_plots import NetworkVisualizer
from typing import Dict, Set, List, Tuple

@pytest.fixture
def test_network() -> Dict[int, Set[int]]:
    """创建测试用网络"""
    return {
        0: {1, 2},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {1, 2}
    }

@pytest.fixture
def test_beliefs() -> List[float]:
    """创建测试用信念值"""
    return [0.2, 0.4, 0.6, 0.8]

@pytest.fixture
def test_positions() -> List[Tuple[float, float]]:
    """创建测试用节点位置"""
    return [(0, 0), (1, 0), (1, 1), (0, 1)]

@pytest.fixture
def visualizer() -> NetworkVisualizer:
    """创建可视化器实例"""
    return NetworkVisualizer()

def test_init():
    """测试初始化"""
    vis = NetworkVisualizer(figsize=(12, 12))
    assert vis.figsize == (12, 12)
    assert vis.belief_cmap is not None

def test_plot_network(visualizer, test_network, test_beliefs, test_positions):
    """测试网络结构绘制"""
    # 基本绘制
    ax = visualizer.plot_network(test_network)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 带颜色和位置的绘制
    ax = visualizer.plot_network(
        adjacency=test_network,
        node_positions=test_positions,
        node_colors=test_beliefs,
        show_labels=True
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 自定义节点大小
    ax = visualizer.plot_network(
        adjacency=test_network,
        node_sizes=[200, 300, 400, 500]
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_belief_distribution(visualizer, test_beliefs):
    """测试信念分布绘制"""
    # 基本绘制
    ax = visualizer.plot_belief_distribution(test_beliefs)
    assert isinstance(ax, plt.Axes)
    
    # 验证轴标签
    assert ax.get_xlabel() == "Belief"
    assert ax.get_ylabel() == "Count"
    plt.close()
    
    # 自定义标题
    ax = visualizer.plot_belief_distribution(
        beliefs=test_beliefs,
        title="Custom Title"
    )
    assert ax.get_title() == "Custom Title"
    plt.close()

def test_plot_network_state(visualizer, test_network, test_beliefs, test_positions):
    """测试网络状态完整视图绘制"""
    # 基本绘制
    fig, (ax1, ax2) = visualizer.plot_network_state(
        adjacency=test_network,
        beliefs=test_beliefs
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax1, plt.Axes)
    assert isinstance(ax2, plt.Axes)
    plt.close()
    
    # 带位置信息的绘制
    fig, (ax1, ax2) = visualizer.plot_network_state(
        adjacency=test_network,
        beliefs=test_beliefs,
        node_positions=test_positions,
        title="Custom Title"
    )
    assert "Custom Title" in ax1.get_title()
    assert "Custom Title" in ax2.get_title()
    plt.close()

def test_plot_belief_evolution(visualizer):
    """测试信念演化过程绘制"""
    # 创建测试用演化历史
    history = [
        [0.2, 0.4, 0.6, 0.8],
        [0.3, 0.4, 0.5, 0.7],
        [0.4, 0.4, 0.4, 0.6],
        [0.5, 0.5, 0.5, 0.5]
    ]
    
    # 基本绘制
    ax = visualizer.plot_belief_evolution(history)
    assert isinstance(ax, plt.Axes)
    
    # 验证轴标签和范围
    assert ax.get_xlabel() == "Time Step"
    assert ax.get_ylabel() == "Belief"
    ymin, ymax = ax.get_ylim()
    assert ymin < 0 and ymax > 1  # 确保y轴范围合适
    plt.close()
    
    # 验证图例
    ax = visualizer.plot_belief_evolution(
        belief_history=history,
        title="Custom Title"
    )
    assert len(ax.get_legend().get_texts()) > 0  # 确保有图例
    assert ax.get_title() == "Custom Title"
    plt.close()

def test_edge_cases(visualizer):
    """测试边界情况"""
    # 空网络
    empty_network = {}
    empty_beliefs = []
    
    ax = visualizer.plot_network(empty_network)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    ax = visualizer.plot_belief_distribution(empty_beliefs)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 单节点网络
    single_node = {0: set()}
    single_belief = [0.5]
    
    ax = visualizer.plot_network(single_node)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    ax = visualizer.plot_belief_distribution(single_belief)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_numerical_stability(visualizer):
    """测试数值稳定性"""
    # 极端信念值
    extreme_beliefs = [0.0, 1.0, 0.5, 0.9999, 0.0001]
    ax = visualizer.plot_belief_distribution(extreme_beliefs)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 大规模网络
    large_network = {i: {(i+1)%100} for i in range(100)}
    large_beliefs = np.random.random(100)
    
    fig, (ax1, ax2) = visualizer.plot_network_state(
        adjacency=large_network,
        beliefs=large_beliefs
    )
    assert isinstance(fig, plt.Figure)
    plt.close()

def test_custom_axes(visualizer, test_network, test_beliefs):
    """测试自定义轴对象"""
    # 创建自定义轴对象
    fig, ax = plt.subplots()
    
    # 在自定义轴上绘制
    ax = visualizer.plot_network(
        adjacency=test_network,
        ax=ax
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1 = visualizer.plot_network(
        adjacency=test_network,
        ax=ax1
    )
    ax2 = visualizer.plot_belief_distribution(
        beliefs=test_beliefs,
        ax=ax2
    )
    assert isinstance(ax1, plt.Axes)
    assert isinstance(ax2, plt.Axes)
    plt.close()