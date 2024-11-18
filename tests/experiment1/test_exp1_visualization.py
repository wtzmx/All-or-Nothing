import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')  # 在导入plt之前设置后端
import matplotlib.pyplot as plt
import shutil
from unittest.mock import Mock, patch

from experiments.experiment1.exp1_visualization import (
    ExperimentVisualizer, VisualizationConfig
)

@pytest.fixture
def test_analysis_dir(tmp_path):
    """创建测试分析结果目录"""
    analysis_dir = tmp_path / "test_analysis"
    analysis_dir.mkdir()
    
    # 创建测试数据
    test_results = {
        "tail_probabilities": {
            0.15: {
                "times": np.linspace(100, 10000, 50),
                "probabilities": np.linspace(1, 0, 50)
            },
            0.3: {
                "times": np.linspace(100, 8000, 50),
                "probabilities": np.linspace(1, 0, 50)
            }
        },
        "convergence_states": pd.DataFrame({
            "radius": [0.15, 0.3],
            "contribution_ratio": [0.3, 0.2],
            "defection_ratio": [0.6, 0.7],
            "not_converged_ratio": [0.1, 0.1],
            "contribution_ci_lower": [0.25, 0.15],
            "contribution_ci_upper": [0.35, 0.25],
            "defection_ci_lower": [0.55, 0.65],
            "defection_ci_upper": [0.65, 0.75],
            "not_converged_ci_lower": [0.08, 0.08],
            "not_converged_ci_upper": [0.12, 0.12]
        }),
        "network_states": {
            0.15: {
                "beliefs": np.random.uniform(0, 1, 50)
            },
            0.3: {
                "beliefs": np.random.uniform(0, 1, 50)
            }
        },
        "belief_histories": {
            0.15: np.random.uniform(0, 1, (100, 50)),
            0.3: np.random.uniform(0, 1, (100, 50))
        }
    }
    
    # 保存测试数据
    with open(analysis_dir / "analysis_results.pkl", 'wb') as f:
        pickle.dump(test_results, f)
        
    test_results["convergence_states"].to_csv(
        analysis_dir / "convergence_states.csv",
        index=False
    )
    
    return analysis_dir

@pytest.fixture
def visualizer(test_analysis_dir):
    """创建可视化器实例"""
    config = VisualizationConfig(
        figure_size=(8, 6),
        dpi=100,
        style="seaborn",
        save_format="png"
    )
    return ExperimentVisualizer(str(test_analysis_dir), config)

@pytest.fixture(autouse=True)
def setup_matplotlib():
    """自动设置matplotlib配置"""
    import seaborn as sns
    with sns.axes_style("darkgrid"):
        yield
    plt.close('all')

def test_initialization(test_analysis_dir):
    """测试可视化器初始化"""
    visualizer = ExperimentVisualizer(str(test_analysis_dir))
    assert visualizer.analysis_dir == Path(test_analysis_dir)
    assert visualizer.config is not None
    assert visualizer.logger is not None
    assert visualizer.analysis_plotter is not None
    assert visualizer.network_plotter is not None
    assert visualizer.results is not None

def test_load_analysis_results(visualizer):
    """测试分析结果加载"""
    results = visualizer.results
    assert "tail_probabilities" in results
    assert "convergence_states" in results
    assert "network_states" in results
    assert "belief_histories" in results
    
    # 检查数据格式
    assert isinstance(results["convergence_states"], pd.DataFrame)
    assert isinstance(results["tail_probabilities"], dict)
    assert isinstance(results["network_states"], dict)
    assert isinstance(results["belief_histories"], dict)

def test_plot_tail_probabilities(visualizer, tmp_path):
    """测试尾概率分布图绘制"""
    # 创建保存目录
    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    
    # 绘制图形
    import seaborn as sns
    with sns.axes_style("darkgrid"):
        visualizer.plot_tail_probabilities(save_dir)
    
    # 检查文件是否创建
    for radius in [0.15, 0.3]:
        assert (save_dir / f"tail_prob_r{radius:.2f}.png").exists()

def test_plot_network_states(visualizer, tmp_path):
    """测试网络状态图绘制"""
    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    
    # 绘制图形
    import seaborn as sns
    with sns.axes_style("darkgrid"):
        visualizer.plot_network_states(
            visualizer.results["network_states"],
            save_dir
        )
    
    # 检查文件是否创建
    for radius in [0.15, 0.3]:
        assert (save_dir / f"network_r{radius:.2f}.png").exists()

def test_plot_belief_evolution(visualizer, tmp_path):
    """测试信念演化图绘制"""
    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    
    # 绘制图形
    import seaborn as sns
    with sns.axes_style("darkgrid"):
        visualizer.plot_belief_evolution(
            visualizer.results["belief_histories"],
            save_dir
        )
    
    # 检查文件是否创建
    for radius in [0.15, 0.3]:
        assert (save_dir / f"belief_evolution_r{radius:.2f}.png").exists()

def test_generate_convergence_table(visualizer, tmp_path):
    """测试收敛状态表生成"""
    save_path = tmp_path / "convergence_states.csv"
    
    # 生成表格
    df = visualizer.generate_convergence_table(save_path)
    
    # 检查表格内容
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # 两个radius
    assert "r_g" in df.columns
    assert all(col in df.columns for col in [
        "contribution_ratio", "defection_ratio", "not_converged_ratio",
        "contribution_ci", "defection_ci", "not_converged_ci"
    ])
    
    # 检查文件是否创建
    assert save_path.exists()

def test_save_all_figures(visualizer, tmp_path):
    """测试所有图表保存"""
    output_dir = tmp_path / "figures"
    
    # 保存所有图表
    import seaborn as sns
    with sns.axes_style("darkgrid"):
        visualizer.save_all_figures(str(output_dir))
    
    # 检查目录结构
    assert output_dir.exists()
    assert (output_dir / "plots").exists()
    assert (output_dir / "convergence_states.csv").exists()
    
    # 检查图表文件
    plots_dir = output_dir / "plots"
    for radius in [0.15, 0.3]:
        assert (plots_dir / f"tail_prob_r{radius:.2f}.png").exists()
        assert (plots_dir / f"network_r{radius:.2f}.png").exists()
        assert (plots_dir / f"belief_evolution_r{radius:.2f}.png").exists()

def test_error_handling():
    """测试错误处理"""
    # 测试不存在的目录
    with pytest.raises(FileNotFoundError):
        ExperimentVisualizer("nonexistent_directory")
    
    # 测试无效的分析结果文件
    with pytest.raises(Exception):
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = Exception("Invalid file")
            ExperimentVisualizer("test_dir")

def teardown_module(module):
    """清理测试产生的临时文件和图形"""
    plt.close('all')  # 关闭所有图形
    shutil.rmtree("test_data", ignore_errors=True)