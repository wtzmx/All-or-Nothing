import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Dict, Any

from experiments.experiment2.exp2_visualization import (
    ExperimentVisualizer, 
    VisualizationConfig
)

class TestExp2Visualization:
    """测试实验二可视化模块"""
    
    @pytest.fixture
    def sample_results(self, tmp_path: Path) -> Path:
        """创建测试用分析结果"""
        # 创建分析结果目录
        analysis_dir = tmp_path / "test_analysis"
        analysis_dir.mkdir()
        
        # 创建测试数据
        results = {
            "tail_probabilities": {
                2: {
                    "times": np.linspace(100, 10000, 100),
                    "probabilities": np.linspace(1, 0, 100)
                },
                4: {
                    "times": np.linspace(100, 10000, 100),
                    "probabilities": np.linspace(1, 0, 100)
                }
            },
            "convergence_states": pd.DataFrame({
                "l_value": [2, 4],
                "total_trials": [500, 500],
                "contribution_ratio": [0.3, 0.4],
                "defection_ratio": [0.6, 0.5],
                "not_converged_ratio": [0.1, 0.1],
                "contribution_ci_lower": [0.25, 0.35],
                "contribution_ci_upper": [0.35, 0.45],
                "defection_ci_lower": [0.55, 0.45],
                "defection_ci_upper": [0.65, 0.55],
                "not_converged_ci_lower": [0.08, 0.08],
                "not_converged_ci_upper": [0.12, 0.12]
            }),
            "convergence_times": {
                2: {
                    "mean": 5000,
                    "median": 4800,
                    "std": 1000,
                    "min": 1000,
                    "max": 9000,
                    "n_samples": 450,
                    "ci_lower": 4800,
                    "ci_upper": 5200
                },
                4: {
                    "mean": 6000,
                    "median": 5800,
                    "std": 1200,
                    "min": 1200,
                    "max": 9500,
                    "n_samples": 450,
                    "ci_lower": 5800,
                    "ci_upper": 6200
                }
            },
            "catastrophe_principle": pd.DataFrame({
                "l_value": [2, 2, 4, 4],
                "sampling": ["no_replacement", "with_replacement"] * 2,
                "max_probability": [0.8, 0.75, 0.7, 0.65],
                "sum_probability": [0.7, 0.65, 0.6, 0.55],
                "ratio": [1.14, 1.15, 1.17, 1.18]
            }),
            "network_states": {
                2: {
                    "beliefs": np.random.random(50)
                },
                4: {
                    "beliefs": np.random.random(50)
                }
            }
        }
        
        # 保存为pickle文件
        with open(analysis_dir / "analysis_results.pkl", 'wb') as f:
            pickle.dump(results, f)
            
        return analysis_dir
    
    @pytest.fixture
    def visualizer(self, sample_results: Path) -> ExperimentVisualizer:
        """创建可视化器实例"""
        return ExperimentVisualizer(str(sample_results))
    
    def test_initialization(self, visualizer: ExperimentVisualizer):
        """测试可视化器初始化"""
        assert visualizer.analysis_dir.exists(), "分析结果目录不存在"
        assert isinstance(visualizer.config, VisualizationConfig), "配置类型错误"
        assert visualizer.results is not None, "结果未正确加载"
        
    def test_plot_tail_probabilities(self, 
                                   visualizer: ExperimentVisualizer, 
                                   tmp_path: Path):
        """测试尾概率分布图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_tail_probabilities(save_path)
        
        # 检查是否生成了图片文件
        for l in [2, 4]:
            assert (save_path / f"tail_prob_l{l}.{visualizer.config.save_format}").exists()
            
    def test_plot_network_states(self, 
                                visualizer: ExperimentVisualizer, 
                                tmp_path: Path):
        """测试网络状态图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_network_states(
            visualizer.results["network_states"],
            save_path
        )
        
        # 检查是否生成了图片文件
        for l in [2, 4]:
            assert (save_path / f"network_l{l}.{visualizer.config.save_format}").exists()
            
    def test_plot_convergence_times(self, 
                                  visualizer: ExperimentVisualizer, 
                                  tmp_path: Path):
        """测试收敛时间统计图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_convergence_times(save_path)
        
        # 检查是否生成了图片文件
        assert (save_path / f"convergence_times.{visualizer.config.save_format}").exists()
        
    def test_plot_catastrophe_ratios(self, 
                                   visualizer: ExperimentVisualizer, 
                                   tmp_path: Path):
        """测试灾难原理比率图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_catastrophe_ratios(save_path)
        
        # 检查是否生成了图片文件
        assert (save_path / f"catastrophe_ratios.{visualizer.config.save_format}").exists()
        
    def test_generate_convergence_table(self, 
                                      visualizer: ExperimentVisualizer, 
                                      tmp_path: Path):
        """测试收敛状态表生成"""
        save_path = tmp_path / "convergence_states.csv"
        
        df = visualizer.generate_convergence_table(save_path)
        
        # 检查表格内容
        assert isinstance(df, pd.DataFrame), "返回类型错误"
        assert "l" in df.columns, "缺少l列"
        assert all(col in df.columns for col in [
            "contribution_ratio", "defection_ratio", "not_converged_ratio",
            "contribution_ci", "defection_ci", "not_converged_ci"
        ]), "缺少必需的列"
        
        # 检查是否保存了文件
        if save_path:
            assert save_path.exists()
            
    def test_save_all_figures(self, 
                             visualizer: ExperimentVisualizer, 
                             tmp_path: Path):
        """测试保存所有图表"""
        output_dir = tmp_path / "figures"
        visualizer.save_all_figures(str(output_dir))
        
        # 检查目录结构
        assert output_dir.exists()
        assert (output_dir / "plots").exists()
        
        # 检查是否生成了所有图表
        plots_dir = output_dir / "plots"
        expected_files = [
            *[f"tail_prob_l{l}.{visualizer.config.save_format}" for l in [2, 4]],
            *[f"network_l{l}.{visualizer.config.save_format}" for l in [2, 4]],
            f"convergence_times.{visualizer.config.save_format}",
            f"catastrophe_ratios.{visualizer.config.save_format}"
        ]
        
        for file in expected_files:
            assert (plots_dir / file).exists(), f"缺少文件: {file}"
            
        # 检查是否生成了表格
        assert (output_dir / "convergence_states.csv").exists()
        
    def test_error_handling(self, tmp_path: Path):
        """测试错误处理"""
        # 测试不存在的目录
        with pytest.raises(FileNotFoundError):
            ExperimentVisualizer(str(tmp_path / "nonexistent"))
            
        # 测试无效的分析结果
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        with open(invalid_dir / "analysis_results.pkl", 'wb') as f:
            pickle.dump({"invalid": "data"}, f)
            
        visualizer = ExperimentVisualizer(str(invalid_dir))
        with pytest.raises(Exception):
            visualizer.plot_tail_probabilities()