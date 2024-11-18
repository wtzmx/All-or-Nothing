import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Any

from experiments.experiment3.exp3_visualization import (
    ExperimentVisualizer,
    VisualizationConfig
)

class TestExp3Visualization:
    """测试实验三可视化器"""
    
    @pytest.fixture
    def sample_results(self, tmp_path: Path) -> Path:
        """创建测试用的分析结果数据"""
        # 创建分析结果目录
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        
        # 创建模拟数据
        results = {
            "tail_probabilities": {
                "geometric": {
                    "radius_0.3": {
                        "times": np.linspace(1, 1000, 100),
                        "probabilities": np.exp(-np.linspace(0, 5, 100))
                    }
                },
                "regular": {
                    "l_2": {
                        "times": np.linspace(1, 1000, 100),
                        "probabilities": np.exp(-np.linspace(0, 4, 100))
                    }
                },
                "random": {
                    "p_0.3": {
                        "times": np.linspace(1, 1000, 100),
                        "probabilities": np.exp(-np.linspace(0, 3, 100))
                    }
                }
            },
            "network_comparison": {
                "convergence_speed": pd.DataFrame({
                    "network_type": ["geometric", "regular", "random"],
                    "params": ["radius_0.3", "l_2", "p_0.3"],
                    "mean_time": [500, 600, 400],
                    "median_time": [450, 550, 350],
                    "std_time": [100, 120, 80]
                }),
                "cooperation_level": pd.DataFrame({
                    "network_type": ["geometric", "regular", "random"],
                    "params": ["radius_0.3", "l_2", "p_0.3"],
                    "cooperation_ratio": [0.6, 0.4, 0.5]
                }),
                "stability": pd.DataFrame({
                    "network_type": ["geometric", "regular", "random"],
                    "params": ["radius_0.3", "l_2", "p_0.3"],
                    "convergence_ratio": [0.9, 0.85, 0.8]
                })
            },
            "network_features": {
                "clustering": {
                    "geometric": {
                        "radius_0.3": {
                            "contribution": {
                                "mean": 0.6,
                                "std": 0.1,
                                "median": 0.58,
                                "count": 30
                            },
                            "defection": {
                                "mean": 0.4,
                                "std": 0.12,
                                "median": 0.38,
                                "count": 25
                            },
                            "not_converged": {
                                "mean": 0.5,
                                "std": 0.15,
                                "median": 0.48,
                                "count": 20
                            }
                        }
                    }
                }
            }
        }
        
        # 保存结果
        with open(analysis_dir / "analysis_results.pkl", 'wb') as f:
            pickle.dump(results, f)
            
        return analysis_dir
        
    @pytest.fixture
    def visualizer(self, sample_results: Path) -> ExperimentVisualizer:
        """创建可视化器实例"""
        return ExperimentVisualizer(str(sample_results))
        
    def test_initialization(self, visualizer: ExperimentVisualizer):
        """测试可视化器初始化"""
        assert visualizer.analysis_dir.exists()
        assert isinstance(visualizer.config, VisualizationConfig)
        assert visualizer.logger is not None
        assert visualizer.results is not None
        
    def test_plot_tail_probabilities(self, 
                                   visualizer: ExperimentVisualizer,
                                   tmp_path: Path):
        """测试尾概率分布图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_tail_probabilities(save_path)
        
        # 检查是否生成了图片文件
        assert (save_path / f"tail_prob_comparison.{visualizer.config.save_format}").exists()
        
    def test_plot_network_comparison(self,
                                   visualizer: ExperimentVisualizer,
                                   tmp_path: Path):
        """测试网络结构对比图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_network_comparison(save_path)
        
        # 检查是否生成了所有对比图
        assert (save_path / f"convergence_speed.{visualizer.config.save_format}").exists()
        assert (save_path / f"cooperation_level.{visualizer.config.save_format}").exists()
        assert (save_path / f"stability.{visualizer.config.save_format}").exists()
        
    def test_plot_network_features(self,
                                 visualizer: ExperimentVisualizer,
                                 tmp_path: Path):
        """测试网络特征分析图绘制"""
        save_path = tmp_path / "plots"
        save_path.mkdir()
        
        visualizer.plot_network_features(save_path)
        
        # 检查是否生成了特征分析图
        for feature in visualizer.results["network_features"].keys():
            assert (save_path / f"feature_{feature}.{visualizer.config.save_format}").exists()
            
    def test_save_all_figures(self,
                            visualizer: ExperimentVisualizer,
                            tmp_path: Path):
        """测试保存所有图表"""
        output_dir = tmp_path / "output"
        visualizer.save_all_figures(str(output_dir))
        
        # 检查输出目录结构
        assert output_dir.exists()
        assert (output_dir / "plots").exists()
        
        # 检查是否生成了所有图表
        plots_dir = output_dir / "plots"
        assert (plots_dir / f"tail_prob_comparison.{visualizer.config.save_format}").exists()
        assert (plots_dir / f"convergence_speed.{visualizer.config.save_format}").exists()
        assert (plots_dir / f"cooperation_level.{visualizer.config.save_format}").exists()
        assert (plots_dir / f"stability.{visualizer.config.save_format}").exists()
        for feature in visualizer.results["network_features"].keys():
            assert (plots_dir / f"feature_{feature}.{visualizer.config.save_format}").exists()
            
    def test_config_customization(self, sample_results: Path):
        """测试配置自定义"""
        custom_config = VisualizationConfig(
            figure_size=(12, 8),
            dpi=150,
            style="default",
            color_palette="Set3",
            save_format="pdf"
        )
        
        visualizer = ExperimentVisualizer(
            str(sample_results),
            config=custom_config
        )
        
        assert visualizer.config.figure_size == (12, 8)
        assert visualizer.config.dpi == 150
        assert visualizer.config.style == "default"
        assert visualizer.config.color_palette == "Set3"
        assert visualizer.config.save_format == "pdf"
        
    def test_error_handling(self, tmp_path: Path):
        """测试错误处理"""
        # 测试无效目录
        with pytest.raises(FileNotFoundError):
            ExperimentVisualizer(str(tmp_path / "nonexistent"))
            
        # 测试无效结果文件
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        with pytest.raises(Exception):
            visualizer = ExperimentVisualizer(str(invalid_dir))
            
    @pytest.mark.parametrize("plot_func", [
        "plot_tail_probabilities",
        "plot_network_comparison",
        "plot_network_features"
    ])
    def test_plot_functions_without_save(self,
                                       visualizer: ExperimentVisualizer,
                                       plot_func: str):
        """测试绘图函数不保存时的行为"""
        # 调用绘图函数但不提供保存路径
        getattr(visualizer, plot_func)()
        
        # 确保没有错误发生
        plt.close('all')

if __name__ == "__main__":
    pytest.main(["-v"])
