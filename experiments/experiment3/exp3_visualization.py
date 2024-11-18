import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.visualization.analysis_plots import AnalysisPlotter
from src.visualization.network_plots import NetworkVisualizer
from src.networks.geometric import RandomGeometricGraph
from src.networks.regular import CirculantGraph

@dataclass
class VisualizationConfig:
    """可视化配置类"""
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 300
    style: str = "seaborn"
    color_palette: str = "Set2"
    font_family: str = "DejaVu Sans"
    save_format: str = "png"
    
    # 尾概率图配置
    tail_prob_config: Dict = None
    # 网络状态图配置
    network_config: Dict = None
    # 热力图配置
    heatmap_config: Dict = None
    # 对比图配置
    comparison_config: Dict = None
    
    def __post_init__(self):
        if self.tail_prob_config is None:
            self.tail_prob_config = {
                "xlabel": "Time steps (t)",
                "ylabel": "P(τ ≥ t)",
                "xscale": "log",
                "yscale": "log",
                "grid": True
            }
        
        if self.network_config is None:
            self.network_config = {
                "node_size": 100,
                "node_color_map": "RdYlBu",
                "edge_color": "gray",
                "edge_alpha": 0.5
            }
            
        if self.heatmap_config is None:
            self.heatmap_config = {
                "cmap": "viridis",
                "xlabel": "Time steps",
                "ylabel": "Agent ID",
                "aspect": "auto"
            }
            
        if self.comparison_config is None:
            self.comparison_config = {
                "bar_width": 0.8,
                "alpha": 0.8,
                "capsize": 5
            }

class ExperimentVisualizer:
    """实验三可视化器：网络结构对比研究"""
    
    def __init__(self, 
                 analysis_dir: str,
                 config: Optional[VisualizationConfig] = None):
        """
        初始化可视化器
        
        Parameters:
        -----------
        analysis_dir : str
            分析结果目录路径
        config : VisualizationConfig, optional
            可视化配置
        """
        self.analysis_dir = Path(analysis_dir)
        if not self.analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory {analysis_dir} not found")
            
        self.config = config or VisualizationConfig()
        
        # 初始化基础可视化工具
        self.analysis_plotter = AnalysisPlotter(style=self.config.style)
        self.network_plotter = NetworkVisualizer(figsize=self.config.figure_size)
        
        # 设置日志
        self._setup_logging()
        
        # 加载分析结果
        self.results = self._load_analysis_results()
        
    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_analysis_results(self) -> Dict:
        """加载分析结果"""
        try:
            with open(self.analysis_dir / "analysis_results.pkl", 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading analysis results: {str(e)}")
            raise
            
    def plot_tail_probabilities(self, 
                              save_path: Optional[str] = None) -> None:
        """绘制不同网络类型的尾概率分布对比图"""
        try:
            tail_probs = self.results["tail_probabilities"]
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            for network_type, type_data in tail_probs.items():
                for param_str, data in type_data.items():
                    ax.plot(
                        data["times"],
                        data["probabilities"],
                        label=f"{network_type} ({param_str})"
                    )
                    
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Time steps (t)")
            ax.set_ylabel("P(τ ≥ t)")
            ax.set_title("Convergence Time Distribution Comparison")
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(
                    Path(save_path) / f"tail_prob_comparison.{self.config.save_format}",
                    dpi=self.config.dpi,
                    bbox_inches='tight'
                )
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error plotting tail probabilities: {str(e)}")
            raise
            
    def plot_network_comparison(self,
                              save_path: Optional[str] = None) -> None:
        """绘制网络结构性能对比图"""
        try:
            comparison_data = self.results["network_comparison"]
            
            # 绘制收敛速度对比
            self._plot_convergence_speed(
                comparison_data["convergence_speed"],
                save_path
            )
            
            # 绘制合作水平对比
            self._plot_cooperation_level(
                comparison_data["cooperation_level"],
                save_path
            )
            
            # 绘制稳定性对比
            self._plot_stability(
                comparison_data["stability"],
                save_path
            )
            
        except Exception as e:
            self.logger.error(f"Error plotting network comparison: {str(e)}")
            raise
            
    def _plot_convergence_speed(self, 
                              df: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
        """绘制收敛速度对比图"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = np.arange(len(df))
        labels = [f"{row.network_type}\n{row.params}" for _, row in df.iterrows()]
        
        ax.bar(
            x,
            df["mean_time"],
            yerr=df["std_time"],
            capsize=self.config.comparison_config["capsize"],
            alpha=self.config.comparison_config["alpha"],
            width=self.config.comparison_config["bar_width"]
        )
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Convergence Time")
        ax.set_title("Convergence Speed Comparison")
        
        if save_path:
            fig.savefig(
                Path(save_path) / f"convergence_speed.{self.config.save_format}",
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
            plt.close(fig)
            
    def _plot_cooperation_level(self,
                              df: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
        """绘制合作水平对比图"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = np.arange(len(df))
        labels = [f"{row.network_type}\n{row.params}" for _, row in df.iterrows()]
        
        ax.bar(
            x,
            df["cooperation_ratio"],
            alpha=self.config.comparison_config["alpha"],
            width=self.config.comparison_config["bar_width"]
        )
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Cooperation Ratio")
        ax.set_title("Cooperation Level Comparison")
        
        if save_path:
            fig.savefig(
                Path(save_path) / f"cooperation_level.{self.config.save_format}",
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
            plt.close(fig)
            
    def _plot_stability(self,
                       df: pd.DataFrame,
                       save_path: Optional[str] = None) -> None:
        """绘制稳定性对比图"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = np.arange(len(df))
        labels = [f"{row.network_type}\n{row.params}" for _, row in df.iterrows()]
        
        ax.bar(
            x,
            df["convergence_ratio"],
            alpha=self.config.comparison_config["alpha"],
            width=self.config.comparison_config["bar_width"]
        )
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Convergence Ratio")
        ax.set_title("Stability Comparison")
        
        if save_path:
            fig.savefig(
                Path(save_path) / f"stability.{self.config.save_format}",
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
            plt.close(fig)
            
    def plot_network_features(self,
                            save_path: Optional[str] = None) -> None:
        """绘制网络特征分析图"""
        try:
            feature_data = self.results["network_features"]
            
            for feature, stats in feature_data.items():
                self._plot_feature_comparison(
                    feature,
                    stats,
                    save_path
                )
                
        except Exception as e:
            self.logger.error(f"Error plotting network features: {str(e)}")
            raise
            
    def _plot_feature_comparison(self,
                               feature: str,
                               stats: Dict,
                               save_path: Optional[str] = None) -> None:
        """绘制单个特征的对比图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 准备数据
            plot_data = []
            for network_type, type_data in stats.items():
                for param_str, param_data in type_data.items():
                    for state, state_stats in param_data.items():
                        plot_data.append({
                            "network_type": network_type,
                            "params": param_str,
                            "state": state,
                            "mean": state_stats["mean"],
                            "std": state_stats["std"]
                        })
                        
            df = pd.DataFrame(plot_data)
            
            # 绘制分组柱状图
            network_types = df["network_type"].unique()
            x = np.arange(len(network_types))
            width = 0.25
            
            for i, state in enumerate(["contribution", "defection", "not_converged"]):
                state_data = df[df["state"] == state]
                if not state_data.empty:
                    # 确保数据按照network_types的顺序
                    means = []
                    stds = []
                    for net_type in network_types:
                        net_data = state_data[state_data["network_type"] == net_type]
                        means.append(net_data["mean"].iloc[0] if not net_data.empty else 0)
                        stds.append(net_data["std"].iloc[0] if not net_data.empty else 0)
                    
                    ax.bar(
                        x + i * width,
                        means,
                        width,
                        yerr=stds,
                        label=state,
                        capsize=5
                    )
            
            # 设置x轴标签
            ax.set_xticks(x + width)
            labels = []
            for net_type in network_types:
                params = df[df["network_type"] == net_type]["params"].iloc[0]
                labels.append(f"{net_type}\n{params}")
            ax.set_xticklabels(labels)
            
            ax.set_ylabel(f"{feature} Value")
            ax.set_title(f"{feature} Comparison")
            ax.legend()
            
            if save_path:
                fig.savefig(
                    Path(save_path) / f"feature_{feature}.{self.config.save_format}",
                    dpi=self.config.dpi,
                    bbox_inches='tight'
                )
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error plotting feature {feature}: {str(e)}")
            raise
            
    def save_all_figures(self, output_dir: str):
        """保存所有图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 创建子目录
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # 保存尾概率分布对比图
            self.plot_tail_probabilities(plots_dir)
            
            # 保存网络结构对比图
            self.plot_network_comparison(plots_dir)
            
            # 保存网络特征分析图
            self.plot_network_features(plots_dir)
            
            self.logger.info(f"All figures saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving figures: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行可视化
    visualizer = ExperimentVisualizer("data/experiment3/analysis")
    visualizer.save_all_figures("data/experiment3/figures")
