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
                "edge_alpha": 0.5,
                "layout": "circular"  # 规则图使用环形布局
            }
            
        if self.heatmap_config is None:
            self.heatmap_config = {
                "cmap": "viridis",
                "xlabel": "Time steps",
                "ylabel": "Agent ID",
                "aspect": "auto"
            }

class ExperimentVisualizer:
    """实验二可视化器"""
    
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
        """绘制尾概率分布图(Figure 5)"""
        try:
            tail_probs = self.results["tail_probabilities"]
            
            # 为每个l值绘制尾概率分布
            for l_value, data in tail_probs.items():
                fig = self.analysis_plotter.plot_convergence_tail_probability(
                    convergence_times=data["times"],
                    labels=[f"l = {l_value}"],
                    title=f"Convergence Time Distribution (l = {l_value})"
                )
                
                if save_path:
                    fig.figure.savefig(
                        Path(save_path) / f"tail_prob_l{l_value}.{self.config.save_format}",
                        dpi=self.config.dpi,
                        bbox_inches='tight'
                    )
                    plt.close(fig.figure)
                    
        except Exception as e:
            self.logger.error(f"Error plotting tail probabilities: {str(e)}")
            raise
            
    def plot_network_states(self,
                           network_data: Dict,
                           save_path: Optional[str] = None) -> None:
        """绘制网络状态可视化图"""
        try:
            for l_value, data in network_data.items():
                # 创建规则图网络
                network = CirculantGraph(
                    n_nodes=len(data["beliefs"]),
                    neighbors=l_value
                )
                
                # 获取节点位置（使用环形布局）
                node_positions = network.get_node_positions()
                
                # 绘制网络状态
                fig = self.network_plotter.plot_network_state(
                    adjacency=network.adjacency,
                    beliefs=data["beliefs"],
                    node_positions=node_positions,
                    title=f"Network State (l = {l_value})"
                )
                
                if save_path:
                    fig[0].savefig(
                        Path(save_path) / f"network_l{l_value}.{self.config.save_format}",
                        dpi=self.config.dpi,
                        bbox_inches='tight'
                    )
                    plt.close(fig[0])
                    
        except Exception as e:
            self.logger.error(f"Error plotting network states: {str(e)}")
            raise
            
    def plot_convergence_times(self,
                             save_path: Optional[str] = None) -> None:
        """绘制收敛时间统计图"""
        try:
            conv_times = self.results["convergence_times"]
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            l_values = sorted(conv_times.keys())
            means = [conv_times[l]["mean"] for l in l_values]
            ci_lower = [conv_times[l]["ci_lower"] for l in l_values]
            ci_upper = [conv_times[l]["ci_upper"] for l in l_values]
            
            # 绘制均值和置信区间
            ax.plot(l_values, means, 'o-', label='Mean')
            ax.fill_between(l_values, ci_lower, ci_upper, alpha=0.2)
            
            ax.set_xlabel('Nearest neighbors (l)')
            ax.set_ylabel('Convergence time')
            ax.set_title('Mean Convergence Time vs. Number of Neighbors')
            ax.grid(True)
            
            if save_path:
                fig.savefig(
                    Path(save_path) / f"convergence_times.{self.config.save_format}",
                    dpi=self.config.dpi,
                    bbox_inches='tight'
                )
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error plotting convergence times: {str(e)}")
            raise
            
    def plot_catastrophe_ratios(self,
                              save_path: Optional[str] = None) -> None:
        """绘制灾难原理比率图"""
        try:
            df = self.results["catastrophe_principle"]
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 绘制不同采样方式的比率
            for sampling in ["no_replacement", "with_replacement"]:
                data = df[df["sampling"] == sampling]
                ax.plot(
                    data["l_value"], 
                    data["ratio"],
                    'o-',
                    label=f'Sampling: {sampling}'
                )
                
            ax.set_xlabel('Nearest neighbors (l)')
            ax.set_ylabel('P(max{X1,X2} > t) : P(X1 + X2 > t)')
            ax.set_title('Catastrophe Principle Ratio')
            ax.grid(True)
            ax.legend()
            
            if save_path:
                fig.savefig(
                    Path(save_path) / f"catastrophe_ratios.{self.config.save_format}",
                    dpi=self.config.dpi,
                    bbox_inches='tight'
                )
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Error plotting catastrophe ratios: {str(e)}")
            raise
            
    def generate_convergence_table(self, 
                                 save_path: Optional[str] = None) -> pd.DataFrame:
        """生成收敛状态统计表"""
        try:
            df = self.results["convergence_states"]
            
            # 格式化表格
            formatted_df = pd.DataFrame()
            formatted_df["l"] = df["l_value"]
            
            # 添加比例和置信区间
            for state in ["contribution", "defection", "not_converged"]:
                formatted_df[f"{state}_ratio"] = df[f"{state}_ratio"].apply(
                    lambda x: f"{x:.3f}"
                )
                formatted_df[f"{state}_ci"] = df.apply(
                    lambda row: f"({row[f'{state}_ci_lower']:.3f}, "
                               f"{row[f'{state}_ci_upper']:.3f})",
                    axis=1
                )
                
            # 保存表格
            if save_path:
                formatted_df.to_csv(save_path, index=False)
                
            return formatted_df
            
        except Exception as e:
            self.logger.error(f"Error generating convergence table: {str(e)}")
            raise
            
    def save_all_figures(self, output_dir: str):
        """保存所有图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 创建子目录
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # 保存尾概率分布图
            self.plot_tail_probabilities(plots_dir)
            
            # 保存网络状态图
            self.plot_network_states(
                self.results["network_states"],
                plots_dir
            )
            
            # 保存收敛时间统计图
            self.plot_convergence_times(plots_dir)
            
            # 保存灾难原理比率图
            self.plot_catastrophe_ratios(plots_dir)
            
            # 保存收敛状态表
            self.generate_convergence_table(
                output_dir / "convergence_states.csv"
            )
            
            self.logger.info(f"All figures saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving figures: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行可视化
    visualizer = ExperimentVisualizer("data/experiment2/analysis")
    visualizer.save_all_figures("data/experiment2/figures")