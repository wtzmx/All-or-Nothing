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
                "edge_alpha": 0.5
            }
            
        if self.heatmap_config is None:
            self.heatmap_config = {
                "cmap": "viridis",
                "xlabel": "Time steps",
                "ylabel": "Agent ID",
                "aspect": "auto"
            }

class ExperimentVisualizer:
    """实验一可视化器"""
    
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
        """绘制尾概率分布图(Figure 2)"""
        try:
            tail_probs = self.results["tail_probabilities"]
            
            # 为每个radius绘制尾概率分布
            for radius, data in tail_probs.items():
                fig = self.analysis_plotter.plot_convergence_tail_probability(
                    convergence_times=data["times"],
                    labels=[f"r_g = {radius:.2f}"],
                    title=f"Convergence Time Distribution (r_g = {radius:.2f})"
                )
                
                if save_path:
                    fig.figure.savefig(
                        Path(save_path) / f"tail_prob_r{radius:.2f}.{self.config.save_format}",
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
            for radius, data in network_data.items():
                # 创建网络
                network = RandomGeometricGraph(
                    n_nodes=len(data["beliefs"]),
                    radius=radius
                )
                
                # 绘制网络状态
                fig = self.network_plotter.plot_network_state(
                    adjacency=network.adjacency,
                    beliefs=data["beliefs"],
                    title=f"Network State (r_g = {radius:.2f})"
                )
                
                if save_path:
                    fig[0].savefig(
                        Path(save_path) / f"network_r{radius:.2f}.{self.config.save_format}",
                        dpi=self.config.dpi,
                        bbox_inches='tight'
                    )
                    plt.close(fig[0])
                    
        except Exception as e:
            self.logger.error(f"Error plotting network states: {str(e)}")
            raise
            
    def plot_belief_evolution(self,
                            belief_histories: Dict,
                            save_path: Optional[str] = None) -> None:
        """绘制信念演化热力图"""
        try:
            for radius, history in belief_histories.items():
                fig = self.analysis_plotter.plot_belief_evolution_heatmap(
                    belief_history=history,
                    title=f"Belief Evolution (r_g = {radius:.2f})"
                )
                
                if save_path:
                    fig.figure.savefig(
                        Path(save_path) / f"belief_evolution_r{radius:.2f}.{self.config.save_format}",
                        dpi=self.config.dpi,
                        bbox_inches='tight'
                    )
                    plt.close(fig.figure)
                    
        except Exception as e:
            self.logger.error(f"Error plotting belief evolution: {str(e)}")
            raise
            
    def generate_convergence_table(self, 
                                 save_path: Optional[str] = None) -> pd.DataFrame:
        """生成收敛状态统计表(Table 1)"""
        try:
            df = self.results["convergence_states"]
            
            # 格式化表格
            formatted_df = pd.DataFrame()
            formatted_df["r_g"] = df["radius"]
            
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
            if "tail_probabilities" in self.results:
                self.logger.info("Generating tail probability plots...")
                self.plot_tail_probabilities(plots_dir)
            
            # 保存收敛状态表
            if "convergence_states" in self.results:
                self.logger.info("Generating convergence states table...")
                self.generate_convergence_table(
                    output_dir / "convergence_states.csv"
                )
            
            # 保存信念演化图
            if "metastable_states" in self.results:
                self.logger.info("Generating belief evolution plots...")
                # 从metastable_states中提取belief_histories
                belief_histories = {
                    radius: data.get("belief_history", [])
                    for radius, data in self.results["metastable_states"].items()
                }
                if any(belief_histories.values()):  # 只在有数据时生成图表
                    self.plot_belief_evolution(belief_histories, plots_dir)
            
            self.logger.info(f"All figures saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving figures: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行可视化
    visualizer = ExperimentVisualizer("data/experiment1/analysis")
    visualizer.save_all_figures("data/experiment1/figures") 