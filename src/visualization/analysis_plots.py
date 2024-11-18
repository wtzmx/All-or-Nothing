import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class AnalysisPlotter:
    """实验结果分析绘图工具"""
    
    def __init__(self, style: str = 'seaborn'):
        """
        初始化绘图工具
        
        Parameters:
        -----------
        style : str
            matplotlib绘图风格
        """
        if style == 'seaborn':
            # 直接使用 seaborn 设置样式，而不是通过 matplotlib
            sns.set_theme()
        else:
            plt.style.use(style)
        
    def plot_convergence_tail_probability(self,
                                        convergence_times: List[int],
                                        labels: Optional[List[str]] = None,
                                        title: str = "Convergence Time Distribution",
                                        ax: Optional[plt.Axes] = None,
                                        log_scale: bool = True) -> plt.Axes:
        """
        绘制收敛时间的尾概率分布图 (Figure 2 in paper)
        
        Parameters:
        -----------
        convergence_times : List[int]
            收敛时间列表
        labels : List[str], optional
            不同实验设置的标签
        title : str
            图标题
        ax : plt.Axes, optional
            指定绘图区域
        log_scale : bool
            是否使用对数坐标轴
            
        Returns:
        --------
        plt.Axes
            绘图区域对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        # 计算尾概率
        sorted_times = np.sort(convergence_times)
        tail_probs = np.arange(len(sorted_times), 0, -1) / len(sorted_times)
        
        # 绘制尾概率分布
        if log_scale:
            ax.loglog(sorted_times, tail_probs, 'o-', alpha=0.6, 
                     label=labels[0] if labels else None)
        else:
            ax.plot(sorted_times, tail_probs, 'o-', alpha=0.6,
                   label=labels[0] if labels else None)
            
        ax.set_title(title)
        ax.set_xlabel("Time Steps (t)")
        ax.set_ylabel("P(τ ≥ t)")
        
        if labels:
            ax.legend()
            
        return ax
        
    def plot_belief_evolution_heatmap(self,
                                    belief_history: Union[List[List[float]], np.ndarray],
                                    title: str = "Belief Evolution Heatmap",
                                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制信念演化热力图
        
        Parameters:
        -----------
        belief_history : Union[List[List[float]], np.ndarray]
            每个时间步的信念值列表或数组
        title : str
            图标题
        ax : plt.Axes, optional
            指定绘图区域
            
        Returns:
        --------
        plt.Axes
            绘图区域对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
            
        # 处理空数据情况
        if isinstance(belief_history, list) and not belief_history:
            ax.set_title(title)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Agent ID")
            return ax
        
        # 确保数据是numpy数组
        beliefs_array = np.asarray(belief_history)
        
        # 检查数组是否为空
        if beliefs_array.size == 0:
            ax.set_title(title)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Agent ID")
            return ax
        
        # 创建热力图
        sns.heatmap(beliefs_array.T, 
                   cmap='RdYlGn',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Belief'},
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Agent ID")
        
        return ax
        
    def plot_network_metrics(self,
                           metrics_history: List[Dict[str, float]],
                           metric_names: List[str],
                           title: str = "Network Metrics Evolution",
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制网络指标随时间的变化
        
        Parameters:
        -----------
        metrics_history : List[Dict[str, float]]
            每个时间步的网络指标字典列表
        metric_names : List[str]
            要绘制的指标名称列表
        title : str
            图标题
        ax : plt.Axes, optional
            指定绘图区域
            
        Returns:
        --------
        plt.Axes
            绘图区域对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        time_steps = range(len(metrics_history))
        
        for metric in metric_names:
            values = [m[metric] for m in metrics_history]
            ax.plot(time_steps, values, '-', label=metric)
            
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Metric Value")
        ax.legend()
        
        return ax
        
    def plot_convergence_analysis(self,
                                convergence_data: Dict[str, List[int]],
                                network_params: List[float],
                                param_name: str = "r_g",
                                title: str = "Convergence Analysis") -> Tuple[Figure, Axes]:
        """
        绘制收敛分析图 (Table 1 in paper)
        
        Parameters:
        -----------
        convergence_data : Dict[str, List[int]]
            不同结果的收敛次数统计
            keys: ['contribution', 'defection', 'not_converge']
        network_params : List[float]
            网络参数值列表
        param_name : str
            参数名称
        title : str
            图标题
            
        Returns:
        --------
        Tuple[Figure, Axes]
            图形对象和轴对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(network_params))
        width = 0.25
        
        # 绘制堆叠柱状图
        bottom = np.zeros(len(network_params))
        for label, data in convergence_data.items():
            ax.bar(x, data, width, bottom=bottom, label=label)
            bottom += np.array(data)
            
        ax.set_title(title)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        ax.set_xticklabels(network_params)
        ax.legend()
        
        return fig, ax
        
    def plot_catastrophe_ratio(self,
                             ratios: List[Tuple[float, float]],
                             network_params: List[float],
                             param_name: str = "r_g",
                             title: str = "Catastrophe Principle Ratio",
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制灾难原理比率图 (Table 2 in paper)
        
        Parameters:
        -----------
        ratios : List[Tuple[float, float]]
            每个参数设置的比率对
        network_params : List[float]
            网络参数值列表
        param_name : str
            参数名称
        title : str
            图标题
        ax : plt.Axes, optional
            指定绘图区域
            
        Returns:
        --------
        plt.Axes
            绘图区域对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        x = np.arange(len(network_params))
        width = 0.35
        
        # 绘制分组柱状图
        ratio1 = [r[0] for r in ratios]
        ratio2 = [r[1] for r in ratios]
        
        ax.bar(x - width/2, ratio1, width, label='No Replacement')
        ax.bar(x + width/2, ratio2, width, label='With Replacement')
        
        ax.set_title(title)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Ratio")
        ax.set_xticks(x)
        ax.set_xticklabels(network_params)
        ax.legend()
        
        return ax