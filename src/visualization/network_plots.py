import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Set, List, Optional, Tuple, Any
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class NetworkVisualizer:
    """网络结构与动态可视化工具"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        """
        初始化可视化器
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            图形大小
        """
        self.figsize = figsize
        # 创建信念值的颜色映射
        self.belief_cmap = LinearSegmentedColormap.from_list(
            'belief_colormap',
            ['#FF4B4B', '#FFB74B', '#4BFF4B']  # 红-黄-绿
        )
        
    def plot_network(self,
                    adjacency: Dict[int, Set[int]],
                    node_positions: Optional[List[Tuple[float, float]]] = None,
                    node_colors: Optional[List[float]] = None,
                    node_sizes: Optional[List[float]] = None,
                    title: str = "Network Structure",
                    show_labels: bool = True,
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制网络结构
        
        Parameters:
        -----------
        adjacency : Dict[int, Set[int]]
            网络邻接表
        node_positions : List[Tuple[float, float]], optional
            节点位置坐标
        node_colors : List[float], optional
            节点颜色值(通常是信念值)
        node_sizes : List[float], optional
            节点大小
        title : str
            图标题
        show_labels : bool
            是否显示节点标签
        ax : plt.Axes, optional
            指定绘图区域
            
        Returns:
        --------
        plt.Axes
            绘图区域对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
            
        # 创建NetworkX图对象
        G = nx.Graph()
        for i in adjacency:
            for j in adjacency[i]:
                if i < j:  # 避免重复边
                    G.add_edge(i, j)
                    
        # 设置节点位置
        if node_positions is None:
            pos = nx.spring_layout(G)
        else:
            pos = {i: (x, y) for i, (x, y) in enumerate(node_positions)}
            
        # 设置默认值
        if node_colors is None:
            node_colors = [0.5] * len(adjacency)  # 默认中性信念
        if node_sizes is None:
            node_sizes = [300] * len(adjacency)  # 默认节点大小
            
        # 绘制网络
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        nodes = nx.draw_networkx_nodes(G, pos,
                                     node_color=node_colors,
                                     node_size=node_sizes,
                                     cmap=self.belief_cmap,
                                     vmin=0, vmax=1,
                                     ax=ax)
        
        if show_labels:
            nx.draw_networkx_labels(G, pos, ax=ax)
            
        # 添加颜色条
        plt.colorbar(nodes, ax=ax, label='Belief')
        
        # 设置图形属性
        ax.set_title(title)
        ax.set_axis_off()
        
        return ax
        
    def plot_belief_distribution(self,
                               beliefs: List[float],
                               title: str = "Belief Distribution",
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制信念分布直方图
        
        Parameters:
        -----------
        beliefs : List[float]
            信念值列表
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
            _, ax = plt.subplots(figsize=(8, 6))
            
        sns.histplot(beliefs, bins=20, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Belief")
        ax.set_ylabel("Count")
        
        return ax
        
    def plot_network_state(self,
                          adjacency: Dict[int, Set[int]],
                          beliefs: List[float],
                          node_positions: Optional[List[Tuple[float, float]]] = None,
                          title: str = "Network State") -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        绘制网络状态完整视图(网络结构+信念分布)
        
        Parameters:
        -----------
        adjacency : Dict[int, Set[int]]
            网络邻接表
        beliefs : List[float]
            信念值列表
        node_positions : List[Tuple[float, float]], optional
            节点位置坐标
        title : str
            图标题
            
        Returns:
        --------
        Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
            图形对象和轴对象元组
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 绘制网络结构
        self.plot_network(
            adjacency=adjacency,
            node_positions=node_positions,
            node_colors=beliefs,
            title=f"{title} - Network Structure",
            ax=ax1
        )
        
        # 绘制信念分布
        self.plot_belief_distribution(
            beliefs=beliefs,
            title=f"{title} - Belief Distribution",
            ax=ax2
        )
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_belief_evolution(self,
                            belief_history: List[List[float]],
                            title: str = "Belief Evolution",
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制信念演化过程
        
        Parameters:
        -----------
        belief_history : List[List[float]]
            每个时间步的信念值列表
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
            
        time_steps = range(len(belief_history))
        beliefs_array = np.array(belief_history)
        
        # 绘制每个智能体的信念轨迹
        for i in range(beliefs_array.shape[1]):
            ax.plot(time_steps, beliefs_array[:, i], alpha=0.3)
            
        # 绘制平均信念
        mean_belief = beliefs_array.mean(axis=1)
        ax.plot(time_steps, mean_belief, 'k-', linewidth=2, label='Mean Belief')
        
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        
        return ax