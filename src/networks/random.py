import numpy as np
from typing import List, Set, Dict, Optional, Tuple

class ERGraph:
    """
    实现Erdős-Rényi随机图模型
    
    以概率p连接任意两个节点
    """
    def __init__(self, 
                 n_nodes: int,
                 p: float,
                 seed: Optional[int] = None) -> None:
        """
        初始化ER随机图
        
        Parameters:
        -----------
        n_nodes : int
            节点数量N
        p : float
            连接概率(0 < p < 1)
        seed : int, optional
            随机数种子
        """
        if n_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
        if not 0 < p < 1:
            raise ValueError("Connection probability must be in (0, 1)")
            
        self.N = n_nodes
        self.p = p
        
        # 设置随机数种子
        if seed is not None:
            np.random.seed(seed)
            
        # 构建邻接表
        self.adjacency = self._build_adjacency()
        
        # 计算并缓存网络特征
        self._cache_network_features()
        
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """构建邻接表"""
        adj = {i: set() for i in range(self.N)}
        
        # 对每对节点以概率p添加边
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if np.random.random() < self.p:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        return adj
        
    def _cache_network_features(self) -> None:
        """计算并缓存网络特征"""
        # 计算度数
        self.degrees = {i: len(self.adjacency[i]) for i in range(self.N)}
        self.max_degree = max(self.degrees.values())
        self.mean_degree = sum(self.degrees.values()) / self.N
        
        # 计算三角形数量
        self.triangles = self._count_triangles()
        
    def _count_triangles(self) -> int:
        """统计网络中的三角形数量"""
        count = 0
        for i in range(self.N):
            neighbors = self.adjacency[i]
            for j in neighbors:
                if j > i:  # 避免重复计数
                    for k in (neighbors & self.adjacency[j]):
                        if k > j:  # 避免重复计数
                            count += 1
        return count
    
    def get_neighbors(self, node_id: int) -> Set[int]:
        """获取指定节点的邻居集合"""
        if node_id not in self.adjacency:
            raise ValueError(f"Node {node_id} not in graph")
        return self.adjacency[node_id].copy()
    
    def get_closed_neighbors(self, node_id: int) -> Set[int]:
        """获取指定节点的闭邻居集合(包含节点自身)"""
        neighbors = self.get_neighbors(node_id)
        neighbors.add(node_id)
        return neighbors
    
    def get_stats(self) -> Dict:
        """获取网络统计信息"""
        return {
            "n_nodes": self.N,
            "connection_probability": self.p,
            "n_edges": sum(len(adj) for adj in self.adjacency.values()) // 2,
            "max_degree": self.max_degree,
            "mean_degree": self.mean_degree,
            "n_triangles": self.triangles
        } 