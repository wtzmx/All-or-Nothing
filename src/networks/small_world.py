import numpy as np
from typing import List, Set, Dict, Optional, Tuple

class WSGraph:
    """
    实现Watts-Strogatz小世界网络模型
    
    从规则环形网络开始，以概率p重连边以创建"捷径"
    """
    def __init__(self, 
                 n_nodes: int,
                 k: int,
                 p: float,
                 seed: Optional[int] = None) -> None:
        """
        初始化WS小世界网络
        
        Parameters:
        -----------
        n_nodes : int
            节点数量N
        k : int
            初始近邻数(必须为偶数)
        p : float
            重连概率(0 ≤ p ≤ 1)
        seed : int, optional
            随机数种子
        """
        if n_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
        if k <= 0 or k >= n_nodes or k % 2 != 0:
            raise ValueError("k must be positive even number less than n_nodes")
        if not 0 <= p <= 1:
            raise ValueError("Rewiring probability must be in [0, 1]")
            
        self.N = n_nodes
        self.k = k
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
        # 首先构建规则环形网络
        adj = {i: set() for i in range(self.N)}
        k = self.k // 2
        
        for i in range(self.N):
            for j in range(1, k + 1):
                adj[i].add((i + j) % self.N)
                adj[i].add((i - j) % self.N)
                adj[(i + j) % self.N].add(i)
                adj[(i - j) % self.N].add(i)
                
        # 重连边
        for i in range(self.N):
            for j in list(adj[i]):  # 使用list避免在迭代时修改集合
                if j > i:  # 只处理一次每条边
                    if np.random.random() < self.p:
                        # 移除原有边
                        adj[i].remove(j)
                        adj[j].remove(i)
                        
                        # 添加新边
                        while True:
                            new_target = np.random.randint(self.N)
                            if new_target != i and new_target not in adj[i]:
                                adj[i].add(new_target)
                                adj[new_target].add(i)
                                break
                                
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
            "initial_degree": self.k,
            "rewiring_probability": self.p,
            "n_edges": sum(len(adj) for adj in self.adjacency.values()) // 2,
            "max_degree": self.max_degree,
            "mean_degree": self.mean_degree,
            "n_triangles": self.triangles
        } 