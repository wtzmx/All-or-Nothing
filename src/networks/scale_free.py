import numpy as np
from typing import List, Set, Dict, Optional, Tuple

class BAGraph:
    """
    实现Barabási-Albert无标度网络模型
    
    使用优先连接机制生成具有幂律度分布的网络
    """
    def __init__(self, 
                 n_nodes: int,
                 m: int,
                 seed: Optional[int] = None) -> None:
        """
        初始化BA无标度网络
        
        Parameters:
        -----------
        n_nodes : int
            最终节点数量N
        m : int
            每个新节点连接的边数
        seed : int, optional
            随机数种子
        """
        if n_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
        if m <= 0 or m >= n_nodes:
            raise ValueError("m must be positive and less than n_nodes")
            
        self.N = n_nodes
        self.m = m
        
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
        
        # 初始化完全图
        for i in range(self.m + 1):
            for j in range(i + 1, self.m + 1):
                adj[i].add(j)
                adj[j].add(i)
                
        # 添加剩余节点
        for i in range(self.m + 1, self.N):
            # 计算现有节点的度数
            degrees = np.array([len(adj[j]) for j in range(i)])
            # 计算连接概率
            probs = degrees / degrees.sum()
            
            # 选择m个不同的目标节点
            targets = np.random.choice(
                range(i), 
                size=self.m, 
                replace=False, 
                p=probs
            )
            
            # 添加新边
            for target in targets:
                adj[i].add(target)
                adj[target].add(i)
                
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
            "edges_per_node": self.m,
            "n_edges": sum(len(adj) for adj in self.adjacency.values()) // 2,
            "max_degree": self.max_degree,
            "mean_degree": self.mean_degree,
            "n_triangles": self.triangles
        } 