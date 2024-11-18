import numpy as np
from typing import List, Set, Dict, Optional, Tuple

class CirculantGraph:
    """
    实现环形规则图模型
    
    N个节点排列成环形，每个节点与其最近的l个邻居相连
    """
    def __init__(self, 
                 n_nodes: int,
                 neighbors: int,
                 seed: Optional[int] = None) -> None:
        """
        初始化环形规则图
        
        Parameters:
        -----------
        n_nodes : int
            节点数量N
        neighbors : int
            每个节点连接到的最近邻居数量l (必须为偶数)
        seed : int, optional
            随机数种子(用于兼容接口，实际未使用)
        """
        if n_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
        if neighbors <= 0 or neighbors >= n_nodes:
            raise ValueError("Number of neighbors must be in (0, n_nodes)")
        if neighbors % 2 != 0:
            raise ValueError("Number of neighbors must be even")
            
        self.N = n_nodes
        self.l = neighbors
        
        # 构建邻接表
        self.adjacency = self._build_adjacency()
        
        # 计算并缓存网络特征
        self._cache_network_features()
        
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """构建邻接表"""
        adj = {i: set() for i in range(self.N)}
        
        # 每个节点连接到其最近的l/2个顺时针和逆时针邻居
        k = self.l // 2
        for i in range(self.N):
            for j in range(1, k + 1):
                # 顺时针连接
                neighbor = (i + j) % self.N
                adj[i].add(neighbor)
                adj[neighbor].add(i)
                
                # 逆时针连接
                neighbor = (i - j) % self.N
                adj[i].add(neighbor)
                adj[neighbor].add(i)
                
        return adj
        
    def _cache_network_features(self) -> None:
        """计算并缓存网络特征"""
        # 在规则图中所有节点的度数相同
        self.degrees = {i: self.l for i in range(self.N)}
        self.max_degree = self.l
        self.mean_degree = float(self.l)
        
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
    
    def get_node_positions(self) -> List[Tuple[float, float]]:
        """获取所有节点的坐标(环形布局)，用于可视化"""
        positions = []
        for i in range(self.N):
            angle = 2 * np.pi * i / self.N
            x = np.cos(angle)
            y = np.sin(angle)
            positions.append((x, y))
        return positions
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """获取所有边的列表，用于可视化"""
        edges = []
        for i in range(self.N):
            for j in self.adjacency[i]:
                if i < j:  # 避免重复边
                    edges.append((i, j))
        return edges
    
    def get_stats(self) -> Dict:
        """获取网络统计信息"""
        return {
            "n_nodes": self.N,
            "n_neighbors": self.l,
            "n_edges": (self.N * self.l) // 2,
            "max_degree": self.max_degree,
            "mean_degree": self.mean_degree,
            "n_triangles": self.triangles,
            "is_regular": True
        }