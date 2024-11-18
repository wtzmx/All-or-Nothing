import numpy as np
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Point:
    """表示2D平面上的一个点"""
    x: float
    y: float
    id: int
    
    def distance_to(self, other: 'Point') -> float:
        """计算到另一个点的欧氏距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class RandomGeometricGraph:
    """
    实现随机几何图模型
    
    在[0,1]²的正方形空间中随机放置N个点，
    当两点间距离小于r时建立连接
    """
    def __init__(self, 
                 n_nodes: int,
                 radius: float,
                 seed: Optional[int] = None) -> None:
        """
        初始化随机几何图
        
        Parameters:
        -----------
        n_nodes : int
            节点数量N
        radius : float
            连接半径r_g，需要在(0,1)范围内
        seed : int, optional
            随机数种子，用于复现
        """
        if not 0 < radius < 1:
            raise ValueError("Radius must be in (0,1)")
        if n_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
            
        self.N = n_nodes
        self.radius = radius
        self.rng = np.random.RandomState(seed)
        
        # 生成点的位置
        self.points = self._generate_points()
        # 建立邻接表
        self.adjacency = self._build_adjacency()
        # 检查连通性并在需要时重新生成
        while not self.is_connected():
            self.points = self._generate_points()
            self.adjacency = self._build_adjacency()
            
        # 计算并缓存一些网络特征
        self._cache_network_features()
        
    def _generate_points(self) -> List[Point]:
        """在单位正方形内随机生成点"""
        points = []
        coords = self.rng.uniform(0, 1, size=(self.N, 2))
        
        for i in range(self.N):
            points.append(Point(
                x=coords[i,0],
                y=coords[i,1],
                id=i
            ))
            
        return points
    
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """构建邻接表"""
        adj = {i: set() for i in range(self.N)}
        
        # 检查所有点对之间的距离
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.points[i].distance_to(self.points[j]) <= self.radius:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        return adj
    
    def is_connected(self) -> bool:
        """检查图是否连通，使用DFS"""
        if not self.points:
            return False
            
        visited = set()
        stack = [0]  # 从节点0开始搜索
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self.adjacency[node] - visited)
                
        return len(visited) == self.N
    
    def _cache_network_features(self) -> None:
        """计算并缓存网络特征"""
        # 计算度
        self.degrees = {i: len(neighbors) 
                       for i, neighbors in self.adjacency.items()}
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
    
    def get_node_positions(self) -> List[Tuple[float, float]]:
        """获取所有节点的坐标，用于可视化"""
        return [(p.x, p.y) for p in self.points]
    
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
            "radius": self.radius,
            "n_edges": sum(len(v) for v in self.adjacency.values()) // 2,
            "max_degree": self.max_degree,
            "mean_degree": self.mean_degree,
            "n_triangles": self.triangles,
            "is_connected": self.is_connected()
        }