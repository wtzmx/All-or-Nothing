import numpy as np
from typing import Dict, Set, List, Tuple
from collections import defaultdict

class NetworkMetrics:
    """网络分析度量工具类"""
    
    @staticmethod
    def adjacency_to_matrix(adjacency: Dict[int, Set[int]], n_nodes: int) -> np.ndarray:
        """将邻接表转换为邻接矩阵"""
        matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in adjacency:
            for j in adjacency[i]:
                matrix[i,j] = 1
        return matrix
    
    @staticmethod
    def calculate_degrees(adjacency: Dict[int, Set[int]]) -> Dict[str, float]:
        """计算度数相关指标"""
        if not adjacency:  # 处理空图
            return {
                "max_degree": 0,
                "min_degree": 0,
                "mean_degree": 0.0,
                "std_degree": 0.0
            }
        degrees = [len(neighbors) for neighbors in adjacency.values()]
        return {
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "mean_degree": np.mean(degrees),
            "std_degree": np.std(degrees)
        }
    
    @staticmethod
    def count_triangles(adjacency: Dict[int, Set[int]]) -> int:
        """计算网络中的三角形数量"""
        if not adjacency:  # 处理空图
            return 0
            
        count = 0
        for i in adjacency:
            neighbors_i = adjacency[i]
            for j in neighbors_i:
                if j > i:  # 避免重复计数
                    neighbors_j = adjacency[j]
                    common = neighbors_i & neighbors_j
                    for k in common:
                        if k > j:  # 避免重复计数
                            count += 1
        return count
    
    @staticmethod
    def calculate_clustering(adjacency: Dict[int, Set[int]]) -> Dict[str, float]:
        """计算聚类系数"""
        if not adjacency:  # 处理空图
            return {
                "global_clustering": 0.0,
                "local_clustering": []
            }
            
        local_coeffs = []
        for i in adjacency:
            neighbors = adjacency[i]
            k = len(neighbors)
            if k < 2:  # 度数小于2的节点聚类系数定义为0
                local_coeffs.append(0.0)
                continue
                
            # 计算邻居之间的连接数
            connections = 0
            for u in neighbors:
                for v in neighbors:
                    if u < v and v in adjacency[u]:
                        connections += 1
            
            # 计算局部聚类系数
            local_coeffs.append(2.0 * connections / (k * (k-1)))
        
        if not local_coeffs:  # 如果没有有效的局部聚类系数
            return {
                "global_clustering": 0.0,
                "local_clustering": local_coeffs
            }
            
        return {
            "global_clustering": np.mean(local_coeffs),
            "local_clustering": local_coeffs
        }
    
    @staticmethod
    def calculate_path_lengths(adjacency: Dict[int, Set[int]]) -> Dict[str, float]:
        """计算最短路径相关指标"""
        if not adjacency:  # 处理空图
            return {
                "average_path_length": 0.0,
                "diameter": 0,
                "is_connected": True  # 空图视为连通的
            }
            
        n = len(adjacency)
        if n == 1:  # 处理单节点图
            return {
                "average_path_length": 0.0,
                "diameter": 0,
                "is_connected": True
            }
            
        distances = defaultdict(lambda: float('inf'))
        
        # 初始化距离
        for i in adjacency:
            distances[(i,i)] = 0
            for j in adjacency[i]:
                distances[(i,j)] = 1
                distances[(j,i)] = 1
        
        # Floyd-Warshall算法
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:  # 只考虑不同节点之间的路径
                        dist_ij = distances[(i,j)]
                        dist_ik = distances[(i,k)]
                        dist_kj = distances[(k,j)]
                        distances[(i,j)] = min(dist_ij, dist_ik + dist_kj)
        
        # 只考虑不同节点对之间的有限路径
        finite_paths = [d for (i,j), d in distances.items() 
                       if d != float('inf') and i != j]
        
        if not finite_paths:  # 如果没有有效路径
            return {
                "average_path_length": 0.0,
                "diameter": 0,
                "is_connected": False
            }
            
        return {
            "average_path_length": np.mean(finite_paths),
            "diameter": max(finite_paths),
            "is_connected": len(finite_paths) == n * (n-1)  # 修正连通性判断
        }
    
    @staticmethod
    def get_complete_stats(adjacency: Dict[int, Set[int]]) -> Dict:
        """获取完整的网络统计信息"""
        if not adjacency:  # 处理空图
            return {
                "n_nodes": 0,
                "n_edges": 0,
                "max_degree": 0,
                "min_degree": 0,
                "mean_degree": 0.0,
                "std_degree": 0.0,
                "n_triangles": 0,
                "global_clustering": 0.0,
                "local_clustering": [],
                "average_path_length": 0.0,
                "diameter": 0,
                "is_connected": True
            }
            
        n_nodes = len(adjacency)
        
        # 合并所有统计指标
        stats = {
            "n_nodes": n_nodes,
            "n_edges": sum(len(v) for v in adjacency.values()) // 2,
        }
        
        stats.update(NetworkMetrics.calculate_degrees(adjacency))
        stats.update({"n_triangles": NetworkMetrics.count_triangles(adjacency)})
        stats.update(NetworkMetrics.calculate_clustering(adjacency))
        stats.update(NetworkMetrics.calculate_path_lengths(adjacency))
        
        return stats