import pytest
import numpy as np
from src.networks.metrics import NetworkMetrics
from typing import Dict, Set

@pytest.fixture
def small_graph_adjacency() -> Dict[int, Set[int]]:
    """创建一个小型测试图的邻接表"""
    return {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1, 3},
        3: {2}
    }

@pytest.fixture
def triangle_graph_adjacency() -> Dict[int, Set[int]]:
    """创建一个包含三角形的测试图"""
    return {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1}
    }

@pytest.fixture
def disconnected_graph_adjacency() -> Dict[int, Set[int]]:
    """创建一个非连通图"""
    return {
        0: {1},
        1: {0},
        2: {3},
        3: {2}
    }

def test_adjacency_to_matrix(small_graph_adjacency):
    """测试邻接表到邻接矩阵的转换"""
    matrix = NetworkMetrics.adjacency_to_matrix(small_graph_adjacency, 4)
    
    # 验证矩阵维度
    assert matrix.shape == (4, 4)
    
    # 验证对称性
    assert np.array_equal(matrix, matrix.T)
    
    # 验证具体连接
    assert matrix[0,1] == 1
    assert matrix[0,2] == 1
    assert matrix[0,3] == 0
    assert matrix[1,2] == 1
    assert matrix[2,3] == 1

def test_calculate_degrees(small_graph_adjacency):
    """测试度数统计的计算"""
    stats = NetworkMetrics.calculate_degrees(small_graph_adjacency)
    
    assert stats["max_degree"] == 3  # 节点2的度数
    assert stats["min_degree"] == 1  # 节点3的度数
    assert stats["mean_degree"] == 2.0  # (2 + 2 + 3 + 1) / 4
    assert stats["std_degree"] > 0

def test_count_triangles():
    """测试三角形计数"""
    # 测试完整三角形
    triangle = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1}
    }
    assert NetworkMetrics.count_triangles(triangle) == 1
    
    # 测试无三角形的图
    line = {
        0: {1},
        1: {0, 2},
        2: {1}
    }
    assert NetworkMetrics.count_triangles(line) == 0
    
    # 测试多个三角形
    diamond = {
        0: {1, 2},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {1, 2}
    }
    assert NetworkMetrics.count_triangles(diamond) == 2

def test_calculate_clustering(triangle_graph_adjacency):
    """测试聚类系数计算"""
    clustering = NetworkMetrics.calculate_clustering(triangle_graph_adjacency)
    
    # 在完整三角形中，所有节点的局部聚类系数应该为1
    assert clustering["global_clustering"] == 1.0
    assert all(c == 1.0 for c in clustering["local_clustering"])
    
    # 测试星形图(中心节点的聚类系数为0)
    star = {
        0: {1, 2, 3},
        1: {0},
        2: {0},
        3: {0}
    }
    star_clustering = NetworkMetrics.calculate_clustering(star)
    assert star_clustering["global_clustering"] == 0.0

def test_calculate_path_lengths(small_graph_adjacency, disconnected_graph_adjacency):
    """测试路径长度计算"""
    # 测试连通图
    connected = NetworkMetrics.calculate_path_lengths(small_graph_adjacency)
    assert connected["is_connected"] is True
    assert connected["diameter"] == 2  # 最长路径为2步
    assert 1 < connected["average_path_length"] < 2
    
    # 测试非连通图
    disconnected = NetworkMetrics.calculate_path_lengths(disconnected_graph_adjacency)
    assert disconnected["is_connected"] is False
    assert disconnected["diameter"] == 1  # 在各个连通分量内的最长路径
    assert disconnected["average_path_length"] == 1.0

def test_get_complete_stats(small_graph_adjacency):
    """测试完整统计信息的计算"""
    stats = NetworkMetrics.get_complete_stats(small_graph_adjacency)
    
    # 验证返回的所有必要字段
    required_keys = {
        "n_nodes", "n_edges", "max_degree", "min_degree",
        "mean_degree", "std_degree", "n_triangles",
        "global_clustering", "local_clustering",
        "average_path_length", "diameter", "is_connected"
    }
    assert set(stats.keys()) == required_keys
    
    # 验证基本统计值
    assert stats["n_nodes"] == 4
    assert stats["n_edges"] == 4
    assert stats["max_degree"] == 3
    assert stats["min_degree"] == 1
    assert stats["is_connected"] is True

def test_numerical_stability():
    """测试数值计算的稳定性"""
    # 测试大型稀疏图
    large_sparse = {i: {(i+1)%100} for i in range(100)}
    stats = NetworkMetrics.get_complete_stats(large_sparse)
    assert np.isfinite(stats["average_path_length"])
    assert np.isfinite(stats["global_clustering"])
    
    # 测试完全图
    complete = {i: set(j for j in range(5) if j != i) for i in range(5)}
    stats = NetworkMetrics.get_complete_stats(complete)
    assert np.isfinite(stats["average_path_length"])
    assert np.isfinite(stats["global_clustering"])

def test_edge_cases():
    """测试边界情况"""
    # 测试单节点图
    single_node = {0: set()}
    stats = NetworkMetrics.get_complete_stats(single_node)
    assert stats["n_nodes"] == 1
    assert stats["n_edges"] == 0
    assert stats["n_triangles"] == 0
    assert stats["global_clustering"] == 0
    
    # 测试空图
    empty = {}
    stats = NetworkMetrics.get_complete_stats(empty)
    assert stats["n_nodes"] == 0
    assert stats["n_edges"] == 0