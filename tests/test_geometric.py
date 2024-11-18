import pytest
import numpy as np
from src.networks.geometric import RandomGeometricGraph, Point

@pytest.fixture
def small_graph():
    """创建一个小型确定性图实例"""
    return RandomGeometricGraph(n_nodes=5, radius=0.5, seed=42)

@pytest.fixture
def point_pairs():
    """创建用于测试的点对"""
    return [
        (Point(0, 0, 0), Point(0.3, 0.4, 1)),  # 距离 0.5
        (Point(0, 0, 0), Point(1, 1, 1)),      # 距离 √2 ≈ 1.414
        (Point(0, 0, 0), Point(0, 0.1, 1))     # 距离 0.1
    ]

def test_point_distance():
    """测试点之间距离的计算"""
    p1 = Point(0, 0, 0)
    p2 = Point(3, 4, 1)
    assert p1.distance_to(p2) == 5.0
    assert p2.distance_to(p1) == 5.0
    assert p1.distance_to(p1) == 0.0

def test_graph_initialization(small_graph):
    """测试图的初始化"""
    assert small_graph.N == 5
    assert small_graph.radius == 0.5
    assert len(small_graph.points) == 5
    assert isinstance(small_graph.adjacency, dict)
    assert all(isinstance(v, set) for v in small_graph.adjacency.values())

def test_point_generation():
    """测试点的生成"""
    graph = RandomGeometricGraph(n_nodes=100, radius=0.3, seed=42)
    points = graph.get_node_positions()
    
    # 检查点的数量
    assert len(points) == 100
    
    # 检查所有坐标是否在[0,1]²范围内
    assert all(0 <= x <= 1 and 0 <= y <= 1 
              for x, y in points)
    
    # 检查随机种子是否生效（结果可复现）
    graph2 = RandomGeometricGraph(n_nodes=100, radius=0.3, seed=42)
    points2 = graph2.get_node_positions()
    assert points == points2

def test_edge_generation(small_graph):
    """测试边的生成"""
    edges = small_graph.get_edges()
    
    # 检查边的格式
    assert all(isinstance(e, tuple) and len(e) == 2 
              for e in edges)
    
    # 检查边的节点索引是否有效
    assert all(0 <= i < small_graph.N and 0 <= j < small_graph.N 
              for i, j in edges)
    
    # 检查是否存在自环
    assert all(i != j for i, j in edges)
    
    # 验证边是否对称
    for i, j in edges:
        assert j in small_graph.adjacency[i]
        assert i in small_graph.adjacency[j]

def test_connectivity(small_graph):
    """测试图的连通性"""
    assert small_graph.is_connected()
    
    # 测试边界情况
    single_node = RandomGeometricGraph(n_nodes=1, radius=0.5, seed=42)
    assert single_node.is_connected()

def test_degree_distribution(small_graph):
    """测试度分布的计算"""
    degrees = small_graph.degrees
    
    # 检查度的基本属性
    assert len(degrees) == small_graph.N
    assert all(d >= 0 for d in degrees.values())
    assert all(isinstance(d, int) for d in degrees.values())
    
    # 验证最大度和平均度
    assert small_graph.max_degree == max(degrees.values())
    assert abs(small_graph.mean_degree - 
              sum(degrees.values()) / small_graph.N) < 1e-10

def test_triangle_counting(small_graph):
    """测试三角形计数"""
    n_triangles = small_graph.triangles
    
    # 基本属性检查
    assert isinstance(n_triangles, int)
    assert n_triangles >= 0
    
    # 验证三角形的存在性
    if n_triangles > 0:
        # 应该能找到至少一个三角形
        found_triangle = False
        for i in range(small_graph.N):
            neighbors_i = small_graph.adjacency[i]
            for j in neighbors_i:
                if j > i:
                    common_neighbors = neighbors_i & small_graph.adjacency[j]
                    for k in common_neighbors:
                        if k > j:
                            found_triangle = True
                            break
                if found_triangle:
                    break
            if found_triangle:
                break
        assert found_triangle

def test_neighbor_queries(small_graph):
    """测试邻居查询功能"""
    node_id = 0
    
    # 测试开邻居集
    neighbors = small_graph.get_neighbors(node_id)
    assert isinstance(neighbors, set)
    assert node_id not in neighbors
    assert all(0 <= n < small_graph.N for n in neighbors)
    
    # 测试闭邻居集
    closed_neighbors = small_graph.get_closed_neighbors(node_id)
    assert isinstance(closed_neighbors, set)
    assert node_id in closed_neighbors
    assert neighbors.issubset(closed_neighbors)
    assert len(closed_neighbors) == len(neighbors) + 1
    
    # 测试无效节点
    with pytest.raises(ValueError):
        small_graph.get_neighbors(small_graph.N + 1)

@pytest.mark.parametrize("n_nodes,radius", [
    (0, 0.5),     # 无效的节点数
    (10, 0),      # 无效的半径(0)
    (10, 1),      # 无效的半径(1)
    (10, -0.5),   # 无效的半径(负数)
    (10, 1.5),    # 无效的半径(>1)
])
def test_invalid_parameters(n_nodes, radius):
    """测试无效参数的处理"""
    with pytest.raises(ValueError):
        RandomGeometricGraph(n_nodes=n_nodes, radius=radius)

def test_stats_computation(small_graph):
    """测试网络统计信息的计算"""
    stats = small_graph.get_stats()
    
    # 检查统计信息的完整性
    required_keys = {
        "n_nodes", "radius", "n_edges", "max_degree",
        "mean_degree", "n_triangles", "is_connected"
    }
    assert set(stats.keys()) == required_keys
    
    # 验证统计值的合理性
    assert stats["n_nodes"] == small_graph.N
    assert stats["radius"] == small_graph.radius
    assert isinstance(stats["n_edges"], int)
    assert stats["max_degree"] <= small_graph.N - 1
    assert 0 <= stats["mean_degree"] <= small_graph.N - 1
    assert isinstance(stats["n_triangles"], int)
    assert isinstance(stats["is_connected"], bool)

def test_reproducibility():
    """测试结果的可复现性"""
    params = {"n_nodes": 20, "radius": 0.3, "seed": 42}
    
    # 创建两个相同参数的图
    graph1 = RandomGeometricGraph(**params)
    graph2 = RandomGeometricGraph(**params)
    
    # 比较节点位置
    pos1 = graph1.get_node_positions()
    pos2 = graph2.get_node_positions()
    assert pos1 == pos2
    
    # 比较边集合
    edges1 = set(graph1.get_edges())
    edges2 = set(graph2.get_edges())
    assert edges1 == edges2
    
    # 比较统计信息
    stats1 = graph1.get_stats()
    stats2 = graph2.get_stats()
    assert stats1 == stats2

def test_different_radii():
    """测试不同半径对图结构的影响"""
    n_nodes = 20
    seed = 42
    
    # 创建不同半径的图
    small_r = RandomGeometricGraph(n_nodes=n_nodes, radius=0.2, seed=seed)
    large_r = RandomGeometricGraph(n_nodes=n_nodes, radius=0.4, seed=seed)
    
    # 较大半径的图应该有更多的边
    assert (sum(len(v) for v in large_r.adjacency.values()) >
            sum(len(v) for v in small_r.adjacency.values()))
    
    # 较大半径的图应该有更高的平均度
    assert large_r.mean_degree > small_r.mean_degree