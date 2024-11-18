import pytest
import numpy as np
from src.networks.regular import CirculantGraph

@pytest.fixture
def small_graph():
    """创建一个小型规则图实例"""
    return CirculantGraph(n_nodes=6, neighbors=2)

@pytest.fixture
def medium_graph():
    """创建一个中等规模的规则图实例"""
    return CirculantGraph(n_nodes=10, neighbors=4)

@pytest.mark.parametrize("n_nodes,neighbors", [
    (-1, 2),      # 负数节点
    (5, 0),       # 零邻居
    (5, 5),       # 邻居数等于节点数
    (5, 3),       # 奇数邻居
    (5, 6)        # 邻居数大于节点数
])
def test_invalid_parameters(n_nodes, neighbors):
    """测试无效参数的处理"""
    with pytest.raises(ValueError):
        CirculantGraph(n_nodes=n_nodes, neighbors=neighbors)

def test_graph_initialization(small_graph):
    """测试图的初始化"""
    assert small_graph.N == 6
    assert small_graph.l == 2
    assert len(small_graph.adjacency) == 6
    assert isinstance(small_graph.adjacency, dict)
    assert all(isinstance(v, set) for v in small_graph.adjacency.values())

def test_degree_consistency(small_graph):
    """测试度数的一致性"""
    # 在规则图中，所有节点的度数应该相同
    degrees = [len(small_graph.adjacency[i]) for i in range(small_graph.N)]
    assert all(d == small_graph.l for d in degrees)
    assert small_graph.max_degree == small_graph.l
    assert small_graph.mean_degree == float(small_graph.l)

def test_neighbor_symmetry(small_graph):
    """测试邻居关系的对称性"""
    for i in range(small_graph.N):
        for j in small_graph.adjacency[i]:
            assert i in small_graph.adjacency[j]

def test_circular_structure(small_graph):
    """测试环形结构的正确性"""
    # 检查每个节点是否与其最近的l/2个邻居相连
    for i in range(small_graph.N):
        neighbors = small_graph.get_neighbors(i)
        k = small_graph.l // 2
        
        # 检查顺时针连接
        for j in range(1, k + 1):
            assert (i + j) % small_graph.N in neighbors
            
        # 检查逆时针连接
        for j in range(1, k + 1):
            assert (i - j) % small_graph.N in neighbors

def test_node_positions(small_graph):
    """测试节点位置的生成"""
    positions = small_graph.get_node_positions()
    
    # 检查位置数量
    assert len(positions) == small_graph.N
    
    # 检查是否在单位圆上
    for x, y in positions:
        # 允许小的数值误差
        assert abs(x*x + y*y - 1.0) < 1e-10

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
    
    # 验证边的数量
    assert len(edges) == (small_graph.N * small_graph.l) // 2

def test_closed_neighbors(small_graph):
    """测试闭邻居集合"""
    for i in range(small_graph.N):
        closed_neighbors = small_graph.get_closed_neighbors(i)
        open_neighbors = small_graph.get_neighbors(i)
        
        # 闭邻居应该包含节点自身
        assert i in closed_neighbors
        # 闭邻居数量应该比开邻居多1
        assert len(closed_neighbors) == len(open_neighbors) + 1
        # 开邻居应该是闭邻居的子集
        assert open_neighbors.issubset(closed_neighbors)

def test_triangle_counting(medium_graph):
    """测试三角形计数"""
    # 手动计算一个已知配置的三角形数量
    triangles = medium_graph._count_triangles()
    assert isinstance(triangles, int)
    assert triangles >= 0
    
    # 验证三角形计数的一致性
    stats = medium_graph.get_stats()
    assert stats["n_triangles"] == triangles

def test_stats_computation(small_graph):
    """测试网络统计信息的计算"""
    stats = small_graph.get_stats()
    
    # 检查统计信息的完整性
    required_keys = {
        "n_nodes", "n_neighbors", "n_edges", "max_degree",
        "mean_degree", "n_triangles", "is_regular"
    }
    assert set(stats.keys()) == required_keys
    
    # 验证统计值的合理性
    assert stats["n_nodes"] == small_graph.N
    assert stats["n_neighbors"] == small_graph.l
    assert stats["n_edges"] == (small_graph.N * small_graph.l) // 2
    assert stats["max_degree"] == small_graph.l
    assert stats["mean_degree"] == float(small_graph.l)
    assert isinstance(stats["n_triangles"], int)
    assert stats["is_regular"] is True

def test_different_sizes():
    """测试不同规模的图"""
    graphs = [
        CirculantGraph(n_nodes=10, neighbors=2),
        CirculantGraph(n_nodes=20, neighbors=2),
        CirculantGraph(n_nodes=30, neighbors=2)
    ]
    
    # 验证节点数和边数的关系
    for g in graphs:
        assert len(g.get_edges()) == (g.N * g.l) // 2
        
    # 验证度数保持不变
    degrees = [g.mean_degree for g in graphs]
    assert all(d == degrees[0] for d in degrees)