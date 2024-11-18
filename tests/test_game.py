import pytest
import numpy as np
from src.models.game import PublicGoodsGame
from typing import Set, Dict

@pytest.fixture
def basic_game():
    """创建一个基础的博弈实例"""
    return PublicGoodsGame(
        n_agents=5,
        learning_rate=0.3,
        initial_belief=0.5,
        lambda_dist="uniform",
        lambda_params={"low": 0.0, "high": 2.0}
    )

@pytest.fixture
def fixed_lambda_game(monkeypatch):
    """创建一个λ值固定的博弈实例，用于确定性测试"""
    class MockGame(PublicGoodsGame):
        def _generate_lambda(self):
            return 1.5
            
    return MockGame(
        n_agents=4,
        learning_rate=0.3,
        initial_belief=0.5
    )

def test_game_initialization(basic_game):
    """测试博弈环境的初始化"""
    assert len(basic_game.agents) == 5
    assert basic_game.lambda_dist == "uniform"
    assert basic_game.lambda_params == {"low": 0.0, "high": 2.0}
    assert basic_game.round_count == 0
    
    # 检查所有智能体的初始信念
    beliefs = basic_game.get_all_beliefs()
    assert all(belief == 0.5 for belief in beliefs)
    
    # 检查智能体ID是否正确分配
    for i, agent in enumerate(basic_game.agents):
        assert agent.id == i

def test_lambda_generation(basic_game):
    """测试λ值的生成"""
    # 生成多个λ值并检查它们是否在指定范围内
    lambda_values = [basic_game._generate_lambda() for _ in range(100)]
    assert all(0.0 <= x <= 2.0 for x in lambda_values)
    
    # 测试正态分布
    normal_game = PublicGoodsGame(
        n_agents=5,
        lambda_dist="normal",
        lambda_params={"mean": 1.0, "std": 0.2}
    )
    lambda_values = [normal_game._generate_lambda() for _ in range(100)]
    assert 0.5 < np.mean(lambda_values) < 1.5  # 均值应该在1.0附近
    
    # 测试无效分布
    with pytest.raises(ValueError):
        invalid_game = PublicGoodsGame(
            n_agents=5,
            lambda_dist="invalid"
        )
        invalid_game._generate_lambda()

def test_all_contribute_scenario(fixed_lambda_game):
    """测试所有智能体都选择贡献的情况"""
    # 选择一个较小的群体，使得贡献是优势策略
    players = {0, 1, 2}  # 3个玩家
    
    # 设置较高的初始信念以促进合作
    for agent in fixed_lambda_game.agents:
        agent.belief = 0.9
    
    # 执行一轮博弈
    actions, payoffs = fixed_lambda_game.play_round(players)
    
    # 验证结果
    assert all(action == 'C' for action in actions.values())
    assert all(payoff == 1.5 for payoff in payoffs.values())

def test_all_defect_scenario(fixed_lambda_game):
    """测试所有智能体都选择背叛的情况"""
    # 选择一个较大的群体，使得背叛是优势策略
    players = {0, 1, 2, 3}  # 4个玩家
    
    # 设置较低的初始信念以促进背叛
    for agent in fixed_lambda_game.agents:
        agent.belief = 0.1
    
    # 执行一轮博弈
    actions, payoffs = fixed_lambda_game.play_round(players)
    
    # 验证结果
    assert all(action == 'D' for action in actions.values())
    assert all(payoff == 1.0 for payoff in payoffs.values())

def test_mixed_scenario(fixed_lambda_game):
    """测试混合策略的情况"""
    players = {0, 1, 2}
    
    # 设置不同的信念
    fixed_lambda_game.agents[0].belief = 0.9  # 倾向于贡献
    fixed_lambda_game.agents[1].belief = 0.9  # 倾向于贡献
    fixed_lambda_game.agents[2].belief = 0.1  # 倾向于背叛
    
    # 执行一轮博弈
    actions, payoffs = fixed_lambda_game.play_round(players)
    
    # 检查至少有一个贡献者获得0收益
    assert any(payoff == 0.0 for payoff in payoffs.values())
    # 检查背叛者获得1的收益
    assert any(payoff == 1.0 for payoff in payoffs.values())

def test_belief_updates(fixed_lambda_game):
    """测试信念更新机制"""
    players = {0, 1, 2}
    initial_beliefs = [agent.get_belief() for agent in fixed_lambda_game.agents]
    
    # 执行多轮博弈
    for _ in range(5):
        fixed_lambda_game.play_round(players)
    
    # 检查信念是否发生变化
    updated_beliefs = fixed_lambda_game.get_all_beliefs()
    assert any(abs(b1 - b2) > 1e-10 
              for b1, b2 in zip(initial_beliefs, updated_beliefs))

def test_convergence_detection():
    """测试收敛检测"""
    game = PublicGoodsGame(n_agents=3)
    
    # 测试收敛到贡献
    for agent in game.agents:
        agent.belief = 0.9999
    assert game.is_converged(epsilon=1e-3)
    
    # 测试收敛到背叛
    for agent in game.agents:
        agent.belief = 0.0001
    assert game.is_converged(epsilon=1e-3)
    
    # 测试未收敛状态
    for agent in game.agents:
        agent.belief = 0.5
    assert not game.is_converged(epsilon=1e-3)

def test_get_stats(basic_game):
    """测试统计信息获取"""
    # 设置不同的信念值
    beliefs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for agent, belief in zip(basic_game.agents, beliefs):
        agent.belief = belief
    
    stats = basic_game.get_stats()
    assert abs(stats["mean_belief"] - np.mean(beliefs)) < 1e-10
    assert abs(stats["min_belief"] - min(beliefs)) < 1e-10
    assert abs(stats["max_belief"] - max(beliefs)) < 1e-10
    assert stats["round"] == 0

@pytest.mark.parametrize("group_size", [2, 3, 4, 5])
def test_different_group_sizes(basic_game, group_size):
    """测试不同大小的群体"""
    players = set(range(group_size))
    actions, payoffs = basic_game.play_round(players)
    
    assert len(actions) == group_size
    assert len(payoffs) == group_size
    assert all(isinstance(a, str) for a in actions.values())
    assert all(isinstance(p, float) for p in payoffs.values())

def test_invalid_inputs():
    """测试无效输入的处理"""
    # 测试无效的智能体数量
    with pytest.raises(ValueError):
        PublicGoodsGame(n_agents=0)
        
    # 测试无效的学习率
    with pytest.raises(ValueError):
        PublicGoodsGame(n_agents=3, learning_rate=1.5)
        
    # 测试无效的初始信念
    with pytest.raises(ValueError):
        PublicGoodsGame(n_agents=3, initial_belief=1.5)

def test_numerical_stability(basic_game):
    """测试数值计算的稳定性"""
    players = {0, 1, 2}
    
    # 执行大量轮次的博弈
    for _ in range(1000):
        actions, payoffs = basic_game.play_round(players)
        
        # 验证数值是否在有效范围内
        beliefs = basic_game.get_all_beliefs()
        assert all(0 <= b <= 1 for b in beliefs)
        assert all(p >= 0 for p in payoffs.values())
        assert all(a in ['C', 'D'] for a in actions.values())

def test_sequential_rounds(basic_game):
    """测试连续多轮博弈"""
    players = {0, 1, 2}
    round_history = []
    
    # 执行多轮博弈并记录结果
    for _ in range(10):
        actions, payoffs = basic_game.play_round(players)
        round_history.append((actions.copy(), payoffs.copy()))
    
    # 验证轮次计数
    assert basic_game.round_count == 10
    
    # 验证每轮的结果格式
    for actions, payoffs in round_history:
        assert len(actions) == len(players)
        assert len(payoffs) == len(players)
        assert all(a in ['C', 'D'] for a in actions.values())
        assert all(isinstance(p, float) for p in payoffs.values())