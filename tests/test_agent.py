import pytest
import numpy as np
from src.models.agent import Agent

@pytest.fixture
def basic_agent():
    """创建一个基础的Agent实例用于测试"""
    return Agent(agent_id=1, initial_belief=0.5, learning_rate=0.3)

def test_agent_initialization():
    """测试Agent的初始化"""
    agent = Agent(agent_id=1, initial_belief=0.7, learning_rate=0.2)
    
    assert agent.id == 1
    assert agent.belief == 0.7
    assert agent.alpha == 0.2
    assert len(agent.action_history) == 0
    assert len(agent.belief_history) == 0
    assert len(agent.lambda_history) == 0

def test_decision_making(basic_agent):
    """测试智能体的决策机制"""
    # 测试情况1：当期望收益大于1时选择贡献
    # 对于belief=0.5，group_size=3，需要lambda>4才会选择贡献
    # 因为 0.5^(3-1) * lambda >= 1
    action = basic_agent.decide_action(lambda_i=5.0, group_size=3)
    assert action == 'C'
    
    # 测试情况2：当期望收益小于1时选择背叛
    action = basic_agent.decide_action(lambda_i=3.0, group_size=3)
    assert action == 'D'
    
    # 测试历史记录是否正确更新
    assert len(basic_agent.action_history) == 2
    assert len(basic_agent.lambda_history) == 2

def test_belief_update(basic_agent):
    """测试信念更新机制"""
    initial_belief = basic_agent.get_belief()
    
    # 测试情况1：所有智能体都选择贡献
    basic_agent.update_belief(['C', 'C', 'C'])
    updated_belief = basic_agent.get_belief()
    expected_belief = initial_belief * (1 - 0.3) + 0.3 * 1.0
    assert abs(updated_belief - expected_belief) < 1e-10
    
    # 测试情况2：所有智能体都选择背叛
    basic_agent.update_belief(['D', 'D', 'D'])
    updated_belief = basic_agent.get_belief()
    expected_belief = expected_belief * (1 - 0.3) + 0.3 * 0.0
    assert abs(updated_belief - expected_belief) < 1e-10
    
    # 测试历史记录是否正确更新
    assert len(basic_agent.belief_history) == 2

def test_empty_observation_handling(basic_agent):
    """测试处理空观察列表的情况"""
    initial_belief = basic_agent.get_belief()
    basic_agent.update_belief([])
    assert basic_agent.get_belief() == initial_belief
    assert len(basic_agent.belief_history) == 0

def test_absorption_detection():
    """测试收敛状态检测"""
    # 创建三个具有不同信念的智能体
    agent_low = Agent(agent_id=1, initial_belief=0.0001)
    agent_mid = Agent(agent_id=2, initial_belief=0.5)
    agent_high = Agent(agent_id=3, initial_belief=0.9999)
    
    epsilon = 1e-4
    
    # 测试不同收敛状态
    assert agent_low.is_absorbed(epsilon) is False  # 收敛到背叛
    assert agent_mid.is_absorbed(epsilon) is None   # 未收敛
    assert agent_high.is_absorbed(epsilon) is True  # 收敛到贡献

def test_get_history(basic_agent):
    """测试历史记录获取"""
    # 进行一些操作来生成历史记录
    basic_agent.decide_action(lambda_i=2.0, group_size=3)
    basic_agent.update_belief(['C', 'D'])
    
    actions, beliefs, lambdas = basic_agent.get_history()
    
    assert len(actions) == 1
    assert len(beliefs) == 1
    assert len(lambdas) == 1
    assert isinstance(actions[0], str)
    assert isinstance(beliefs[0], float)
    assert isinstance(lambdas[0], float)

@pytest.mark.parametrize("group_size,lambda_value,initial_belief,expected_action", [
    (2, 3.0, 0.8, 'C'),    # 较小组容易达成合作
    (5, 3.0, 0.6, 'D'),    # 调整初始信念，使较大组确实难以达成合作
    (3, 10.0, 0.8, 'C'),   # 高收益促进合作
    (3, 0.5, 0.8, 'D')     # 低收益导致背叛
])
def test_group_size_and_lambda_effects(group_size, lambda_value, initial_belief, expected_action):
    """测试组大小和lambda值对决策的影响"""
    agent = Agent(agent_id=1, initial_belief=initial_belief)
    action = agent.decide_action(lambda_i=lambda_value, group_size=group_size)
    assert action == expected_action

def test_numerical_stability():
    """测试数值计算的稳定性"""
    agent = Agent(agent_id=1, initial_belief=0.5)
    
    # 测试极端lambda值
    action_high = agent.decide_action(lambda_i=1e6, group_size=3)
    action_low = agent.decide_action(lambda_i=1e-6, group_size=3)
    
    assert action_high in ['C', 'D']
    assert action_low in ['C', 'D']
    
    # 测试极端belief更新
    agent.update_belief(['C'] * 1000)  # 大量的观察
    assert 0 <= agent.get_belief() <= 1  # belief应该保持在有效范围内