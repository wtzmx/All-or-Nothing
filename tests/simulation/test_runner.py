import pytest
import numpy as np
import networkx as nx
from pathlib import Path
import shutil
import yaml
import json
import time
from typing import Dict

from src.simulation.runner import SimulationRunner
from src.simulation.config import ExperimentConfig

@pytest.fixture
def basic_config() -> Dict:
    """创建基础测试配置"""
    return {
        "experiment_name": "test_experiment",
        "network": {
            "type": "geometric",
            "n_agents": 10,
            "r_g": 0.3,
            "degree": 4  # 用于regular网络
        },
        "game": {
            "learning_rate": 0.3,
            "initial_belief": 0.5,
            "lambda_dist": "uniform",
            "lambda_params": {
                "low": 1.0,
                "high": 2.0
            }
        },
        "simulation": {
            "max_rounds": 1000,
            "save_interval": 10,  # 修改为更小的间隔以确保生成中间结果
            "convergence_threshold": 1e-4,
            "seed": 42
        }
    }

@pytest.fixture
def config(basic_config) -> ExperimentConfig:
    """创建ExperimentConfig实例"""
    return ExperimentConfig.from_dict(basic_config)

@pytest.fixture
def runner(config) -> SimulationRunner:
    """创建SimulationRunner实例"""
    return SimulationRunner(config)

@pytest.fixture(autouse=True)
def cleanup():
    """清理测试生成文件"""
    # 在测试前执行
    yield
    # 在测试后清理
    paths = [
        Path("data/results/test_experiment"),
        Path("data/intermediate/test_experiment"),
        Path("logs")
    ]
    for path in paths:
        if path.exists():
            shutil.rmtree(path)

def test_initialization(runner):
    """测试初始化过程"""
    runner.initialize()
    
    # 验证网络初始化
    assert runner.network is not None
    assert runner.network.number_of_nodes() == 10
    assert nx.is_connected(runner.network)
    
    # 验证邻接表
    assert len(runner.adjacency) == 10
    assert all(isinstance(neighbors, set) for neighbors in runner.adjacency.values())
    
    # 验证网络统计信息
    assert isinstance(runner.network_stats, dict)
    assert "n_nodes" in runner.network_stats
    assert "n_edges" in runner.network_stats
    
    # 验证博弈初始化
    assert runner.game is not None
    beliefs = runner.game.get_all_beliefs()
    assert len(beliefs) == 10
    assert all(0 <= b <= 1 for b in beliefs)

def test_network_types(basic_config):
    """测试不同类型的网络初始化"""
    # 测试几何网络
    config = ExperimentConfig.from_dict(basic_config)
    runner = SimulationRunner(config)
    runner.initialize()
    assert isinstance(runner.network, nx.Graph)
    assert "r_g" in runner.network_stats
    assert runner.network_stats["r_g"] == basic_config["network"]["r_g"]
    
    # 测���规则网络
    basic_config["network"]["type"] = "regular"
    config = ExperimentConfig.from_dict(basic_config)
    runner = SimulationRunner(config)
    runner.initialize()
    assert isinstance(runner.network, nx.Graph)
    assert all(d == basic_config["network"]["degree"] 
              for _, d in runner.network.degree())

def test_config_validation(basic_config):
    """测试配置验证"""
    # 测试无效网络类型
    invalid_config = basic_config.copy()
    invalid_config["network"] = invalid_config["network"].copy()
    invalid_config["network"]["type"] = "invalid"
    with pytest.raises(ValueError, match="Unknown network type"):
        ExperimentConfig.from_dict(invalid_config)

    # 测试无效的r_g值
    invalid_config = basic_config.copy()
    invalid_config["network"] = invalid_config["network"].copy()
    invalid_config["network"]["type"] = "geometric"
    invalid_config["network"]["r_g"] = -1
    with pytest.raises(ValueError, match=r"r_g must be between 0 and sqrt\(2\)"):
        ExperimentConfig.from_dict(invalid_config)

def test_run_simulation_flow(runner):
    """测试仿真流程的完整性"""
    runner.initialize()
    runner.run()
    
    # 验证数据收集
    assert len(runner.belief_history) > 0
    assert len(runner.action_history) > 0
    assert len(runner.payoff_history) > 0
    
    # 验证数据格式
    assert all(len(beliefs) == 10 for beliefs in runner.belief_history)
    assert all(isinstance(actions, dict) for actions in runner.action_history)
    assert all(isinstance(payoffs, dict) for payoffs in runner.payoff_history)
    
    # 验证收敛统计
    assert set(runner.convergence_stats.keys()) == {
        "converged", "rounds", "time", "final_beliefs", 
        "mean_belief", "std_belief"
    }

def test_data_saving_structure(runner):
    """测试数据保存的结构完整性"""
    runner.initialize()
    runner.run()
    
    results_dir = Path("data/results/test_experiment")
    expected_files = {
        "network.edgelist",
        "network_stats.json",
        "simulation_data.npz",
        "convergence_stats.json",
        "config.yaml"
    }
    
    actual_files = {f.name for f in results_dir.glob("*")}
    assert expected_files.issubset(actual_files)
    
    # 验证数据格式
    data = np.load(results_dir / "simulation_data.npz")
    assert set(data.files) == {"belief_history", "action_history", "payoff_history"}
    
    with open(results_dir / "network_stats.json") as f:
        stats = json.load(f)
    assert isinstance(stats, dict)
    assert "n_nodes" in stats
    assert "n_edges" in stats

def test_intermediate_results_saving(runner):
    """测试中间结果保存机制"""
    runner.initialize()
    runner.run()
    
    # 等待一小段时间确保文件写入完成
    time.sleep(1)
    
    intermediate_dir = Path("data/intermediate/test_experiment")
    intermediate_files = list(intermediate_dir.glob("intermediate_*.npz"))
    
    # 验证至少有一个中间结果文件
    assert len(intermediate_files) > 0
    
    # 验证文件格式和内容
    data = np.load(intermediate_files[0])
    assert set(data.files) == {"belief_history", "action_history", "payoff_history"}
    assert data["belief_history"].shape[1] == runner.config.network.n_agents

@pytest.mark.parametrize("n_agents,max_rounds,save_interval,expected_convergence", [
    (5, 200, 20, True),     # 小规模快速收敛
    (20, 1000, 50, True),   # 中等规模收敛
    (10, 30, 10, False)     # 轮次不足，预期不收敛（小于最小轮次要求）
])
def test_simulation_scales(basic_config, n_agents, max_rounds, save_interval, expected_convergence):
    """测试不同规模的仿真及收敛行为"""
    basic_config["network"]["n_agents"] = n_agents
    basic_config["simulation"]["max_rounds"] = max_rounds
    basic_config["simulation"]["save_interval"] = save_interval
    
    config = ExperimentConfig.from_dict(basic_config)
    runner = SimulationRunner(config)
    
    runner.initialize()
    runner.run()
    
    assert len(runner.game.get_all_beliefs()) == n_agents
    assert runner.convergence_stats["rounds"] <= max_rounds
    assert runner.convergence_stats["converged"] == expected_convergence

def test_numerical_stability_and_bounds(runner):
    """测试数值计算的稳定性和边界"""
    runner.initialize()
    runner.run()
    
    # 验证信念值始终在[0,1]范围内
    for beliefs in runner.belief_history:
        assert all(0 <= b <= 1 for b in beliefs)
        assert not any(np.isnan(b) for b in beliefs)
        assert not any(np.isinf(b) for b in beliefs)
    
    # 验证收敛统计
    final_beliefs = runner.convergence_stats["final_beliefs"]
    assert all(0 <= b <= 1 for b in final_beliefs)
    assert 0 <= runner.convergence_stats["mean_belief"] <= 1
    assert runner.convergence_stats["std_belief"] >= 0
    assert runner.convergence_stats["time"] > 0

def test_error_handling(runner):
    """测试错误处理机制"""
    # 测试未初始化就运行
    with pytest.raises(RuntimeError, match="Must call initialize()"):
        runner.run()