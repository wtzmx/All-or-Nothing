import pytest
import tempfile
from pathlib import Path
import yaml
import json
import numpy as np
from src.simulation.config import (
    NetworkConfig,
    GameConfig,
    SimulationConfig,
    ExperimentConfig
)

# NetworkConfig测试
def test_network_config_geometric():
    """测试几何网络配置"""
    # 有效配置
    config = NetworkConfig(type="geometric", n_agents=50, r_g=0.3)
    config.validate()  # 不应抛出异常
    
    # 无效的n_agents
    with pytest.raises(ValueError, match="Number of agents must be positive"):
        NetworkConfig(type="geometric", n_agents=0, r_g=0.3).validate()
        
    # 缺少r_g
    with pytest.raises(ValueError, match="r_g must be specified"):
        NetworkConfig(type="geometric", n_agents=50).validate()
        
    # r_g范围无效
    with pytest.raises(ValueError, match="r_g must be between 0 and sqrt"):
        NetworkConfig(type="geometric", n_agents=50, r_g=2.0).validate()

def test_network_config_regular():
    """测试规则网络配置"""
    # 有效配置
    config = NetworkConfig(type="regular", n_agents=50, degree=4)
    config.validate()  # 不应抛出异常
    
    # 无效的degree
    with pytest.raises(ValueError, match="Invalid degree"):
        NetworkConfig(type="regular", n_agents=50, degree=50).validate()
        
    # 奇数degree
    with pytest.raises(ValueError, match="degree must be even"):
        NetworkConfig(type="regular", n_agents=50, degree=3).validate()
        
    # 缺少degree
    with pytest.raises(ValueError, match="degree must be specified"):
        NetworkConfig(type="regular", n_agents=50).validate()

# GameConfig测试
def test_game_config_uniform():
    """测试均匀分布的游戏配置"""
    # 有效配置
    config = GameConfig(
        learning_rate=0.3,
        initial_belief=0.5,
        lambda_dist="uniform",
        lambda_params={"low": 0.0, "high": 2.0}
    )
    config.validate()  # 不应抛出异常
    
    # 无效的学习率
    with pytest.raises(ValueError, match="Learning rate must be between"):
        GameConfig(learning_rate=1.5).validate()
        
    # 无效的初始信念
    with pytest.raises(ValueError, match="Initial belief must be between"):
        GameConfig(initial_belief=1.5).validate()
        
    # 缺少lambda参数
    with pytest.raises(ValueError, match="requires 'low' and 'high'"):
        GameConfig(lambda_params={}).validate()
        
    # 无效的lambda范围
    with pytest.raises(ValueError, match="'low' must be less than 'high'"):
        GameConfig(lambda_params={"low": 2.0, "high": 1.0}).validate()

def test_game_config_normal():
    """测试正态分布的游戏配置"""
    # 有效配置
    config = GameConfig(
        lambda_dist="normal",
        lambda_params={"mean": 1.0, "std": 0.5}
    )
    config.validate()  # 不应抛出异常
    
    # 缺少参数
    with pytest.raises(ValueError, match="requires 'mean' and 'std'"):
        GameConfig(
            lambda_dist="normal",
            lambda_params={"mean": 1.0}
        ).validate()
        
    # 无效的标准差
    with pytest.raises(ValueError, match="Standard deviation must be positive"):
        GameConfig(
            lambda_dist="normal",
            lambda_params={"mean": 1.0, "std": 0.0}
        ).validate()

# SimulationConfig测试
def test_simulation_config():
    """测试仿真配置"""
    # 有效配置
    config = SimulationConfig(
        max_rounds=1000000,
        convergence_threshold=1e-4,
        save_interval=1000
    )
    config.validate()  # 不应抛出异常
    
    # 无效的最大轮次
    with pytest.raises(ValueError, match="Maximum rounds must be positive"):
        SimulationConfig(max_rounds=0).validate()
        
    # 无效的收敛阈值
    with pytest.raises(ValueError, match="Convergence threshold must be positive"):
        SimulationConfig(convergence_threshold=0).validate()
        
    # 无效的保存间隔
    with pytest.raises(ValueError, match="Save interval must be positive"):
        SimulationConfig(save_interval=0).validate()
        
    # 保存间隔大于最大轮次
    with pytest.raises(ValueError, match="Save interval cannot be larger"):
        SimulationConfig(max_rounds=1000, save_interval=2000).validate()

# ExperimentConfig测试
@pytest.fixture
def valid_config_dict():
    """创建有效的配置字典"""
    return {
        "network": {
            "type": "geometric",
            "n_agents": 50,
            "r_g": 0.3
        },
        "game": {
            "learning_rate": 0.3,
            "initial_belief": 0.5,
            "lambda_dist": "uniform",
            "lambda_params": {"low": 0.0, "high": 2.0}
        },
        "simulation": {
            "max_rounds": 1000000,
            "convergence_threshold": 1e-4,
            "save_interval": 1000
        },
        "experiment_name": "test_experiment"
    }

def test_experiment_config_from_dict(valid_config_dict):
    """测试从字典创建实验配置"""
    config = ExperimentConfig.from_dict(valid_config_dict)
    config.validate()  # 不应抛出异常
    
    # 验证转换回字典
    config_dict = config.to_dict()
    assert config_dict["network"]["type"] == "geometric"
    assert config_dict["network"]["r_g"] == 0.3
    assert config_dict["game"]["learning_rate"] == 0.3
    assert config_dict["simulation"]["max_rounds"] == 1000000

def test_experiment_config_file_operations(valid_config_dict):
    """测试配置文件操作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试YAML文件
        yaml_path = Path(tmpdir) / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(valid_config_dict, f)
        
        config = ExperimentConfig.from_file(yaml_path)
        config.validate()
        
        # 测试JSON文件
        json_path = Path(tmpdir) / "config.json"
        with open(json_path, 'w') as f:
            json.dump(valid_config_dict, f)
        
        config = ExperimentConfig.from_file(json_path)
        config.validate()
        
        # 测试保存配置
        save_path = Path(tmpdir) / "saved_config.yaml"
        config.save(save_path)
        assert save_path.exists()
        
        # 测试无效文件格式
        invalid_path = Path(tmpdir) / "config.txt"
        with pytest.raises(ValueError, match="Unsupported file format"):
            config.save(invalid_path)

def test_experiment_config_validation():
    """测试实验配置验证"""
    # 空实验名称
    with pytest.raises(ValueError, match="Experiment name cannot be empty"):
        ExperimentConfig(
            network=NetworkConfig(type="geometric", r_g=0.3),
            game=GameConfig(),
            simulation=SimulationConfig(),
            experiment_name=""
        ).validate()

def test_config_default_values():
    """测试配置默认值"""
    game_config = GameConfig()
    assert game_config.learning_rate == 0.3
    assert game_config.initial_belief == 0.5
    assert game_config.lambda_dist == "uniform"
    assert game_config.lambda_params == {"low": 0.0, "high": 2.0}
    
    sim_config = SimulationConfig()
    assert sim_config.max_rounds == 10_000_000
    assert sim_config.convergence_threshold == 1e-4
    assert sim_config.save_interval == 1000
    assert sim_config.seed is None