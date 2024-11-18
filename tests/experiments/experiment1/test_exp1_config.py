import pytest
import yaml
import os
from pathlib import Path

@pytest.fixture
def config_path():
    """配置文件路径"""
    return Path("experiments/experiment1/exp1_config.yaml")

@pytest.fixture
def config_data(config_path):
    """加载配置文件数据"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_config_file_exists(config_path):
    """测试配置文件是否存在"""
    assert config_path.exists()

def test_required_sections(config_data):
    """测试必需的配置部分是否存在"""
    required_sections = {
        "experiment_name", "network", "game", 
        "simulation", "output", "visualization",
        "parallel", "logging", "analysis"
    }
    assert set(config_data.keys()) >= required_sections

def test_network_config(config_data):
    """测试网络配置的正确性"""
    network = config_data["network"]
    assert network["type"] == "geometric"
    assert network["n_agents"] == 50
    assert network["radius_list"] == [0.15, 0.2, 0.25, 0.3]
    assert isinstance(network["seed"], int)

def test_game_config(config_data):
    """测试博弈参数配置的正确性"""
    game = config_data["game"]
    assert game["learning_rate"] == 0.3
    assert game["initial_belief"] == 0.5
    
    # 测试奖励函数配置
    reward = game["reward_function"]
    assert reward["type"] == "power"
    assert reward["exponent"] == 0.25
    
    # 测试λ分布配置
    lambda_dist = game["lambda_distribution"]
    assert lambda_dist["type"] == "uniform"
    assert lambda_dist["params"]["low"] == 0.0
    assert lambda_dist["params"]["high"] == 2.0

def test_simulation_config(config_data):
    """测试仿真参数配置的正确性"""
    sim = config_data["simulation"]
    assert sim["max_rounds"] == 10_000_000
    assert sim["convergence_threshold"] == 0.0001
    assert sim["n_trials"] == 500
    assert isinstance(sim["save_interval"], int)

def test_output_config(config_data):
    """测试输出配置的正确性"""
    output = config_data["output"]
    assert isinstance(output["base_dir"], str)
    assert isinstance(output["save_network"], bool)
    assert isinstance(output["save_beliefs"], bool)
    assert isinstance(output["save_actions"], bool)
    assert "csv" in output["formats"]
    assert "pickle" in output["formats"]

def test_visualization_config(config_data):
    """测试可视化配置的正确性"""
    viz = config_data["visualization"]
    required_plots = {"tail_probability", "network_state", "belief_evolution"}
    assert set(viz["plot_types"]) >= required_plots
    assert viz["figure_format"] in ["png", "pdf", "svg"]
    assert isinstance(viz["dpi"], int)

def test_parallel_config(config_data):
    """测试并行计算配置的正确性"""
    parallel = config_data["parallel"]
    assert isinstance(parallel["enabled"], bool)
    assert parallel["n_processes"] > 0
    assert parallel["chunk_size"] > 0

def test_logging_config(config_data):
    """测试日志配置的正确性"""
    logging = config_data["logging"]
    assert logging["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]
    assert isinstance(logging["save_to_file"], bool)
    assert logging["file_name"].endswith(".log")

def test_analysis_config(config_data):
    """测试分析配置的正确性"""
    analysis = config_data["analysis"]
    required_features = {"degree", "clustering", "triangles"}
    required_metrics = {"time", "final_state", "belief_distribution"}
    
    assert set(analysis["compute_features"]) >= required_features
    assert set(analysis["convergence_metrics"]) >= required_metrics
    assert len(analysis["statistical_tests"]) > 0

def test_config_values_ranges(config_data):
    """测试配置值的范围约束"""
    # 测试网络参数范围
    assert all(0 < r < 1 for r in config_data["network"]["radius_list"])
    
    # 测试学习率范围
    assert 0 < config_data["game"]["learning_rate"] < 1
    
    # 测试初始信念范围
    assert 0 <= config_data["game"]["initial_belief"] <= 1
    
    # 测试收敛阈值范围
    assert config_data["simulation"]["convergence_threshold"] > 0
    
    # 测试并行进程数范围
    assert 0 < config_data["parallel"]["n_processes"] <= os.cpu_count()

def test_path_validity(config_data):
    """测试路径配置的有效性"""
    base_dir = Path(config_data["output"]["base_dir"])
    assert not base_dir.is_file()  # 确保不是文件
    
    # 测试日志文件路径
    log_file = config_data["logging"]["file_name"]
    assert log_file.endswith(".log") 