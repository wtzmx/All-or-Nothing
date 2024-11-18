import pytest
import yaml
import pandas as pd
from pathlib import Path
import shutil
from unittest.mock import Mock, patch

from experiments.experiment1.exp1_runner import ExperimentRunner

@pytest.fixture
def test_config():
    """创建测试配置"""
    return {
        "experiment_name": "test_experiment",
        "network": {
            "type": "geometric",
            "n_agents": 10,  # 减少智能体数量加快测试
            "radius_list": [0.3],  # 只测试一个半径值
            "seed": 42
        },
        "game": {
            "learning_rate": 0.3,
            "initial_belief": 0.5,
            "reward_function": {
                "type": "power",
                "exponent": 0.25
            },
            "lambda_distribution": {
                "type": "uniform",
                "params": {"low": 0.0, "high": 2.0}
            }
        },
        "simulation": {
            "max_rounds": 1000,  # 减少轮数加快测试
            "convergence_threshold": 0.0001,
            "n_trials": 2,  # 减少实验次数加快测试
            "save_interval": 100
        },
        "output": {
            "base_dir": "tests/temp_data",
            "save_network": True,
            "save_beliefs": True,
            "save_actions": True,
            "formats": ["csv", "pickle"]
        },
        "visualization": {
            "plot_types": ["tail_probability"],
            "figure_format": "png",
            "dpi": 300,
            "style": "default"
        },
        "parallel": {
            "enabled": False,  # 测试时禁用并行
            "n_processes": 1,
            "chunk_size": 1
        },
        "logging": {
            "level": "INFO",
            "save_to_file": True,
            "file_name": "test_experiment.log"
        },
        "analysis": {
            "compute_features": ["degree", "clustering", "triangles"],
            "convergence_metrics": ["time", "final_state"],
            "statistical_tests": ["ks_test"]
        }
    }

@pytest.fixture
def test_config_path(test_config, tmp_path):
    """创建临时配置文件"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    return config_path

@pytest.fixture
def runner(test_config_path):
    """创建实验运行器实例"""
    return ExperimentRunner(str(test_config_path))

def test_initialization(runner, test_config_path):
    """测试实验运行器初始化"""
    assert runner.config is not None
    assert isinstance(runner.output_dir, Path)
    assert runner.output_dir.exists()
    assert runner.logger is not None

def test_single_trial(runner):
    """测试单次实验运行"""
    radius = runner.config["network"]["radius_list"][0]
    result = runner._run_single_trial(radius, 0)
    
    # 检查结果格式
    assert isinstance(result, dict)
    required_keys = {
        "trial_id", "radius", "convergence_time", 
        "final_state", "network_features", "belief_history"
    }
    assert set(result.keys()) >= required_keys
    
    # 检查数值合理性
    assert result["trial_id"] == 0
    assert result["radius"] == radius
    assert result["convergence_time"] >= 0
    assert result["final_state"] in {"contribution", "defection", "not_converged"}
    assert isinstance(result["network_features"], dict)
    assert isinstance(result["belief_history"], list)

def test_parallel_trials(runner):
    """测试并行实验运行"""
    radius = runner.config["network"]["radius_list"][0]
    results = runner._run_parallel_trials(radius)
    
    # 检查结果数量
    assert len(results) == runner.config["simulation"]["n_trials"]
    
    # 检查每个结果的式
    for result in results:
        assert isinstance(result, dict)
        assert result["radius"] == radius

@pytest.mark.integration
def test_complete_experiment(runner):
    """测试完整实验流程"""
    # 运行实验
    runner.run_experiment()
    
    # 检查结果文件
    for radius in runner.config["network"]["radius_list"]:
        results_dir = runner.output_dir / f"radius_{radius}"
        assert results_dir.exists()
        
        if "csv" in runner.config["output"]["formats"]:
            assert (results_dir / "results.csv").exists()
        if "pickle" in runner.config["output"]["formats"]:
            assert (results_dir / "results.pkl").exists()

def test_save_results(runner, tmp_path):
    """测试结果保存功能"""
    # 创建测试数据
    test_results = [
        {
            "trial_id": 0,
            "radius": 0.3,
            "convergence_time": 100,
            "final_state": "contribution",
            "network_features": {"degree": 4},
            "belief_history": [[0.5, 0.5]]
        }
    ]
    
    # 保存结果
    runner._save_results(test_results, 0.3)
    
    # 检查保存的文件
    results_dir = runner.output_dir / "radius_0.3"
    if "csv" in runner.config["output"]["formats"]:
        csv_path = results_dir / "results.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        
    if "pickle" in runner.config["output"]["formats"]:
        pkl_path = results_dir / "results.pkl"
        assert pkl_path.exists()

@pytest.mark.parametrize("final_beliefs,expected_state", [
    ([0.9999, 0.9999], "contribution"),
    ([0.0001, 0.0001], "defection"),
    ([0.5, 0.5], "not_converged")
])
def test_convergence_detection(runner, final_beliefs, expected_state):
    """测试收敛状态检测"""
    with patch('experiments.experiment1.exp1_runner.PublicGoodsGame') as MockGame:
        # 创建mock对象
        mock_game = Mock()
        
        # 设置mock方法的返回值和行为
        def get_beliefs():
            return final_beliefs
        mock_game.get_beliefs = Mock(side_effect=get_beliefs)
        
        def check_convergence(threshold):
            if expected_state == "not_converged":
                return False
            return True
        mock_game.check_convergence = Mock(side_effect=check_convergence)
        
        # 设置play_round的返回值
        mock_game.play_round = Mock(return_value=({"0": "C"}, {"0": 1.0}))
        
        # 设置mock对象作为类的返回值
        MockGame.return_value = mock_game
        
        # 运行测试
        result = runner._run_single_trial(0.3, 0)
        
        # 验证结果
        assert result["final_state"] == expected_state
        
        # 验证mock方法被正确调用
        if expected_state != "not_converged":
            assert mock_game.check_convergence.call_count > 0
            assert mock_game.get_beliefs.call_count > 0
        assert mock_game.play_round.call_count > 0

def test_error_handling(test_config, tmp_path):
    """测试错误处理"""
    # 测试无效配置文件
    with pytest.raises(Exception):
        ExperimentRunner("nonexistent_config.yaml")
    
    # 测试无效输出目录
    test_config["output"]["base_dir"] = "/invalid/path"
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    with pytest.raises(Exception):
        ExperimentRunner(str(config_path))

def teardown_module(module):
    """清理测试产生的临时文件"""
    shutil.rmtree("tests/temp_data", ignore_errors=True) 