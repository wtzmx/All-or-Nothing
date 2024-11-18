import pytest
import yaml
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, Any

from experiments.experiment3.exp3_runner import ExperimentRunner

class TestExp3Runner:
    """测试实验三运行器"""
    
    @pytest.fixture
    def config_path(self, tmp_path: Path) -> Path:
        """创建测试配置文件"""
        config = {
            "experiment_name": "test_network_comparison",
            "networks": {
                "geometric": {
                    "enabled": True,
                    "n_agents": 10,
                    "radius_list": [0.3]
                },
                "regular": {
                    "enabled": True,
                    "n_agents": 10,
                    "l_values": [2]
                },
                "random": {
                    "enabled": True,
                    "n_agents": 10,
                    "p_values": [0.3]
                },
                "small_world": {
                    "enabled": False,
                    "n_agents": 10,
                    "k": 4,
                    "p_values": [0.3]
                },
                "scale_free": {
                    "enabled": False,
                    "n_agents": 10,
                    "m_values": [2]
                },
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
                "max_rounds": 1000,
                "convergence_threshold": 0.01,
                "n_trials": 2,
                "save_interval": 100
            },
            "output": {
                "base_dir": str(tmp_path / "test_output"),
                "save_network": True,
                "save_beliefs": True,
                "save_actions": True,
                "formats": ["csv", "pickle"]
            },
            "parallel": {
                "enabled": False,
                "n_processes": 2,
                "chunk_size": 1
            },
            "logging": {
                "level": "INFO",
                "save_to_file": False,
                "file_name": "test.log"
            }
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            
        return config_path
        
    @pytest.fixture
    def runner(self, config_path: Path) -> ExperimentRunner:
        """创建实验运行器实例"""
        return ExperimentRunner(str(config_path))
        
    def test_initialization(self, runner: ExperimentRunner, config_path: Path):
        """测试运行器初始化"""
        assert runner.config is not None
        assert isinstance(runner.output_dir, Path)
        assert runner.output_dir.exists()
        assert runner.logger is not None
        
    def test_create_network(self, runner: ExperimentRunner):
        """测试网络创建"""
        # 测试几何网络
        network = runner._create_network(
            "geometric", 
            {"radius": 0.3}, 
            trial_id=0
        )
        assert network.N == runner.config["networks"]["geometric"]["n_agents"]
        
        # 测试规则网络
        network = runner._create_network(
            "regular",
            {"l": 2},
            trial_id=0
        )
        assert network.N == runner.config["networks"]["regular"]["n_agents"]
        
        # 测试随机网络
        network = runner._create_network(
            "random",
            {"p": 0.3},
            trial_id=0
        )
        assert network.N == runner.config["networks"]["random"]["n_agents"]
        
        # 测试无效网络类型
        with pytest.raises(ValueError):
            runner._create_network("invalid", {}, 0)
            
    def test_run_single_trial(self, runner: ExperimentRunner):
        """测试单次实验运行"""
        # 运行几何网络实验
        result = runner._run_single_trial(
            "geometric",
            {"radius": 0.3},
            trial_id=0
        )
        
        # 检查结果格式
        assert isinstance(result, dict)
        assert "trial_id" in result
        assert "network_type" in result
        assert "params" in result
        assert "convergence_time" in result
        assert "final_state" in result
        assert "network_features" in result
        assert "belief_history" in result
        
        # 检查数值范围
        assert result["convergence_time"] >= 0
        assert result["final_state"] in ["contribution", "defection", "not_converged"]
        
    def test_run_parallel_trials(self, runner: ExperimentRunner):
        """测试并行实验运行"""
        results = runner._run_parallel_trials(
            "geometric",
            {"radius": 0.3}
        )
        
        # 检查结果数量
        assert len(results) == runner.config["simulation"]["n_trials"]
        
        # 检查每个结果的格式
        for result in results:
            assert isinstance(result, dict)
            assert "trial_id" in result
            assert "network_type" in result
            assert "params" in result
            
    def test_save_results(self, runner: ExperimentRunner, tmp_path: Path):
        """测试结果保存"""
        # 创建测试结果
        results = [
            {
                "trial_id": 0,
                "network_type": "geometric",
                "params": {"radius": 0.3},
                "convergence_time": 100,
                "final_state": "contribution",
                "network_features": {"n_nodes": 10},
                "belief_history": [[0.5] * 10]
            }
        ]
        
        # 保存结果
        runner._save_results(
            results,
            "geometric",
            {"radius": 0.3}
        )
        
        # 检查文件是否创建
        results_dir = runner.output_dir / "geometric" / "radius_0.3"
        assert results_dir.exists()
        assert (results_dir / "results.csv").exists()
        assert (results_dir / "results.pkl").exists()
        
    def test_run_experiment(self, runner: ExperimentRunner):
        """测试完整实验运行"""
        # 运行实验
        runner.run_experiment()
        
        # 检查是否为每个启用的网络类型创建了结果目录
        for network_type, config in runner.config["networks"].items():
            if network_type != "seed" and config["enabled"]:
                network_dir = runner.output_dir / network_type
                assert network_dir.exists()
                
    def test_error_handling(self, tmp_path: Path):
        """测试错误处理"""
        # 测试无效配置文件
        with pytest.raises(FileNotFoundError):
            ExperimentRunner(str(tmp_path / "nonexistent.yaml"))
            
        # 测试无效网络类型
        runner = ExperimentRunner(str(tmp_path / "test_config.yaml"))
        with pytest.raises(ValueError):
            runner._create_network("invalid", {}, 0)
            
    @pytest.fixture(autouse=True)
    def cleanup(self, runner: ExperimentRunner):
        """测试清理"""
        yield
        # 清理测试输出目录
        if runner.output_dir.exists():
            shutil.rmtree(runner.output_dir)
