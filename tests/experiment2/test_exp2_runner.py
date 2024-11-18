# 添加必要的导入
import os
import yaml
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from experiments.experiment2.exp2_runner import ExperimentRunner
from src.networks.regular import CirculantGraph

class TestExp2Runner:
    """测试实验二运行器"""
    
    @pytest.fixture
    def config_path(self, tmp_path: Path) -> str:
        """创建测试用配置文件"""
        config = {
            "experiment_name": "test_regular_network",
            "network": {
                "type": "regular",
                "n_agents": 10,
                "l_values": [2, 4],
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
                    "params": {
                        "low": 0.0,
                        "high": 2.0
                    }
                }
            },
            "simulation": {
                "max_rounds": 1000,
                "convergence_threshold": 0.0001,
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
            "visualization": {
                "plot_types": ["tail_probability", "network_state", "belief_evolution"],
                "figure_format": "png",
                "dpi": 300
            },
            "parallel": {
                "enabled": False,
                "n_processes": 1,
                "chunk_size": 1
            },
            "logging": {
                "level": "INFO",
                "save_to_file": True,
                "file_name": "test_experiment2.log"
            },
            "analysis": {
                "compute_features": ["degree", "clustering", "triangles"],
                "convergence_metrics": ["time", "final_state"],
                "statistical_tests": ["ks_test"]
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
            
        return str(config_file)
    
    @pytest.fixture
    def runner(self, config_path: str) -> ExperimentRunner:
        """创建实验运行器实例"""
        return ExperimentRunner(config_path)
    
    def test_runner_initialization(self, runner: ExperimentRunner, config_path: str):
        """测试运行器初始化"""
        assert runner.config is not None, "配置未正确加载"
        assert isinstance(runner.output_dir, Path), "输出目录路径类型错误"
        assert runner.output_dir.exists(), "输出目录未创建"
        
    def test_single_trial_execution(self, runner: ExperimentRunner):
        """测试单次实验执行"""
        result = runner._run_single_trial(l_value=2, trial_id=0)
        
        # 检查结果字典的键
        required_keys = {
            "trial_id", "l_value", "convergence_time", 
            "final_state", "network_features", "belief_history"
        }
        assert set(result.keys()) == required_keys, "结果字典缺少必需的键"
        
        # 检查数值
        assert result["trial_id"] == 0, "trial_id不正确"
        assert result["l_value"] == 2, "l_value不正确"
        assert isinstance(result["convergence_time"], int), "收敛时间类型错误"
        assert result["final_state"] in {"contribution", "defection", "not_converged"}, "无效的最终状态"
        assert isinstance(result["network_features"], dict), "网络特征类型错误"
        assert isinstance(result["belief_history"], list), "信念历史类型错误"
        
    def test_parallel_trials_execution(self, runner: ExperimentRunner):
        """测试并行实验执行"""
        results = runner._run_parallel_trials(l_value=2)
        
        assert len(results) == runner.config["simulation"]["n_trials"], "实验次数不正确"
        assert all(isinstance(r, dict) for r in results), "结果格式错误"
        
    def test_results_saving(self, runner: ExperimentRunner, tmp_path: Path):
        """测试结果保存"""
        # 运行一组实验
        results = runner._run_parallel_trials(l_value=2)
        
        # 保存结果
        runner._save_results(results, l_value=2)
        
        # 检查文件是否创建
        results_dir = runner.output_dir / "l_value_2"
        assert results_dir.exists(), "结果目录未创建"
        
        if "csv" in runner.config["output"]["formats"]:
            assert (results_dir / "results.csv").exists(), "CSV文件未创建"
            
        if "pickle" in runner.config["output"]["formats"]:
            assert (results_dir / "results.pkl").exists(), "Pickle文件未创建"
            
    def test_complete_experiment_execution(self, runner: ExperimentRunner):
        """测试完整实验执行"""
        # 运行完整实验
        runner.run_experiment()
        
        # 检查每个l值的结果目录
        for l_value in runner.config["network"]["l_values"]:
            results_dir = runner.output_dir / f"l_value_{l_value}"
            assert results_dir.exists(), f"l={l_value}的结果目录未创建"
            
            if "csv" in runner.config["output"]["formats"]:
                assert (results_dir / "results.csv").exists(), f"l={l_value}的CSV文件未创建"
                
            if "pickle" in runner.config["output"]["formats"]:
                assert (results_dir / "results.pkl").exists(), f"l={l_value}的Pickle文件未创建"
                
    def test_network_generation(self, runner: ExperimentRunner):
        """测试网络生成"""
        l_value = 2
        network = CirculantGraph(
            n_nodes=runner.config["network"]["n_agents"],
            l=l_value,
            seed=runner.config["network"]["seed"]
        )
        
        # 检查网络属性
        assert network.n_nodes == runner.config["network"]["n_agents"], "节点数量不正确"
        assert network.l == l_value, "l值不正确"
        
        # 检查邻居数量
        for node in range(network.n_nodes):
            neighbors = network.get_closed_neighbors(node)
            assert len(neighbors) == l_value + 1, f"节点{node}的邻居数量不正确"
            
    def test_error_handling(self, runner: ExperimentRunner):
        """测试错误处理"""
        # 测试无效的l值
        with pytest.raises(ValueError):
            runner._run_single_trial(l_value=1, trial_id=0)  # l必须是偶数
            
        with pytest.raises(ValueError):
            runner._run_single_trial(
                l_value=runner.config["network"]["n_agents"], 
                trial_id=0
            )  # l必须小于节点数量