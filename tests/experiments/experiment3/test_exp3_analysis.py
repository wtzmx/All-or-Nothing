import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Any

from experiments.experiment3.exp3_analysis import (
    ExperimentAnalyzer, 
    AnalysisConfig
)

class TestExp3Analysis:
    """测试实验三分析器"""
    
    @pytest.fixture
    def sample_data(self, tmp_path: Path) -> Path:
        """创建测试数据"""
        # 创建数据目录结构
        data_dir = tmp_path / "test_data"
        
        # 为每种网络类型创建数据
        network_types = {
            "geometric": {"radius": 0.3},
            "regular": {"l": 2},
            "random": {"p": 0.3}
        }
        
        for network_type, params in network_types.items():
            # 创建网络类型目录
            network_dir = data_dir / network_type
            param_str = "_".join(f"{k}_{v}" for k, v in params.items())
            param_dir = network_dir / param_str
            param_dir.mkdir(parents=True)
            
            # 创建测试数据
            df = pd.DataFrame({
                "trial_id": range(100),
                "network_type": network_type,
                "params": [params] * 100,
                "convergence_time": np.random.randint(100, 1000, 100),
                "final_state": np.random.choice(
                    ["contribution", "defection", "not_converged"],
                    100
                ),
                "network_features": [
                    {
                        "degree_distribution": list(np.random.random(10)),
                        "clustering": 0.3,
                        "path_length": 2.5,
                        "centrality": 0.4,
                        "modularity": 0.6,
                        "assortativity": 0.2
                    }
                ] * 100,
                "belief_history": [
                    [list(np.random.random(10)) for _ in range(5)]
                ] * 100
            })
            
            # 保存数据
            df.to_csv(param_dir / "results.csv", index=False)
            
        return data_dir
        
    @pytest.fixture
    def analyzer(self, sample_data: Path) -> ExperimentAnalyzer:
        """创建分析器实例"""
        return ExperimentAnalyzer(str(sample_data))
        
    def test_initialization(self, analyzer: ExperimentAnalyzer):
        """测试分析器初始化"""
        assert analyzer.data_dir.exists()
        assert isinstance(analyzer.config, AnalysisConfig)
        assert analyzer.logger is not None
        assert len(analyzer.data) > 0
        
    def test_compute_tail_probabilities(self, analyzer: ExperimentAnalyzer):
        """测试尾概率分布计算"""
        tail_probs = analyzer.compute_tail_probabilities()
        
        # 检查结果结构
        assert isinstance(tail_probs, dict)
        for network_type in analyzer.data:
            assert network_type in tail_probs
            for param_str in analyzer.data[network_type]:
                assert param_str in tail_probs[network_type]
                result = tail_probs[network_type][param_str]
                assert "times" in result
                assert "probabilities" in result
                assert len(result["times"]) == len(result["probabilities"])
                
    def test_analyze_convergence_states(self, analyzer: ExperimentAnalyzer):
        """测试收敛状态分析"""
        results = analyzer.analyze_convergence_states()
        
        # 检查结果格式
        assert isinstance(results, pd.DataFrame)
        required_columns = {
            "network_type", "params", "total_trials",
            "contribution_ratio", "defection_ratio", "not_converged_ratio",
            "contribution_ci_lower", "contribution_ci_upper",
            "defection_ci_lower", "defection_ci_upper",
            "not_converged_ci_lower", "not_converged_ci_upper"
        }
        assert set(results.columns) >= required_columns
        
        # 修改这里的检查方式
        for col in ["contribution_ratio", "defection_ratio", "not_converged_ratio"]:
            assert (results[col] >= 0).all() and (results[col] <= 1).all(), f"{col} values out of range [0,1]"
        
    def test_analyze_network_features(self, analyzer: ExperimentAnalyzer):
        """测试网络特征分析"""
        results = analyzer.analyze_network_features()
        
        # 检查结果结构
        assert isinstance(results, dict)
        for feature in analyzer.config.network_feature_names:
            assert feature in results
            feature_stats = results[feature]
            
            for network_type in analyzer.data:
                assert network_type in feature_stats
                for param_str in analyzer.data[network_type]:
                    stats = feature_stats[network_type][param_str]
                    for state in ["contribution", "defection", "not_converged"]:
                        if state in stats:
                            state_stats = stats[state]
                            assert "mean" in state_stats
                            assert "std" in state_stats
                            assert "median" in state_stats
                            assert "count" in state_stats
                            
    def test_analyze_network_comparison(self, analyzer: ExperimentAnalyzer):
        """测试网络结构对比分析"""
        results = analyzer.analyze_network_comparison()
        
        # 检查结果结构
        assert isinstance(results, dict)
        required_metrics = {
            "convergence_speed", "cooperation_level", "stability"
        }
        assert set(results.keys()) >= required_metrics
        
        # 检查每个指标的数据框
        for metric, df in results.items():
            assert isinstance(df, pd.DataFrame)
            assert "network_type" in df.columns
            assert "params" in df.columns
            
    def test_perform_statistical_tests(self, analyzer: ExperimentAnalyzer):
        """测试统计检验"""
        results = analyzer.perform_statistical_tests()
        
        # 检查结果结构
        assert isinstance(results, dict)
        required_tests = {"ks_test", "mann_whitney", "kruskal_wallis"}
        assert set(results.keys()) >= required_tests
        
        # 检查每个检验的结果
        for test_name, df in results.items():
            assert isinstance(df, pd.DataFrame)
            if test_name in ["ks_test", "mann_whitney"]:
                assert "group1" in df.columns
                assert "group2" in df.columns
            assert "statistic" in df.columns
            assert "p_value" in df.columns
            
    def test_save_analysis_results(self, 
                                 analyzer: ExperimentAnalyzer,
                                 tmp_path: Path):
        """测试结果保存"""
        # 保存结果
        results_dir = tmp_path / "analysis"
        analyzer.save_analysis_results(results_dir)
        
        # 检查文件是否创建
        assert (results_dir / "analysis_results.pkl").exists()
        assert (results_dir / "convergence_states.csv").exists()
        
        # 检查网络对比结果
        for metric in ["convergence_speed", "cooperation_level", "stability"]:
            assert (results_dir / f"{metric}_comparison.csv").exists()
            
        # 检查统计检验结果
        for test in ["ks_test", "mann_whitney", "kruskal_wallis"]:
            assert (results_dir / f"{test}_results.csv").exists()
            
        # 检查保存的结果格式
        with open(results_dir / "analysis_results.pkl", 'rb') as f:
            results = pickle.load(f)
            assert isinstance(results, dict)
            required_keys = {
                "tail_probabilities", "convergence_states",
                "network_features", "network_comparison",
                "statistical_tests"
            }
            assert set(results.keys()) >= required_keys
            
    def test_error_handling(self, tmp_path: Path):
        """测试错误处理"""
        # 测试无效目录
        with pytest.raises(FileNotFoundError):
            ExperimentAnalyzer(str(tmp_path / "nonexistent"))
            
        # 测试数据不足
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        analyzer = ExperimentAnalyzer(str(empty_dir))
        assert len(analyzer.data) == 0
        
    def test_config_customization(self, sample_data: Path):
        """测试配置自定义"""
        custom_config = AnalysisConfig(
            min_samples=50,
            confidence_level=0.99,
            tail_bins=100,
            network_feature_names=["clustering", "path_length"]
        )
        
        analyzer = ExperimentAnalyzer(
            str(sample_data),
            config=custom_config
        )
        
        assert analyzer.config.min_samples == 50
        assert analyzer.config.confidence_level == 0.99
        assert analyzer.config.tail_bins == 100
        assert analyzer.config.network_feature_names == ["clustering", "path_length"]
