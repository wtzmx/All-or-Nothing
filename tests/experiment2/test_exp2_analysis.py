import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Any

from experiments.experiment2.exp2_analysis import ExperimentAnalyzer, AnalysisConfig

class TestExp2Analysis:
    """测试实验二分析模块"""
    
    @pytest.fixture
    def sample_data(self, tmp_path: Path) -> Path:
        """创建测试用数据"""
        # 创建测试数据目录
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # 为不同的l值创建数据
        l_values = [2, 4, 6, 8]
        for l in l_values:
            l_dir = data_dir / f"l_value_{l}"
            l_dir.mkdir()
            
            # 创建测试数据
            df = pd.DataFrame({
                "trial_id": range(150),
                "l_value": l,
                "convergence_time": np.random.randint(100, 10000, 150),
                "final_state": np.random.choice(
                    ["contribution", "defection", "not_converged"],
                    150
                ),
                "network_features": [
                    {
                        "degree": l,
                        "n_triangles": l * (l-1) // 2,
                        "clustering_coefficient": 1.0 if l > 1 else 0.0,
                        "average_path_length": 2.5,
                        "diameter": 5
                    }
                    for _ in range(150)
                ],
                "belief_history": [
                    [[0.5] * 50] * 10
                    for _ in range(150)
                ]
            })
            
            # 保存数据
            df.to_csv(l_dir / "results.csv", index=False)
            
        return data_dir
    
    @pytest.fixture
    def analyzer(self, sample_data: Path) -> ExperimentAnalyzer:
        """创建分析器实例"""
        return ExperimentAnalyzer(str(sample_data))
    
    def test_initialization(self, analyzer: ExperimentAnalyzer):
        """测试分析器初始化"""
        assert analyzer.data_dir.exists(), "数据目录不存在"
        assert isinstance(analyzer.config, AnalysisConfig), "配置类型错误"
        assert len(analyzer.data) > 0, "数据未正确加载"
        
    def test_load_all_data(self, analyzer: ExperimentAnalyzer):
        """测试数据加载"""
        assert set(analyzer.data.keys()) == {2, 4, 6, 8}, "l值不正确"
        
        for l_value, df in analyzer.data.items():
            assert isinstance(df, pd.DataFrame), "数据格式错误"
            assert len(df) >= analyzer.config.min_samples, "样本数量不足"
            assert all(col in df.columns for col in [
                "trial_id", "l_value", "convergence_time",
                "final_state", "network_features", "belief_history"
            ]), "缺少必需的列"
            
    def test_compute_tail_probabilities(self, analyzer: ExperimentAnalyzer):
        """测试尾概率计算"""
        tail_probs = analyzer.compute_tail_probabilities()
        
        assert isinstance(tail_probs, dict), "返回类型错误"
        assert set(tail_probs.keys()) == {2, 4, 6, 8}, "l值不正确"
        
        for l_value, prob_dict in tail_probs.items():
            assert "times" in prob_dict, "缺少times键"
            assert "probabilities" in prob_dict, "缺少probabilities键"
            assert len(prob_dict["times"]) == len(prob_dict["probabilities"]), "长度不匹配"
            assert all(0 <= p <= 1 for p in prob_dict["probabilities"]), "概率值范围错误"
            
    def test_analyze_convergence_states(self, analyzer: ExperimentAnalyzer):
        """测试收敛状态分析"""
        df = analyzer.analyze_convergence_states()
        
        assert isinstance(df, pd.DataFrame), "返回类型错误"
        assert len(df) == 4, "结果数量错误"  # 4个l值
        
        required_columns = {
            "l_value", "total_trials", 
            "contribution_ratio", "defection_ratio", "not_converged_ratio",
            "contribution_ci_lower", "contribution_ci_upper",
            "defection_ci_lower", "defection_ci_upper",
            "not_converged_ci_lower", "not_converged_ci_upper"
        }
        assert set(df.columns) == required_columns, "列名不正确"
        
        # 检查比率和置信区间
        for _, row in df.iterrows():
            assert 0 <= row["contribution_ratio"] <= 1, "贡献率范围错误"
            assert 0 <= row["defection_ratio"] <= 1, "背叛率范围错误"
            assert 0 <= row["not_converged_ratio"] <= 1, "未收敛率范围错误"
            assert abs(row["contribution_ratio"] + row["defection_ratio"] + 
                      row["not_converged_ratio"] - 1.0) < 1e-10, "比率之和不为1"
            
    def test_analyze_convergence_times(self, analyzer: ExperimentAnalyzer):
        """测试收敛时间分析"""
        stats = analyzer.analyze_convergence_times()
        
        assert isinstance(stats, dict), "返回类型错误"
        assert set(stats.keys()) == {2, 4, 6, 8}, "l值不正确"
        
        required_keys = {
            "mean", "median", "std", "min", "max", 
            "n_samples", "ci_lower", "ci_upper"
        }
        
        for l_value, l_stats in stats.items():
            assert set(l_stats.keys()) == required_keys, "统计量不完整"
            assert l_stats["min"] <= l_stats["median"] <= l_stats["max"], "统计量顺序错误"
            assert l_stats["ci_lower"] <= l_stats["mean"] <= l_stats["ci_upper"], "置信区间错误"
            
    def test_analyze_catastrophe_principle(self, analyzer: ExperimentAnalyzer):
        """测试灾难原理分析"""
        df = analyzer.analyze_catastrophe_principle()
        
        assert isinstance(df, pd.DataFrame), "返回类型错误"
        assert len(df) == 8, "结果数量错误"  # 4个l值 * 2种采样方式
        
        required_columns = {
            "l_value", "sampling", "max_probability",
            "sum_probability", "ratio"
        }
        assert set(df.columns) == required_columns, "列名不正确"
        
        # 检查采样方式
        assert set(df["sampling"].unique()) == {
            "no_replacement", "with_replacement"
        }, "采样方式错误"
        
        # 检查概率值
        assert all(0 <= p <= 1 for p in df["max_probability"]), "最大概率范围错误"
        assert all(0 <= p <= 1 for p in df["sum_probability"]), "和概率范围错误"
        
    def test_save_analysis_results(self, analyzer: ExperimentAnalyzer, tmp_path: Path):
        """测试结果保存"""
        # 保存结果
        results_dir = tmp_path / "analysis_results"
        analyzer.save_analysis_results(results_dir)
        
        # 检查文件是否创建
        assert (results_dir / "analysis_results.pkl").exists(), "Pickle文件未创建"
        assert (results_dir / "convergence_states.csv").exists(), "收敛状态CSV未创建"
        assert (results_dir / "catastrophe_principle.csv").exists(), "灾难原理CSV未创建"
        
        # 检查保存的结果
        with open(results_dir / "analysis_results.pkl", 'rb') as f:
            results = pickle.load(f)
            
        assert isinstance(results, dict), "结果格式错误"
        assert set(results.keys()) == {
            "tail_probabilities", "convergence_states",
            "convergence_times", "catastrophe_principle"
        }, "结果键不完整"
        
    def test_error_handling(self, tmp_path: Path):
        """测试错误处理"""
        # 测试不存在的目录
        with pytest.raises(FileNotFoundError):
            ExperimentAnalyzer(str(tmp_path / "nonexistent"))
            
        # 测试样本数不足
        small_data_dir = tmp_path / "small_data"
        small_data_dir.mkdir()
        (small_data_dir / "l_value_2").mkdir()
        pd.DataFrame({
            "trial_id": range(10),  # 少于min_samples
            "convergence_time": range(10),
            "final_state": ["contribution"] * 10
        }).to_csv(small_data_dir / "l_value_2" / "results.csv", index=False)
        
        analyzer = ExperimentAnalyzer(str(small_data_dir))
        assert len(analyzer.data) == 0, "应该跳过样本数不足的数据"