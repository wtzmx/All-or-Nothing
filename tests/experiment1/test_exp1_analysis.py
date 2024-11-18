import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import json
import pickle
from experiments.experiment1.exp1_analysis import ExperimentAnalyzer, AnalysisConfig

@pytest.fixture
def test_data_dir(tmp_path):
    """创建测试数据目录"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # 为不同的radius创建测试数据
    radii = [0.15, 0.2, 0.25, 0.3]
    for radius in radii:
        radius_dir = data_dir / f"radius_{radius}"
        radius_dir.mkdir()
        
        # 创建测试数据
        test_data = []
        for i in range(150):  # 确保超过min_samples
            test_data.append({
                "trial_id": i,
                "radius": radius,
                "convergence_time": np.random.randint(100, 10000),
                "final_state": np.random.choice(
                    ["contribution", "defection", "not_converged"],
                    p=[0.3, 0.6, 0.1]
                ),
                "network_features": json.dumps({
                    "mean_degree": np.random.uniform(2, 8),
                    "max_degree": np.random.randint(5, 15),
                    "n_triangles": np.random.randint(10, 100),
                    "clustering_coefficient": np.random.uniform(0, 1)
                }),
                "belief_history": json.dumps([
                    list(np.random.uniform(0, 1, 10))
                    for _ in range(5)
                ])
            })
        
        # 保存为CSV
        df = pd.DataFrame(test_data)
        df.to_csv(radius_dir / "results.csv", index=False)
    
    return data_dir

@pytest.fixture
def analyzer(test_data_dir):
    """创建分析器实例"""
    config = AnalysisConfig(
        min_samples=100,
        confidence_level=0.95,
        tail_bins=50,
        network_feature_names=[
            "mean_degree", "max_degree", "n_triangles",
            "clustering_coefficient"
        ]
    )
    return ExperimentAnalyzer(str(test_data_dir), config)

def test_initialization(test_data_dir):
    """测试分析器初始化"""
    analyzer = ExperimentAnalyzer(str(test_data_dir))
    assert analyzer.data_dir == Path(test_data_dir)
    assert analyzer.config is not None
    assert analyzer.logger is not None
    assert len(analyzer.data) == 4  # 应该加载4个radius的数据

def test_load_data(analyzer):
    """测试数据加载"""
    assert len(analyzer.data) == 4
    for radius, df in analyzer.data.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= analyzer.config.min_samples
        assert all(col in df.columns for col in [
            "trial_id", "radius", "convergence_time",
            "final_state", "network_features", "belief_history"
        ])

def test_compute_tail_probabilities(analyzer):
    """测试尾概率计算"""
    tail_probs = analyzer.compute_tail_probabilities()
    
    assert len(tail_probs) == 4  # 4个radius
    for radius, data in tail_probs.items():
        assert "times" in data
        assert "probabilities" in data
        assert len(data["times"]) == len(data["probabilities"])
        assert all(0 <= p <= 1 for p in data["probabilities"])
        assert np.all(np.diff(data["times"]) >= 0)  # 时间应该是升序的

def test_analyze_convergence_states(analyzer):
    """测试收敛状态分析"""
    results = analyzer.analyze_convergence_states()
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 4  # 4个radius
    
    required_columns = {
        "radius", "total_trials",
        "contribution_ratio", "defection_ratio", "not_converged_ratio",
        "contribution_ci_lower", "contribution_ci_upper",
        "defection_ci_lower", "defection_ci_upper",
        "not_converged_ci_lower", "not_converged_ci_upper"
    }
    assert set(results.columns) >= required_columns
    
    # 检查比例和置信区间的合理性
    for _, row in results.iterrows():
        assert abs(row["contribution_ratio"] + row["defection_ratio"] + 
                  row["not_converged_ratio"] - 1.0) < 1e-10
        for state in ["contribution", "defection", "not_converged"]:
            assert 0 <= row[f"{state}_ratio"] <= 1
            assert row[f"{state}_ci_lower"] <= row[f"{state}_ratio"]
            assert row[f"{state}_ratio"] <= row[f"{state}_ci_upper"]

def test_analyze_network_features(analyzer):
    """测试网络特征分析"""
    results = analyzer.analyze_network_features()
    
    assert isinstance(results, dict)
    assert set(results.keys()) == set(analyzer.config.network_feature_names)
    
    for feature, feature_stats in results.items():
        assert len(feature_stats) == 4  # 4个radius
        for radius, stats_by_state in feature_stats.items():
            for state, stats in stats_by_state.items():
                assert set(stats.keys()) >= {"mean", "std", "median", "count"}
                assert stats["count"] > 0
                assert not np.isnan(stats["mean"])
                assert not np.isnan(stats["std"])

def test_analyze_metastable_states(analyzer):
    """测试元稳态分析"""
    results = analyzer.analyze_metastable_states()
    
    assert isinstance(results, dict)
    assert len(results) == 4  # 4个radius
    
    for radius, stats in results.items():
        required_keys = {
            "n_metastable", "mean_duration",
            "mean_belief", "std_belief"
        }
        assert set(stats.keys()) >= required_keys
        assert stats["n_metastable"] >= 0
        assert stats["mean_duration"] >= 0
        if not np.isnan(stats["mean_belief"]):
            assert isinstance(stats["mean_belief"], (float, np.float64))
        if not np.isnan(stats["std_belief"]):
            assert isinstance(stats["std_belief"], (float, np.float64))

def test_detect_stable_periods(analyzer):
    """测试稳定区间检测"""
    # 创建测试数据
    changes = np.array([
        [0.005, 0.005],  # 稳定
        [0.005, 0.005],  # 稳定
        [0.015, 0.015],  # 不稳定
        [0.005, 0.005],  # 稳定
        [0.005, 0.005],  # 稳定
    ])
    
    periods = analyzer._detect_stable_periods(
        changes, threshold=0.01, min_duration=2
    )
    
    assert isinstance(periods, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in periods)
    assert all(start < end for start, end in periods)

def test_save_analysis_results(analyzer, tmp_path):
    """测试结果保存"""
    results_dir = tmp_path / "analysis_results"
    analyzer.save_analysis_results(str(results_dir))
    
    # 检查文件是否创建
    assert (results_dir / "analysis_results.pkl").exists()
    assert (results_dir / "convergence_states.csv").exists()
    
    # 检查CSV文件内容
    df = pd.read_csv(results_dir / "convergence_states.csv")
    assert len(df) == 4  # 4个radius
    
    # 检查pickle文件内容
    try:
        with open(results_dir / "analysis_results.pkl", 'rb') as f:
            results = pickle.load(f)
            assert set(results.keys()) >= {
                "tail_probabilities",
                "convergence_states",
                "network_features",
                "metastable_states"
            }
    except Exception as e:
        pytest.fail(f"Failed to load pickle file: {str(e)}")

def test_error_handling():
    """测试错误处理"""
    # 测试不存在的目录
    with pytest.raises(FileNotFoundError):
        ExperimentAnalyzer("nonexistent_directory")
    
    # 测试数据不足
    with pytest.raises(Exception):
        config = AnalysisConfig(min_samples=1000000)  # 设置一个很大的最小样本数
        ExperimentAnalyzer("test_data", config)

def teardown_module(module):
    """清理测试产生的临时文件"""
    shutil.rmtree("test_data", ignore_errors=True) 