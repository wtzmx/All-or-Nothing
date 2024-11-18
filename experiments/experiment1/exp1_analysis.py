import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class AnalysisConfig:
    """分析配置类"""
    min_samples: int = 100  # 最小样本数
    confidence_level: float = 0.95  # 置信水平
    tail_bins: int = 50  # 尾概率分布的bin数
    network_feature_names: List[str] = None  # 需要分析的网络特征
    
    def __post_init__(self):
        if self.network_feature_names is None:
            self.network_feature_names = [
                "mean_degree", "max_degree", "n_triangles",
                "clustering_coefficient"
            ]

class ExperimentAnalyzer:
    """实验一数据分析器"""
    
    def __init__(self, 
                 data_dir: str,
                 config: Optional[AnalysisConfig] = None):
        """
        初始化分析器
        
        Parameters:
        -----------
        data_dir : str
            数据目录路径
        config : AnalysisConfig, optional
            分析配置
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
            
        self.config = config or AnalysisConfig()
        
        # 设置日志
        self._setup_logging()
        
        # 加载数据
        self.data = self._load_all_data()
        
    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_all_data(self) -> Dict[float, pd.DataFrame]:
        """加载所有实验数据"""
        data = {}
        for path in self.data_dir.glob("radius_*/results.csv"):
            radius = float(path.parent.name.split("_")[1])
            try:
                df = pd.read_csv(path)
                if len(df) >= self.config.min_samples:
                    data[radius] = df
                else:
                    self.logger.warning(
                        f"Insufficient samples for radius {radius}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error loading data for radius {radius}: {str(e)}"
                )
        return data
    
    def compute_tail_probabilities(self) -> Dict[float, np.ndarray]:
        """
        计算收敛时间的尾概率分布
        P(τ ≥ t) vs t
        """
        tail_probs = {}
        for radius, df in self.data.items():
            try:
                # 获取收敛时间
                conv_times = df["convergence_time"].values
                
                # 计算经验分布
                sorted_times = np.sort(conv_times)
                probs = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                
                tail_probs[radius] = {
                    "times": sorted_times,
                    "probabilities": probs
                }
                
            except Exception as e:
                self.logger.error(
                    f"Error computing tail probabilities for radius {radius}: {str(e)}"
                )
                
        return tail_probs
    
    def analyze_convergence_states(self) -> pd.DataFrame:
        """分析不同r_g值下的收敛情况"""
        results = []
        for radius, df in self.data.items():
            try:
                total = len(df)
                state_counts = df["final_state"].value_counts()
                
                result = {
                    "radius": radius,
                    "total_trials": total,
                    "contribution_ratio": state_counts.get("contribution", 0) / total,
                    "defection_ratio": state_counts.get("defection", 0) / total,
                    "not_converged_ratio": state_counts.get("not_converged", 0) / total
                }
                
                # 计算95%置信区间
                for state in ["contribution", "defection", "not_converged"]:
                    count = state_counts.get(state, 0)
                    ci = stats.binom.interval(
                        self.config.confidence_level, total, count/total
                    )
                    result[f"{state}_ci_lower"] = ci[0] / total
                    result[f"{state}_ci_upper"] = ci[1] / total
                    
                results.append(result)
                
            except Exception as e:
                self.logger.error(
                    f"Error analyzing convergence states for radius {radius}: {str(e)}"
                )
                
        return pd.DataFrame(results)
    
    def analyze_network_features(self) -> Dict[str, Dict]:
        """分析网络特征与最终状态的关系"""
        results = {}
        for feature in self.config.network_feature_names:
            try:
                feature_stats = {}
                for radius, df in self.data.items():
                    # 提取特征值
                    feature_values = pd.json_normalize(
                        df["network_features"].apply(eval)
                    )[feature]
                    
                    # 按最终状态分组计算统计量
                    stats_by_state = {}
                    for state in df["final_state"].unique():
                        values = feature_values[df["final_state"] == state]
                        stats_by_state[state] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "median": np.median(values),
                            "count": len(values)
                        }
                    
                    feature_stats[radius] = stats_by_state
                    
                results[feature] = feature_stats
                
            except Exception as e:
                self.logger.error(
                    f"Error analyzing feature {feature}: {str(e)}"
                )
                
        return results
    
    def analyze_metastable_states(self) -> Dict[float, Dict]:
        """分析元稳态特征"""
        results = {}
        for radius, df in self.data.items():
            try:
                # 提取信念历史
                belief_histories = df["belief_history"].apply(eval)
                
                # 检测元稳态
                metastable_stats = {
                    "n_metastable": 0,  # 元稳态数量
                    "mean_duration": 0,  # 平均持续时间
                    "mean_belief": np.nan,   # 元稳态期间的平均信念
                    "std_belief": np.nan     # 元稳态期间的信念标准差
                }
                
                belief_means = []
                belief_stds = []
                total_duration = 0
                
                for history in belief_histories:
                    # 计算信念变化率
                    belief_array = np.array(history)
                    if len(belief_array) < 2:  # 跳过太短的历史
                        continue
                    
                    changes = np.abs(np.diff(belief_array, axis=0))
                    
                    # 检测稳定区间
                    stable_periods = self._detect_stable_periods(
                        changes, threshold=0.01
                    )
                    
                    if stable_periods:
                        metastable_stats["n_metastable"] += len(stable_periods)
                        durations = [end - start for start, end in stable_periods]
                        total_duration += sum(durations)
                        
                        for start, end in stable_periods:
                            period_beliefs = belief_array[start:end]
                            belief_means.append(np.mean(period_beliefs))
                            belief_stds.append(np.std(period_beliefs))
                
                # 计算平均值，避免除以零
                n_trials = len(belief_histories)
                if n_trials > 0 and total_duration > 0:
                    metastable_stats["mean_duration"] = total_duration / n_trials
                    
                if belief_means:  # 只在有数据时计算统计量
                    metastable_stats["mean_belief"] = np.mean(belief_means)
                    metastable_stats["std_belief"] = np.mean(belief_stds)
                    
                results[radius] = metastable_stats
                
            except Exception as e:
                self.logger.error(
                    f"Error analyzing metastable states for radius {radius}: {str(e)}"
                )
                results[radius] = {
                    "n_metastable": 0,
                    "mean_duration": 0,
                    "mean_belief": np.nan,
                    "std_belief": np.nan
                }
                
        return results
    
    def _detect_stable_periods(self, 
                             changes: np.ndarray, 
                             threshold: float,
                             min_duration: int = 100) -> List[Tuple[int, int]]:
        """检测稳定区间"""
        stable_mask = np.all(changes < threshold, axis=1)
        stable_periods = []
        
        start = None
        for i, stable in enumerate(stable_mask):
            if stable and start is None:
                start = i
            elif not stable and start is not None:
                if i - start >= min_duration:
                    stable_periods.append((start, i))
                start = None
                
        # 处理最后一个区间
        if start is not None and len(stable_mask) - start >= min_duration:
            stable_periods.append((start, len(stable_mask)))
            
        return stable_periods
    
    def save_analysis_results(self, 
                            results_dir: Optional[str] = None):
        """保存分析结果"""
        if results_dir is None:
            results_dir = self.data_dir / "analysis"
        else:
            results_dir = Path(results_dir)
            
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 计算并保存各类分析结果
            analysis_results = {
                "tail_probabilities": self.compute_tail_probabilities(),
                "convergence_states": self.analyze_convergence_states(),
                "network_features": self.analyze_network_features(),
                "metastable_states": self.analyze_metastable_states()
            }
            
            # 保存为pickle格式
            with open(results_dir / "analysis_results.pkl", 'wb') as f:
                pickle.dump(analysis_results, f)
                
            # 保存convergence_states为CSV
            analysis_results["convergence_states"].to_csv(
                results_dir / "convergence_states.csv",
                index=False
            )
            
            self.logger.info(f"Analysis results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行分析
    analyzer = ExperimentAnalyzer("data/experiment1")
    analyzer.save_analysis_results()
