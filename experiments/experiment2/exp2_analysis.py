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
            # 针对规则图的特征
            self.network_feature_names = [
                "degree", "n_triangles", "clustering_coefficient",
                "average_path_length", "diameter"
            ]

class ExperimentAnalyzer:
    """实验二数据分析器"""
    
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
        
    def _load_all_data(self) -> Dict[int, pd.DataFrame]:
        """加载所有实验数据"""
        data = {}
        for path in self.data_dir.glob("l_value_*/results.csv"):
            l_value = int(path.parent.name.split("_")[2])  # 从l_value_X提取X
            try:
                df = pd.read_csv(path)
                if len(df) >= self.config.min_samples:
                    data[l_value] = df
                else:
                    self.logger.warning(
                        f"Insufficient samples for l = {l_value}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error loading data for l = {l_value}: {str(e)}"
                )
        return data
    
    def compute_tail_probabilities(self) -> Dict[int, np.ndarray]:
        """
        计算收敛时间的尾概率分布
        P(τ ≥ t) vs t
        """
        tail_probs = {}
        for l_value, df in self.data.items():
            try:
                # 获取收敛时间
                conv_times = df["convergence_time"].values
                
                # 计算经验分布
                sorted_times = np.sort(conv_times)
                probs = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                
                tail_probs[l_value] = {
                    "times": sorted_times,
                    "probabilities": probs
                }
                
            except Exception as e:
                self.logger.error(
                    f"Error computing tail probabilities for l = {l_value}: {str(e)}"
                )
                
        return tail_probs
    
    def analyze_convergence_states(self) -> pd.DataFrame:
        """分析不同l值下的收敛情况"""
        results = []
        for l_value, df in self.data.items():
            try:
                total = len(df)
                state_counts = df["final_state"].value_counts()
                
                result = {
                    "l_value": l_value,
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
                    f"Error analyzing convergence states for l = {l_value}: {str(e)}"
                )
                
        return pd.DataFrame(results)
    
    def analyze_convergence_times(self) -> Dict[int, Dict]:
        """分析收敛时间的统计特征"""
        results = {}
        for l_value, df in self.data.items():
            try:
                # 只考虑已收敛的案例
                conv_times = df[df["final_state"] != "not_converged"]["convergence_time"]
                
                if len(conv_times) > 0:
                    stats_dict = {
                        "mean": np.mean(conv_times),
                        "median": np.median(conv_times),
                        "std": np.std(conv_times),
                        "min": np.min(conv_times),
                        "max": np.max(conv_times),
                        "n_samples": len(conv_times)
                    }
                    
                    # 计算置信区间
                    ci = stats.t.interval(
                        self.config.confidence_level,
                        len(conv_times) - 1,
                        loc=stats_dict["mean"],
                        scale=stats.sem(conv_times)
                    )
                    stats_dict["ci_lower"] = ci[0]
                    stats_dict["ci_upper"] = ci[1]
                    
                    results[l_value] = stats_dict
                    
            except Exception as e:
                self.logger.error(
                    f"Error analyzing convergence times for l = {l_value}: {str(e)}"
                )
                
        return results
    
    def analyze_catastrophe_principle(self) -> pd.DataFrame:
        """
        分析灾难原理比率
        对于n=2的情况计算P(max{X1,X2} > t) : P(X1 + X2 > t)
        """
        results = []
        t_threshold = 1e6  # 时间阈值
        
        for l_value, df in self.data.items():
            try:
                conv_times = df["convergence_time"].values
                n_samples = len(conv_times)
                
                if n_samples < 2:
                    continue
                    
                # 无放回抽样
                pairs_no_replace = np.array([
                    conv_times[i:i+2] 
                    for i in range(0, n_samples-1, 2)
                ])
                
                # 有放回抽样
                pairs_replace = np.random.choice(
                    conv_times, 
                    size=(n_samples//2, 2)
                )
                
                # 计算比率
                for sample_type, pairs in [
                    ("no_replacement", pairs_no_replace),
                    ("with_replacement", pairs_replace)
                ]:
                    max_prob = np.mean(np.max(pairs, axis=1) > t_threshold)
                    sum_prob = np.mean(np.sum(pairs, axis=1) > t_threshold)
                    
                    results.append({
                        "l_value": l_value,
                        "sampling": sample_type,
                        "max_probability": max_prob,
                        "sum_probability": sum_prob,
                        "ratio": max_prob / sum_prob if sum_prob > 0 else np.nan
                    })
                    
            except Exception as e:
                self.logger.error(
                    f"Error analyzing catastrophe principle for l = {l_value}: {str(e)}"
                )
                
        return pd.DataFrame(results)
    
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
                "convergence_times": self.analyze_convergence_times(),
                "catastrophe_principle": self.analyze_catastrophe_principle()
            }
            
            # 保存为pickle格式
            with open(results_dir / "analysis_results.pkl", 'wb') as f:
                pickle.dump(analysis_results, f)
                
            # 保存表格数据为CSV
            analysis_results["convergence_states"].to_csv(
                results_dir / "convergence_states.csv",
                index=False
            )
            analysis_results["catastrophe_principle"].to_csv(
                results_dir / "catastrophe_principle.csv",
                index=False
            )
            
            self.logger.info(f"Analysis results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行分析
    analyzer = ExperimentAnalyzer("data/experiment2")
    analyzer.save_analysis_results()