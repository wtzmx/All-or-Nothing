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
            # 针对多种网络类型的特征
            self.network_feature_names = [
                "degree_distribution", "clustering", "path_length",
                "centrality", "modularity", "assortativity"
            ]

class ExperimentAnalyzer:
    """实验三数据分析器：网络结构对比研究"""
    
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
        
    def _load_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """加载所有实验数据"""
        data = {}
        for network_dir in self.data_dir.iterdir():
            if not network_dir.is_dir() or network_dir.name == "analysis":
                continue
                
            network_type = network_dir.name
            data[network_type] = {}
            
            for param_dir in network_dir.iterdir():
                if not param_dir.is_dir():
                    continue
                    
                try:
                    df = pd.read_csv(param_dir / "results.csv")
                    if len(df) >= self.config.min_samples:
                        data[network_type][param_dir.name] = df
                    else:
                        self.logger.warning(
                            f"Insufficient samples for {network_type} - {param_dir.name}"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error loading data for {network_type} - {param_dir.name}: {str(e)}"
                    )
        return data
    
    def compute_tail_probabilities(self) -> Dict[str, Dict[str, Dict]]:
        """
        计算收敛时间的尾概率分布
        P(τ ≥ t) vs t
        """
        tail_probs = {}
        for network_type, param_data in self.data.items():
            tail_probs[network_type] = {}
            
            for param_str, df in param_data.items():
                try:
                    # 获取收敛时间
                    conv_times = df["convergence_time"].values
                    
                    # 计算经验分布
                    sorted_times = np.sort(conv_times)
                    probs = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                    
                    tail_probs[network_type][param_str] = {
                        "times": sorted_times,
                        "probabilities": probs
                    }
                    
                except Exception as e:
                    self.logger.error(
                        f"Error computing tail probabilities for {network_type} - {param_str}: {str(e)}"
                    )
                    
        return tail_probs
    
    def analyze_convergence_states(self) -> pd.DataFrame:
        """分析不同网络类型和参数下的收敛情况"""
        results = []
        for network_type, param_data in self.data.items():
            for param_str, df in param_data.items():
                try:
                    total = len(df)
                    state_counts = df["final_state"].value_counts()
                    
                    result = {
                        "network_type": network_type,
                        "params": param_str,
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
                        f"Error analyzing convergence states for {network_type} - {param_str}: {str(e)}"
                    )
                    
        return pd.DataFrame(results)
    
    def analyze_network_features(self) -> Dict[str, Dict[str, Dict]]:
        """分析网络特征与最终状态的关系"""
        results = {}
        for feature in self.config.network_feature_names:
            try:
                feature_stats = {}
                for network_type, param_data in self.data.items():
                    feature_stats[network_type] = {}
                    
                    for param_str, df in param_data.items():
                        # 提取特征值
                        feature_values = pd.json_normalize(
                            df["network_features"].apply(eval)
                        )[feature]
                        
                        # 特殊处理度分布特征
                        if feature == "degree_distribution":
                            # 对于度分布，我们计算一些统计量
                            feature_values = feature_values.apply(
                                lambda x: {
                                    'mean': np.mean(x),
                                    'std': np.std(x),
                                    'min': np.min(x),
                                    'max': np.max(x),
                                    'median': np.median(x)
                                }
                            )
                        
                        # 按最终状态分组计算统计量
                        stats_by_state = {}
                        for state in df["final_state"].unique():
                            values = feature_values[df["final_state"] == state]
                            if feature == "degree_distribution":
                                # 对度分布特征计算汇总统计
                                stats = {
                                    'mean': np.mean([v['mean'] for v in values]),
                                    'std': np.mean([v['std'] for v in values]),
                                    'median': np.mean([v['median'] for v in values]),
                                    'min': np.mean([v['min'] for v in values]),
                                    'max': np.mean([v['max'] for v in values]),
                                    'count': len(values)
                                }
                            else:
                                # 其他特征的常规统计
                                stats = {
                                    "mean": np.mean(values),
                                    "std": np.std(values),
                                    "median": np.median(values),
                                    "count": len(values)
                                }
                            stats_by_state[state] = stats
                        
                        feature_stats[network_type][param_str] = stats_by_state
                        
                results[feature] = feature_stats
                
            except Exception as e:
                self.logger.error(
                    f"Error analyzing feature {feature}: {str(e)}"
                )
                
        return results
    
    def analyze_network_comparison(self) -> Dict[str, pd.DataFrame]:
        """分不同网络结构的性能对比"""
        comparison_results = {}
        
        try:
            # 收敛速度对比
            speed_data = []
            for network_type, param_data in self.data.items():
                for param_str, df in param_data.items():
                    conv_times = df[df["final_state"] != "not_converged"]["convergence_time"]
                    if len(conv_times) > 0:
                        speed_data.append({
                            "network_type": network_type,
                            "params": param_str,
                            "mean_time": np.mean(conv_times),
                            "median_time": np.median(conv_times),
                            "std_time": np.std(conv_times)
                        })
            comparison_results["convergence_speed"] = pd.DataFrame(speed_data)
            
            # 合作水平对比
            coop_data = []
            for network_type, param_data in self.data.items():
                for param_str, df in param_data.items():
                    coop_ratio = len(df[df["final_state"] == "contribution"]) / len(df)
                    coop_data.append({
                        "network_type": network_type,
                        "params": param_str,
                        "cooperation_ratio": coop_ratio
                    })
            comparison_results["cooperation_level"] = pd.DataFrame(coop_data)
            
            # 稳定性对比
            stability_data = []
            for network_type, param_data in self.data.items():
                for param_str, df in param_data.items():
                    conv_ratio = len(df[df["final_state"] != "not_converged"]) / len(df)
                    stability_data.append({
                        "network_type": network_type,
                        "params": param_str,
                        "convergence_ratio": conv_ratio
                    })
            comparison_results["stability"] = pd.DataFrame(stability_data)
            
        except Exception as e:
            self.logger.error(f"Error in network comparison analysis: {str(e)}")
            
        return comparison_results
    
    def perform_statistical_tests(self) -> Dict[str, pd.DataFrame]:
        """执行统计检验"""
        test_results = {}
        
        try:
            # 准备数据
            conv_times = defaultdict(list)
            for network_type, param_data in self.data.items():
                for param_str, df in param_data.items():
                    key = f"{network_type}_{param_str}"
                    times = df[df["final_state"] != "not_converged"]["convergence_time"]
                    conv_times[key].extend(times)
            
            # KS检验
            ks_results = []
            keys = list(conv_times.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    stat, pval = stats.ks_2samp(
                        conv_times[keys[i]], 
                        conv_times[keys[j]]
                    )
                    ks_results.append({
                        "group1": keys[i],
                        "group2": keys[j],
                        "statistic": stat,
                        "p_value": pval
                    })
            test_results["ks_test"] = pd.DataFrame(ks_results)
            
            # Mann-Whitney U检验
            mw_results = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    stat, pval = stats.mannwhitneyu(
                        conv_times[keys[i]], 
                        conv_times[keys[j]],
                        alternative="two-sided"
                    )
                    mw_results.append({
                        "group1": keys[i],
                        "group2": keys[j],
                        "statistic": stat,
                        "p_value": pval
                    })
            test_results["mann_whitney"] = pd.DataFrame(mw_results)
            
            # Kruskal-Wallis H检验
            h_stat, p_val = stats.kruskal(*conv_times.values())
            test_results["kruskal_wallis"] = pd.DataFrame([{
                "statistic": h_stat,
                "p_value": p_val
            }])
            
        except Exception as e:
            self.logger.error(f"Error in statistical testing: {str(e)}")
            
        return test_results
    
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
                "network_comparison": self.analyze_network_comparison(),
                "statistical_tests": self.perform_statistical_tests()
            }
            
            # 保存为pickle格式
            with open(results_dir / "analysis_results.pkl", 'wb') as f:
                pickle.dump(analysis_results, f)
                
            # 保存表格数据为CSV
            analysis_results["convergence_states"].to_csv(
                results_dir / "convergence_states.csv",
                index=False
            )
            
            for metric, df in analysis_results["network_comparison"].items():
                df.to_csv(
                    results_dir / f"{metric}_comparison.csv",
                    index=False
                )
                
            for test_name, df in analysis_results["statistical_tests"].items():
                df.to_csv(
                    results_dir / f"{test_name}_results.csv",
                    index=False
                )
            
            self.logger.info(f"Analysis results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            raise

if __name__ == "__main__":
    # 运行分析
    analyzer = ExperimentAnalyzer("data/experiment3")
    analyzer.save_analysis_results()
