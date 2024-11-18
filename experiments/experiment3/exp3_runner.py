import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from multiprocessing import Pool
import pickle

from src.networks.geometric import RandomGeometricGraph
from src.networks.regular import CirculantGraph
from src.networks.random import ERGraph
from src.networks.small_world import WSGraph
from src.networks.scale_free import BAGraph
from src.models.agent import Agent
from src.models.game import PublicGoodsGame

class ExperimentRunner:
    """实验三运行器：网络结构对比研究"""
    
    def __init__(self, config_path: str):
        """
        初始化实验运行器
        
        Parameters:
        -----------
        config_path : str
            配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 设置输出目录
        self.output_dir = Path(self.config["output"]["base_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化结果存储
        self.results = {
            "convergence_times": [],
            "final_states": [],
            "network_features": [],
            "belief_histories": []
        }
        
    def _setup_logging(self):
        """配置日志系统"""
        log_config = self.config["logging"]
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.output_dir / log_config["file_name"]
                )
            ] if log_config["save_to_file"] else [logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_network(self, network_type: str, params: Dict, trial_id: int):
        """
        创建指定类型的网络
        
        Parameters:
        -----------
        network_type : str
            网络类型
        params : Dict
            网络参数
        trial_id : int
            实验ID
            
        Returns:
        --------
        network : Network
            网络实例
        
        Raises:
        -------
        ValueError
            当网络类型无效时
        """
        # 首先验证网络类型
        valid_types = {"geometric", "regular", "random", "small_world", "scale_free"}
        if network_type not in valid_types:
            raise ValueError(f"Invalid network type: {network_type}")
        
        seed = self.config["networks"]["seed"] + trial_id
        n_agents = self.config["networks"][network_type]["n_agents"]
        
        if network_type == "geometric":
            return RandomGeometricGraph(
                n_nodes=n_agents,
                radius=params["radius"],
                seed=seed
            )
        elif network_type == "regular":
            return CirculantGraph(
                n_nodes=n_agents,
                neighbors=params["l"],
                seed=seed
            )
        elif network_type == "random":
            return ERGraph(
                n_nodes=n_agents,
                p=params["p"],
                seed=seed
            )
        elif network_type == "small_world":
            return WSGraph(
                n_nodes=n_agents,
                k=self.config["networks"]["small_world"]["k"],
                p=params["p"],
                seed=seed
            )
        elif network_type == "scale_free":
            return BAGraph(
                n_nodes=n_agents,
                m=params["m"],
                seed=seed
            )
        
    def _run_single_trial(self, 
                         network_type: str,
                         params: Dict,
                         trial_id: int) -> Dict:
        """
        运行单次实验
        
        Parameters:
        -----------
        network_type : str
            网络类型
        params : Dict
            网络参数
        trial_id : int
            实验ID
            
        Returns:
        --------
        Dict : 实验结果
        """
        # 生成网络
        network = self._create_network(network_type, params, trial_id)
        
        # 初始化游戏
        game = PublicGoodsGame(
            n_agents=network.N,
            learning_rate=self.config["game"]["learning_rate"],
            initial_belief=self.config["game"]["initial_belief"],
            lambda_dist=self.config["game"]["lambda_distribution"]["type"],
            lambda_params=self.config["game"]["lambda_distribution"]["params"]
        )
        
        # 记录信念历史
        belief_history = []
        
        # 运行实验
        t = 0
        converged = False
        threshold = self.config["simulation"]["convergence_threshold"]
        
        while t < self.config["simulation"]["max_rounds"]:
            # 选择焦点智能体
            focal_agent = np.random.randint(network.N)
            
            # 获取邻居集合
            neighbors = network.get_closed_neighbors(focal_agent)
            
            # 执行一轮博弈
            actions, payoffs = game.play_round(neighbors)
            
            # 记录信念
            if t % self.config["simulation"]["save_interval"] == 0:
                belief_history.append(game.get_beliefs())
            
            # 检查收敛
            if game.check_convergence(threshold):
                converged = True
                beliefs = game.get_beliefs()
                # 立即检查收敛状态
                if all(b >= (1 - threshold) for b in beliefs):
                    final_state = "contribution"
                    break
                elif all(b <= threshold for b in beliefs):
                    final_state = "defection"
                    break
            
            t += 1
        
        # 如果没有收敛或循环结束，设置为not_converged
        if not converged:
            final_state = "not_converged"
        
        # 返回结果
        return {
            "trial_id": trial_id,
            "network_type": network_type,
            "params": params,
            "convergence_time": t,
            "final_state": final_state,
            "network_features": network.get_stats(),
            "belief_history": belief_history
        }
        
    def _run_parallel_trials(self, 
                           network_type: str,
                           params: Dict) -> List[Dict]:
        """并行运行多次实验"""
        if self.config["parallel"]["enabled"]:
            with Pool(self.config["parallel"]["n_processes"]) as pool:
                results = pool.starmap(
                    self._run_single_trial,
                    [(network_type, params, i) 
                     for i in range(self.config["simulation"]["n_trials"])]
                )
        else:
            results = [
                self._run_single_trial(network_type, params, i) 
                for i in range(self.config["simulation"]["n_trials"])
            ]
        return results
    
    def run_experiment(self):
        """运行完整实验"""
        self.logger.info("Starting experiment")
        start_time = datetime.now()
        
        # 遍历所有网络类型
        for network_type, config in self.config["networks"].items():
            # 跳过seed和未启用的网络
            if network_type == "seed" or not config["enabled"]:
                continue
                
            self.logger.info(f"Running trials for {network_type} network")
            
            # 获取参数列表
            if network_type == "geometric":
                param_list = [{"radius": r} for r in config["radius_list"]]
            elif network_type == "regular":
                param_list = [{"l": l} for l in config["l_values"]]
            elif network_type == "random":
                param_list = [{"p": p} for p in config["p_values"]]
            elif network_type == "small_world":
                param_list = [{"p": p} for p in config["p_values"]]
            elif network_type == "scale_free":
                param_list = [{"m": m} for m in config["m_values"]]
                
            # 对每组参数运行实验
            for params in param_list:
                self.logger.info(f"Running trials for {network_type} with params {params}")
                
                # 运行实验并收集结果
                trial_results = self._run_parallel_trials(network_type, params)
                
                # 保存结果
                self._save_results(trial_results, network_type, params)
            
        end_time = datetime.now()
        self.logger.info(f"Experiment completed in {end_time - start_time}")
        
    def _save_results(self, 
                     results: List[Dict], 
                     network_type: str,
                     params: Dict):
        """保存实验结果"""
        # 创建结果目录
        param_str = "_".join(f"{k}_{v}" for k, v in params.items())
        results_dir = self.output_dir / network_type / param_str
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存CSV格式
        if "csv" in self.config["output"]["formats"]:
            df.to_csv(results_dir / "results.csv", index=False)
            
        # 保存pickle格式
        if "pickle" in self.config["output"]["formats"]:
            with open(results_dir / "results.pkl", 'wb') as f:
                pickle.dump(results, f)
                
        self.logger.info(f"Results saved for {network_type} with params {params}")

if __name__ == "__main__":
    # 运行实验
    runner = ExperimentRunner("experiments/experiment3/exp3_config.yaml")
    runner.run_experiment()
