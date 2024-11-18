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
from tqdm import tqdm

from src.networks.geometric import RandomGeometricGraph
from src.models.agent import Agent
from src.models.game import PublicGoodsGame

class ExperimentRunner:
    """实验一运行器"""
    
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
        
    def _run_single_trial(self, 
                         radius: float, 
                         trial_id: int) -> Dict:
        """
        运行单次实验
        
        Parameters:
        -----------
        radius : float
            网络半径r_g
        trial_id : int
            实验ID
            
        Returns:
        --------
        Dict : 实验结果
        """
        # 生成网络
        network = RandomGeometricGraph(
            n_nodes=self.config["network"]["n_agents"],
            radius=radius,
            seed=self.config["network"]["seed"] + trial_id
        )
        
        # 初始化游戏
        game = PublicGoodsGame(
            n_agents=self.config["network"]["n_agents"],
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
            focal_agent = np.random.randint(self.config["network"]["n_agents"])
            
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
            "radius": radius,
            "convergence_time": t,
            "final_state": final_state,
            "network_features": network.get_stats(),
            "belief_history": belief_history
        }
        
    def _run_parallel_trials(self, radius: float) -> List[Dict]:
        """并行运行多次实验"""
        n_trials = self.config["simulation"]["n_trials"]
        
        # 创建参数列表
        params = [(radius, i) for i in range(n_trials)]
        
        if self.config["parallel"]["enabled"]:
            with Pool(self.config["parallel"]["n_processes"]) as pool:
                # 使用starmap替代imap和lambda
                results = list(tqdm(
                    pool.starmap(
                        self._run_single_trial,
                        params
                    ),
                    total=n_trials,
                    desc=f"Radius {radius:.3f}",
                    ncols=100,
                    position=0
                ))
        else:
            # 使用tqdm包装串行任务
            results = []
            for i in tqdm(range(n_trials), 
                         desc=f"Radius {radius:.3f}", 
                         ncols=100):
                results.append(self._run_single_trial(radius, i))
                
        return results
    
    def run_experiment(self):
        """运行完整实验"""
        self.logger.info("Starting experiment")
        start_time = datetime.now()
        
        # 显示实验配置信息
        self.logger.info(f"Configuration:")
        self.logger.info(f"- Number of agents: {self.config['network']['n_agents']}")
        self.logger.info(f"- Number of trials per radius: {self.config['simulation']['n_trials']}")
        self.logger.info(f"- Max rounds per trial: {self.config['simulation']['max_rounds']}")
        
        # 使用tqdm包装radius循环
        radius_list = self.config["network"]["radius_list"]
        for radius in tqdm(radius_list, 
                          desc="Overall progress", 
                          position=1, 
                          leave=True):
            self.logger.info(f"\nRunning trials for radius {radius}")
            
            # 运行实验并收集结果
            trial_results = self._run_parallel_trials(radius)
            
            # 保存结果
            self._save_results(trial_results, radius)
            
            # 显示当前radius的简要统计
            conv_times = [r['convergence_time'] for r in trial_results]
            self.logger.info(f"Results for radius {radius}:")
            self.logger.info(f"- Average convergence time: {np.mean(conv_times):.2f}")
            self.logger.info(f"- Convergence rate: {sum(r['final_state'] != 'not_converged' for r in trial_results) / len(trial_results):.2%}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"\nExperiment completed in {duration}")
        self.logger.info(f"Results saved in {self.output_dir}")
        
    def _save_results(self, results: List[Dict], radius: float):
        """保存实验结果"""
        # 创建结果目录
        results_dir = self.output_dir / f"radius_{radius}"
        results_dir.mkdir(exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存CSV格式
        if "csv" in self.config["output"]["formats"]:
            df.to_csv(results_dir / "results.csv", index=False)
            
        # 保存pickle格式
        if "pickle" in self.config["output"]["formats"]:
            with open(results_dir / "results.pkl", 'wb') as f:
                pickle.dump(results, f)
                
        self.logger.info(f"Results saved for radius {radius}")

if __name__ == "__main__":
    # 运行实验
    runner = ExperimentRunner("experiments/experiment1/exp1_config.yaml")
    runner.run_experiment()
