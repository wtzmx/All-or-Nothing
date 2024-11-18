import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
import json
import time
from pathlib import Path
import logging
from datetime import datetime

from src.models.game import PublicGoodsGame
from src.networks.geometric import RandomGeometricGraph
from src.networks.regular import CirculantGraph
from src.networks.metrics import NetworkMetrics
from src.simulation.config import ExperimentConfig

class SimulationRunner:
    """
    仿真运行器，负责执行完整的仿真实验
    包括网络初始化、博弈执行、数据收集和结果保存
    """
    def __init__(self, config: ExperimentConfig) -> None:
        """
        初始化仿真运行器
        
        Parameters:
        -----------
        config : ExperimentConfig
            实验配置对象
        """
        self.config = config
        self.config.validate()  # 验证配置合法性
        
        # 初始化网络和博弈环境
        self.network = None
        self.game = None
        self.adjacency: Dict[int, Set[int]] = {}
        
        # 初始化数据收集器
        self.belief_history: List[List[float]] = []
        self.action_history: List[Dict[int, str]] = []
        self.payoff_history: List[Dict[int, float]] = []
        self.network_stats: Dict = {}
        self.convergence_stats: Dict = {}
        
        # 设置随机种子
        if self.config.simulation.seed is not None:
            np.random.seed(self.config.simulation.seed)
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """配置日志系统"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"simulation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def initialize(self) -> None:
        """
        初始化网络和博弈环境
        """
        # 生成网络
        if self.config.network.type == "geometric":
            network = RandomGeometricGraph(
                n_nodes=self.config.network.n_agents,
                radius=self.config.network.r_g,
                seed=self.config.simulation.seed
            )
            # 添加r_g到网络统计信息
            self.network_stats = network.get_stats()
            self.network_stats["r_g"] = self.config.network.r_g
            
        elif self.config.network.type == "regular":
            network = CirculantGraph(
                n_nodes=self.config.network.n_agents,
                neighbors=self.config.network.degree,
                seed=self.config.simulation.seed
            )
            self.network_stats = network.get_stats()
            
        else:
            raise ValueError(f"Unknown network type: {self.config.network.type}")
            
        # 获取邻接表
        self.adjacency = network.adjacency
        
        # 转换为NetworkX图(用于可视化和某些分析)
        self.network = nx.Graph(self.adjacency)
        
        # 初始化博弈
        self.game = PublicGoodsGame(
            n_agents=self.config.network.n_agents,
            learning_rate=self.config.game.learning_rate,
            initial_belief=self.config.game.initial_belief,
            lambda_dist=self.config.game.lambda_dist,
            lambda_params=self.config.game.lambda_params
        )
        
        logging.info(f"Initialized network with {self.network_stats['n_nodes']} nodes "
                    f"and {self.network_stats['n_edges']} edges")
        
    def run(self) -> None:
        """执行完整的仿真实验"""
        if not self.adjacency or self.game is None:
            raise RuntimeError("Must call initialize() before run()")
            
        start_time = time.time()
        round_count = 0
        stable_rounds = 0  # 跟踪稳定轮次数
        last_beliefs = None
        min_rounds = 50  # 最小轮次要求
        
        logging.info("Starting simulation...")
        
        while round_count < self.config.simulation.max_rounds:
            # 随机选择一个智能体及其邻居进行博弈
            focal_agent = np.random.randint(self.config.network.n_agents)
            players = {focal_agent} | self.adjacency[focal_agent]
            
            # 执行一轮博弈
            actions, payoffs = self.game.play_round(players)
            current_beliefs = self.game.get_all_beliefs()
            
            # 收集数据
            if round_count % self.config.simulation.save_interval == 0:
                self.belief_history.append(current_beliefs)
                self.action_history.append(actions)
                self.payoff_history.append(payoffs)
                self._save_intermediate_results()  # 每个保存间隔都保存中间结果
                    
            # 检查收敛
            if last_beliefs is not None and round_count >= min_rounds:
                belief_change = np.max(np.abs(np.array(current_beliefs) - np.array(last_beliefs)))
                if belief_change < self.config.simulation.convergence_threshold:
                    stable_rounds += 1
                    if stable_rounds >= 10:  # 连续10轮稳定才算真正收敛
                        logging.info(f"Simulation converged after {round_count} rounds")
                        self.convergence_stats = {
                            "converged": True,
                            "rounds": round_count,
                            "time": time.time() - start_time,
                            "final_beliefs": current_beliefs,
                            "mean_belief": float(np.mean(current_beliefs)),
                            "std_belief": float(np.std(current_beliefs))
                        }
                        break
                else:
                    stable_rounds = 0  # 重置稳定轮次计数
                    
            last_beliefs = current_beliefs.copy()
            round_count += 1
            
            # 打印进度
            if round_count % 100000 == 0:
                logging.info(
                    f"Completed {round_count} rounds... "
                    f"Mean belief: {np.mean(current_beliefs):.4f}, "
                    f"Std: {np.std(current_beliefs):.4f}"
                )
                
        # 如果达到最大轮次仍未收敛
        if "converged" not in self.convergence_stats:
            logging.warning("Reached maximum rounds without convergence")
            final_beliefs = self.game.get_all_beliefs()
            self.convergence_stats = {
                "converged": False,
                "rounds": round_count,
                "time": time.time() - start_time,
                "final_beliefs": final_beliefs,
                "mean_belief": float(np.mean(final_beliefs)),
                "std_belief": float(np.std(final_beliefs))
            }
            
        self._save_final_results()
        
    def _save_intermediate_results(self) -> None:
        """保存中间结果"""
        if not self.belief_history:  # 如果没有数据则跳过
            return
            
        save_dir = Path(f"data/intermediate/{self.config.experiment_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"intermediate_{timestamp}.npz"
        
        np.savez_compressed(
            save_path,
            belief_history=np.array(self.belief_history),
            action_history=np.array(self.action_history),
            payoff_history=np.array(self.payoff_history)
        )
        
        logging.info(f"Saved intermediate results to {save_path}")
        
    def _save_final_results(self) -> None:
        """保存最终结果"""
        save_dir = Path(f"data/results/{self.config.experiment_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存网络数据
        nx.write_edgelist(self.network, save_dir / "network.edgelist")
        
        # 保存网络统计信息
        with open(save_dir / "network_stats.json", "w") as f:
            json.dump(self.network_stats, f, indent=4)
        
        # 保存实验数据
        np.savez_compressed(
            save_dir / "simulation_data.npz",
            belief_history=np.array(self.belief_history),
            action_history=np.array(self.action_history),
            payoff_history=np.array(self.payoff_history)
        )
        
        # 保存收敛统计信息
        with open(save_dir / "convergence_stats.json", "w") as f:
            json.dump(self.convergence_stats, f, indent=4)
            
        # 保存配置信息
        self.config.save(save_dir / "config.yaml")
        
        logging.info(f"Results saved to {save_dir}")
        
    def get_results(self) -> Tuple[np.ndarray, Dict, Dict]:
        """
        获取仿真结果
        
        Returns:
        --------
        Tuple[np.ndarray, Dict, Dict]
            - 信念历史数据
            - 收敛统计信息
            - 网络统计信息
        """
        return (
            np.array(self.belief_history), 
            self.convergence_stats,
            self.network_stats
        )