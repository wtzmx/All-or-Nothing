import numpy as np
from typing import List, Dict, Tuple, Set
from src.models.agent import Agent

class PublicGoodsGame:
    """
    实现all-or-nothing公共品博弈的核心逻辑
    处理智能体互动、收益计算和信念更新
    """
    def __init__(self, 
                 n_agents: int,
                 learning_rate: float = 0.3,
                 initial_belief: float = 0.5,
                 lambda_dist: str = "uniform",
                 lambda_params: Dict = {"low": 0.0, "high": 2.0}) -> None:
        # 输入验证
        if n_agents <= 0:
            raise ValueError("Number of agents must be positive")
        if not 0 < learning_rate < 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if not 0 <= initial_belief <= 1:
            raise ValueError("Initial belief must be between 0 and 1")
        """
        初始化博弈环境
        
        Parameters:
        -----------
        n_agents : int
            总智能体数量N
        learning_rate : float
            信念更新的学习率α
        initial_belief : float
            初始信念
        lambda_dist : str
            λ的分布类型，可选 "uniform" 或 "normal"
        lambda_params : Dict
            λ分布的参数
        """
        self.N = n_agents
        self.agents = [
            Agent(agent_id=i, 
                 initial_belief=initial_belief,
                 learning_rate=learning_rate)
            for i in range(n_agents)
        ]
        self.lambda_dist = lambda_dist
        self.lambda_params = lambda_params
        self.round_count = 0
        
    def _generate_lambda(self) -> float:
        """生成λ值"""
        if self.lambda_dist == "uniform":
            return np.random.uniform(
                self.lambda_params["low"],
                self.lambda_params["high"]
            )
        elif self.lambda_dist == "normal":
            return np.random.normal(
                self.lambda_params["mean"],
                self.lambda_params["std"]
            )
        else:
            raise ValueError(f"Unknown distribution: {self.lambda_dist}")
            
    def play_round(self, players: Set[int]) -> Tuple[Dict[int, str], Dict[int, float]]:
        """
        执行一轮博弈
        
        Parameters:
        -----------
        players : Set[int]
            参与本轮博弈的智能体ID集合 (K_t)
        
        Returns:
        --------
        Tuple[Dict[int, str], Dict[int, float]]
            - 智能体的行动 (C或D)
            - 智能体获得的收益
        """
        self.round_count += 1
        # 复制players集合以避免修改原始集合
        current_players = players.copy()
        group_size = len(current_players)
        
        # 1. 每个智能体观察自己的λ并做出决策
        lambda_values = {i: self._generate_lambda() for i in current_players}
        actions = {}
        
        for player_id in current_players:
            agent = self.agents[player_id]
            action = agent.decide_action(
                lambda_i=lambda_values[player_id],
                group_size=group_size
            )
            actions[player_id] = action
            
        # 2. 计算收益
        payoffs = self._calculate_payoffs(actions, lambda_values)
        
        # 3. 更新信念
        self._update_beliefs(current_players, actions)
        
        return actions, payoffs
    
    def _calculate_payoffs(self, 
                          actions: Dict[int, str],
                          lambda_values: Dict[int, float]) -> Dict[int, float]:
        """计算每个智能体的收益"""
        payoffs = {}
        
        # 检查所有人是否都贡献
        all_contributed = all(action == 'C' for action in actions.values())
        
        for player_id, action in actions.items():
            if action == 'D':
                # 背叛者总是获得1的收益
                payoffs[player_id] = 1.0
            else:  # action == 'C'
                if all_contributed:
                    # 如果所有人都贡献，获得λ值
                    payoffs[player_id] = lambda_values[player_id]
                else:
                    # 如果有人背叛，贡献者获得0
                    payoffs[player_id] = 0.0
                    
        # 校验收益合法性
        assert all(p >= 0 for p in payoffs.values()), "All payoffs must be non-negative"
        assert all(p == 1.0 for p, a in zip(payoffs.values(), actions.values()) if a == 'D'), \
            "All defectors must get payoff of 1.0"
                    
        return payoffs
    
    def _update_beliefs(self,
                       players: Set[int],
                       actions: Dict[int, str]) -> None:
        """更新每个智能体的信念"""
        for player_id in players:
            # 获取除自己外其他智能体的行动
            others_actions = [
                actions[i] for i in players 
                if i != player_id
            ]
            self.agents[player_id].update_belief(others_actions)
    
    def get_all_beliefs(self) -> List[float]:
        """获取所有智能体当前的信念"""
        return [agent.get_belief() for agent in self.agents]
    
    def get_agent(self, agent_id: int) -> Agent:
        """获取指定ID的智能体"""
        return self.agents[agent_id]
    
    def is_converged(self, epsilon: float = 1e-4) -> bool:
        """
        检查系统是否已收敛
        当所有智能体都收敛到相同的角落状态时返回True
        """
        beliefs = self.get_all_beliefs()
        
        # 检查是否都收敛到0附近
        all_defect = all(b <= epsilon for b in beliefs)
        if all_defect:
            return True
            
        # 检查是否都收敛到1附近
        all_contribute = all(b >= 1 - epsilon for b in beliefs)
        if all_contribute:
            return True
            
        return False
        
    def get_stats(self) -> Dict:
        """获取当前博弈的统计信息"""
        beliefs = self.get_all_beliefs()
        return {
            "round": self.round_count,
            "mean_belief": np.mean(beliefs),
            "std_belief": np.std(beliefs),
            "min_belief": np.min(beliefs),
            "max_belief": np.max(beliefs),
        }
    
    def get_beliefs(self) -> List[float]:
        """获取所有智能体的当前信念"""
        return [agent.belief for agent in self.agents]
    
    def check_convergence(self, threshold: float) -> bool:
        """
        检查是否达到收敛
        
        Parameters:
        -----------
        threshold : float
            收敛阈值
            
        Returns:
        --------
        bool : 是否收敛
        """
        beliefs = self.get_beliefs()
        # 检查是否所有信念都接近0或1
        return all(b <= threshold or b >= 1 - threshold 
                  for b in beliefs)