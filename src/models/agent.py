import numpy as np
from typing import List, Tuple, Optional

class Agent:
    """
    实现论文中的智能体类
    每个智能体维护一个信念状态，并能在博弈中做出贡献或背叛的决策
    """
    def __init__(self, 
                 agent_id: int,
                 initial_belief: float = 0.5,
                 learning_rate: float = 0.3) -> None:
        """
        初始化智能体
        
        Parameters:
        -----------
        agent_id : int
            智能体的唯一标识
        initial_belief : float, optional (default=0.5)
            初始信念，即对其他智能体会贡献的概率估计
        learning_rate : float, optional (default=0.3)
            学习率α，用于信念更新
        """
        self.id = agent_id
        self.belief = initial_belief  # x_i(t)
        self.alpha = learning_rate
        
        # 用于追踪智能体的历史状态
        self.action_history: List[str] = []
        self.belief_history: List[float] = []
        self.lambda_history: List[float] = []
        
    def decide_action(self, lambda_i: float, group_size: int) -> str:
        """
        基于当前信念和潜在收益决定是否贡献
        
        Parameters:
        -----------
        lambda_i : float
            如果所有人都贡献时，该智能体获得的收益
        group_size : int
            当前博弈组的大小k_t
            
        Returns:
        --------
        str : 'C' for contribute, 'D' for defect
        """
        # 记录lambda值用于分析
        self.lambda_history.append(lambda_i)
        
        try:
            # 根据论文公式计算期望效用
            # 其他group_size-1个智能体都需要贡献才能获得收益
            others_contribute_prob = self.belief ** (group_size - 1)
            expected_utility_contribute = lambda_i * others_contribute_prob
            expected_utility_defect = 1.0  # 背叛的收益固定为1

            # 打印调试信息
            # print(f"Debug: belief={self.belief}, group_size={group_size}, lambda={lambda_i}")
            # print(f"Debug: others_prob={others_contribute_prob}, exp_utility={expected_utility_contribute}")
            
            # 做出决策：如果贡献的期望效用大于等于背叛的效用，则选择贡献
            action = 'C' if expected_utility_contribute >= expected_utility_defect else 'D'
            
        except Exception as e:
            # 如果计算过程出现问题（如数值溢出），选择安全的背叛策略
            print(f"Warning: Decision calculation failed - {str(e)}")
            action = 'D'
            
        self.action_history.append(action)
        return action
    
    def update_belief(self, observed_actions: List[str]) -> None:
        """
        根据观察到的其他智能体行为更新��念
        使用指数移动平均(EMA)更新规则
        
        Parameters:
        -----------
        observed_actions : List[str]
            在当前轮次中观察到的其他智能体的行动
        """
        if not observed_actions:  # 如果没有观察到行动，保持信念不变
            return
        
        # 计算观察到的贡献比例
        contribute_ratio = sum(1 for a in observed_actions if a == 'C') / len(observed_actions)
        
        # 使用EMA更新规则更新信念
        self.belief = self.belief * (1 - self.alpha) + self.alpha * contribute_ratio
        
        # 记录信念历史
        self.belief_history.append(self.belief)
    
    def get_belief(self) -> float:
        """获取当前信念值"""
        return self.belief
    
    def get_history(self) -> Tuple[List[str], List[float], List[float]]:
        """获取智能体的历史记录"""
        return self.action_history, self.belief_history, self.lambda_history
    
    def is_absorbed(self, epsilon: float) -> Optional[bool]:
        """
        检查智能体是否已经收敛到某个角落状态
        
        Parameters:
        -----------
        epsilon : float
            判断收敛的阈值
            
        Returns:
        --------
        Optional[bool] : 
            True表示收敛到贡献状态(x ≈ 1)
            False表示收敛到背叛状态(x ≈ 0)
            None表示未收敛
        """
        if self.belief <= epsilon:
            return False
        elif self.belief >= 1 - epsilon:
            return True
        return None