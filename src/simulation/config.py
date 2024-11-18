from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Literal, Union
import yaml
import json
import numpy as np
from pathlib import Path

@dataclass
class NetworkConfig:
    """网络配置参数"""
    type: Literal["geometric", "regular"]  # 网络类型
    n_agents: int = 50                     # 智能体数量
    seed: Optional[int] = None             # 网络生成的随机种子
    
    # 随机几何图参数
    r_g: Optional[float] = None           # 单个几何图连接半径
    radius_list: Optional[list] = None    # 多个几何图连接半径列表
    
    # 规则图参数
    degree: Optional[int] = None          # 规则图度数
    
    def __post_init__(self):
        """初始化后立即验证网络类型和基本参数"""
        if self.type not in ["geometric", "regular"]:
            raise ValueError(f"Unknown network type: {self.type}")
            
        # 立即验证网络特定参数
        self.validate()
    
    def validate(self) -> None:
        """验证网络配置参数的合法性"""
        if self.n_agents <= 0:
            raise ValueError("Number of agents must be positive")
            
        if self.type == "geometric":
            # 检查是否至少提供了一种半径配置
            if self.r_g is None and self.radius_list is None:
                raise ValueError("Either r_g or radius_list must be specified for geometric network")
                
            # 验证单个半径
            if self.r_g is not None:
                if not isinstance(self.r_g, (int, float)):
                    raise ValueError("r_g must be a number")
                if self.r_g <= 0 or self.r_g >= np.sqrt(2):
                    raise ValueError("r_g must be between 0 and sqrt(2)")
                    
            # 验证半径列表
            if self.radius_list is not None:
                if not isinstance(self.radius_list, (list, tuple)):
                    raise ValueError("radius_list must be a list or tuple")
                if not self.radius_list:
                    raise ValueError("radius_list cannot be empty")
                for r in self.radius_list:
                    if not isinstance(r, (int, float)):
                        raise ValueError("All values in radius_list must be numbers")
                    if r <= 0 or r >= np.sqrt(2):
                        raise ValueError("All values in radius_list must be between 0 and sqrt(2)")
                        
        elif self.type == "regular":
            if self.degree is None:
                raise ValueError("degree must be specified for regular network")
            if self.degree <= 0 or self.degree >= self.n_agents:
                raise ValueError("Invalid degree for regular network")
            if self.degree % 2 != 0:
                raise ValueError("degree must be even for regular network")
                
    def get_radius_values(self) -> list:
        """获取所有要使用的半径值"""
        if self.type != "geometric":
            return []
            
        if self.radius_list is not None:
            return self.radius_list
        elif self.r_g is not None:
            return [self.r_g]
        else:
            return []

@dataclass
class GameConfig:
    """博弈参数配置"""
    learning_rate: float = 0.3            # 学习率α
    initial_belief: float = 0.5           # 初始信念
    
    # λ分布参数
    lambda_dist: Literal["uniform", "normal"] = "uniform"
    lambda_params: Dict = field(default_factory=lambda: {"low": 0.0, "high": 2.0})
    
    def __post_init__(self):
        """初始化后立即验证分布类型"""
        if self.lambda_dist not in ["uniform", "normal"]:
            raise ValueError(f"Unknown lambda distribution: {self.lambda_dist}")
    
    def validate(self) -> None:
        """验证博弈参数的合法性"""
        if not 0 < self.learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1")
            
        if not 0 <= self.initial_belief <= 1:
            raise ValueError("Initial belief must be between 0 and 1")
            
        if self.lambda_dist == "uniform":
            if "low" not in self.lambda_params or "high" not in self.lambda_params:
                raise ValueError("Uniform distribution requires 'low' and 'high' parameters")
            if self.lambda_params["low"] >= self.lambda_params["high"]:
                raise ValueError("'low' must be less than 'high' for uniform distribution")
            if self.lambda_params["low"] < 0:
                raise ValueError("Lambda parameters must be non-negative")
                
        elif self.lambda_dist == "normal":
            if "mean" not in self.lambda_params or "std" not in self.lambda_params:
                raise ValueError("Normal distribution requires 'mean' and 'std' parameters")
            if self.lambda_params["std"] <= 0:
                raise ValueError("Standard deviation must be positive")
            if self.lambda_params["mean"] < 0:
                raise ValueError("Mean lambda must be non-negative")

@dataclass
class SimulationConfig:
    """仿真参数配置"""
    max_rounds: int = 10_000_000          # 最大仿真轮次
    convergence_threshold: float = 1e-4    # 收敛阈值ε
    save_interval: int = 1000             # 结果保存间隔
    seed: Optional[int] = None            # 随机数种子
    
    def __post_init__(self):
        """初始化后立即验证基本参数"""
        if self.max_rounds <= 0:
            raise ValueError("Maximum rounds must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")
    
    def validate(self) -> None:
        """验证仿真参数的合法性"""
        if self.max_rounds <= 0:
            raise ValueError("Maximum rounds must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")
        if self.save_interval > self.max_rounds:
            raise ValueError("Save interval cannot be larger than maximum rounds")

@dataclass
class ExperimentConfig:
    """完整实验配置"""
    network: NetworkConfig
    game: GameConfig
    simulation: SimulationConfig
    experiment_name: str
    
    def __post_init__(self):
        """初始化后立即验证实验名称"""
        if not self.experiment_name:
            raise ValueError("Experiment name cannot be empty")
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """从字典创建配置"""
        if "experiment_name" not in config_dict:
            raise ValueError("Experiment name must be specified")
            
        return cls(
            network=NetworkConfig(**config_dict.get("network", {})),
            game=GameConfig(**config_dict.get("game", {})),
            simulation=SimulationConfig(**config_dict.get("simulation", {})),
            experiment_name=config_dict["experiment_name"]
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'ExperimentConfig':
        """从配置文件加���配置"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        if file_path.suffix in {'.yaml', '.yml'}:
            with open(file_path) as f:
                config_dict = yaml.safe_load(f)
        elif file_path.suffix == '.json':
            with open(file_path) as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml or .json")
            
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """将配置转换为字典"""
        return {
            "network": asdict(self.network),
            "game": asdict(self.game),
            "simulation": asdict(self.simulation),
            "experiment_name": self.experiment_name
        }
    
    def save(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if file_path.suffix in {'.yaml', '.yml'}:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml or .json")
    
    def validate(self) -> None:
        """验证所有配置参数"""
        self.network.validate()
        self.game.validate()
        self.simulation.validate()