# All-or-Nothing Public Goods Game Simulation

这个项目实现了基于网络的全或无公共品博弈(All-or-Nothing Public Goods Game)多智能体学习仿真，基于集体行为和合作动力学的理论模型。

## 项目结构

```
project/
├── README.md                     # 项目说明文档
├── requirements.txt              # 项目依赖
├── pytest.ini                    # pytest配置文件
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── models/                  # 模型核心实现
│   │   ├── __init__.py
│   │   ├── agent.py            # 智能体类定义
│   │   ├── game.py             # 博弈游戏逻辑
│   │   └── belief_update.py    # 信念更新机制
│   ├── networks/               # 网络生成与分析
│   │   ├── __init__.py 
│   │   ├── geometric.py        # 随机几何图生成
│   │   ├── regular.py          # 规则图生成
│   │   └── metrics.py          # 网络特征计算
│   ├── simulation/             # 仿真实验
│   │   ├── __init__.py
│   │   ├── config.py           # 实验配置
│   │   └── runner.py           # 仿真运行器
│   └── visualization/          # 可视化模块
│       ├── __init__.py
│       ├── network_plots.py    # 网络可视化
│       └── analysis_plots.py   # 结果分析图
├── experiments/                # 实验配置和结果
│   ├── experiment1/           # 实验一：随机几何图上的合作演化
│   │   ├── exp1_config.yaml   # 实验配置
│   │   ├── exp1_runner.py     # 实验运行器
│   │   ├── exp1_analysis.py   # 实验分析
│   │   └── exp1_visualization.py # 可视化
│   ├── experiment2/           # 实验二：规则图上的群体动力学
│   └── experiment3/           # 实验三：网络结构对比研究
├── data/                      # 数据存储
├── notebooks/                 # Jupyter notebooks
└── tests/                    # 单元测试
    ├── experiment1/          # 实验一测试
    │   ├── test_exp1_config.py  # 配置测试
    │   └── test_exp1_runner.py  # 运行器测试
    └── ...
```

## 已实现组件

### 1. Agent类 (`agent.py`)
智能体实现了以下功能：
- 信念维护和更新(EMA规则)
- 基于期望效用的决策机制
- 行动和信念历史记录
- 收敛状态检测

示例用法：
```python
agent = Agent(agent_id=1, initial_belief=0.5, learning_rate=0.3)
action = agent.decide_action(lambda_i=2.0, group_size=3)  # 返回 'C' 或 'D'
agent.update_belief(['C', 'D', 'C'])  # 基于观察更新信念
```

### 2. PublicGoodsGame类 (`game.py`)
游戏环境管理：
- 多智能体环境协调
- λ值生成(支持多种分布)
- 博弈回合执行
- 收益计算
- 群体信念更新

示例用法：
```python
game = PublicGoodsGame(
    n_agents=5,
    learning_rate=0.3,
    initial_belief=0.5,
    lambda_dist="uniform",
    lambda_params={"low": 0.0, "high": 2.0}
)
actions, payoffs = game.play_round({0, 1, 2})  # 执行一个回合
```

### 3. 配置系统 (`config.py`)
实验配置管理：
- 网络参数配置(NetworkConfig)
- 博弈参数配置(GameConfig)
- 仿真参数配置(SimulationConfig)
- 完整实验配置(ExperimentConfig)

示例用法：
```python
# 从配置文件加载
config = ExperimentConfig.from_file("experiments/config.yaml")

# 从字典创建
config_dict = {
    "network": {
        "type": "geometric",
        "n_agents": 50,
        "r_g": 0.3
    },
    "game": {
        "learning_rate": 0.3,
        "initial_belief": 0.5,
        "lambda_dist": "uniform",
        "lambda_params": {"low": 0.0, "high": 2.0}
    },
    "simulation": {
        "max_rounds": 1000000,
        "convergence_threshold": 1e-4,
        "save_interval": 1000
    },
    "experiment_name": "test_experiment"
}
config = ExperimentConfig.from_dict(config_dict)

# 保存配置
config.save("experiments/new_config.yaml")
```

### 4. 网络分析工具 (`metrics.py`)
网络分析工具提供了全面的网络特征计算功能：
- 度数统计（最大度、最小度、平均度、标准差）
- 三角形计数
- 聚类系数（全局和局部）
- 路径长度分析（平均路径长度、直径）
- 连通性检查
- 完整网络统计信息生成

### 5. 可视化模块
#### 5.1 网络可视化 (`network_plots.py`)
NetworkVisualizer 类提供了以下功能：
- 网络结构可视化（支持自定义节点位置、颜色和大小）
- 信念分布直方图
- 网络状态完整视图（结构+分布）
- 信念演化过程动态图

#### 5.2 结果分析图 (`analysis_plots.py`)
AnalysisPlotter 类实现了论文中的关键图表：
- 收敛时间尾概率分布图（Figure 2）
- 信念演化热力图
- 网络指标变化图
- 收敛分析图（Table 1）
- 灾难原理比率图（Table 2）

特点：
- 支持自定义样式和参数
- 灵活的轴对象管理
- 完善的边界情况处理
- 支持大规模数据可视化

### 6. 实验一：随机几何图上的合作演化分析

实验一实现了论文中关于随机几何网络上的合作演化研究，包括完整的实验流程和分析工具。

#### 6.1 随机几何网络生成 (`src/networks/geometric.py`)
- 实现随机几何图生成算法
- 确保网络连通性
- 支持参数r_g配置
- 计算网络特征(平均度、三角形数等)

示例用法：
```python
network = RandomGeometricGraph(
    n_nodes=50,  # 节点数量
    radius=0.3,  # 连接半径r_g
    seed=42      # 随机种子
)

# 获取网络特征
stats = network.get_stats()
# 获取邻居集合
neighbors = network.get_closed_neighbors(node_id=0)
```

#### 6.2 实验配置系统 (`experiments/experiment1/exp1_config.yaml`)
完整的实验参数配置：
- 网络参数：N=50, r_g∈[0.15, 0.2, 0.25, 0.3]
- 博弈参数：α=0.3, F(x)=x^(1/4)
- 仿真参数：T=10^7, ε_s=10^(-4)
- 实验重复：每个r_g值500次
- 支持并行计算和数据存储配置

示例用法：
```python
# 加载实验配置
with open("experiments/experiment1/exp1_config.yaml", 'r') as f:
    config = yaml.safe_load(f)
```

#### 6.3 实验运行器 (`experiments/experiment1/exp1_runner.py`)
实验执行和数据收集：
- 支持单次和并行实验运行
- 实现收敛状态检测
- 记录实验数据：
  - 收敛时间
  - 最终状态(贡献/背叛/未收敛)
  - 网络特征
  - 信念演化过程
- 灵活的数据保存机制

示例用法：
```python
# 运行实验
runner = ExperimentRunner("experiments/experiment1/exp1_config.yaml")
runner.run_experiment()
```

#### 6.4 数据分析工具 (`experiments/experiment1/exp1_analysis.py`)
全面的数据分析功能：
- 收敛时间尾概率分布计算
- 不同r_g值下的收敛状态统计
- 网络特征与最终状态关系分析
- 元稳态特征分析

示例用法：
```python
# 运行分析
analyzer = ExperimentAnalyzer("data/experiment1")
analyzer.save_analysis_results()

# 获取特定分析结果
tail_probs = analyzer.compute_tail_probabilities()
conv_states = analyzer.analyze_convergence_states()
```

#### 6.5 可视化工具 (`experiments/experiment1/exp1_visualization.py`)
论文图表复现：
- Figure 2：收敛时间尾概率分布图
- Table 1：收敛状态统计表
- 网络状态可视化
- 信念演化热力图

示例用法：
```python
# 创建可视化器
visualizer = ExperimentVisualizer("data/experiment1/analysis")

# 生成所有图表
visualizer.save_all_figures("data/experiment1/figures")

# 生成特定图表
visualizer.plot_tail_probabilities("figures/tail_prob.png")
visualizer.generate_convergence_table("tables/conv_stats.csv")
```

#### 6.6 测试覆盖
完整的测试套件：
- 网络生成测试
- 配置系统测试
- 运行器功能测试
- 分析工具测试
- 可视化功能测试

运行测试：
```bash
# 运行所有测试
pytest tests/experiment1/

# 运行特定模块测试
pytest tests/experiment1/test_exp1_runner.py -v
pytest tests/experiment1/test_exp1_analysis.py -v
pytest tests/experiment1/test_exp1_visualization.py -v
```

## 当前进度

已完成:
- ✓ 核心模型实现
- ✓ 随机几何网络生成
- ✓ 实验一配置系统
- ✓ 实验一运行器
- ✓ 完整测试套件

进行中:
- 实验一分析模块
- 实验一可视化
- 并行性能优化

待开始:
- 实验二：规则图研究
- 实验三：网络结构对比
- 完整文档编写

## 下一步工作

1. 实验一后续工作：
   - 实现分析模块(exp1_analysis.py)
   - 实现可视化模块(exp1_visualization.py)
   - 优化并行计算性能

2. 实验二准备：
   - 实现规则图生成
   - 设计实验配置
   - 编写运行器

3. 文档完善：
   - API文档
   - 实验设计文档
   - 使用教程

## 贡献指南

1. Fork 仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

[MIT License](LICENSE)
