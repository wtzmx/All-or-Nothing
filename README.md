# All-or-Nothing Public Goods Game Simulation

基于网络的全或无公共品博弈多智能体学习仿真项目，研究集体行为和合作动力学。

## 项目特点

- 多种网络类型支持(随机几何图、规则图、ER随机图、WS小世界网络、BA无标度网络)
- 完整的实验流程(配置、运行、分析、可视化)
- 灵活的配置系统
- 全面的数据分析
- 丰富的可视化功能
- 完整的测试覆盖

## 项目结构

```
project/
├── src/                         # 源代码目录
│   ├── models/                  # 模型核心实现
│   │   ├── agent.py            # 智能体类定义
│   │   ├── game.py             # 博弈游戏逻辑
│   │   └── belief_update.py    # 信念更新机制
│   ├── networks/               # 网络生成与分析
│   │   ├── geometric.py        # 随机几何图生成
│   │   ├── regular.py          # 规则图生成
│   │   ├── random.py           # ER随机图生成
│   │   ├── small_world.py      # WS小世界网络生成
│   │   ├── scale_free.py       # BA无标度网络生成
│   │   └── metrics.py          # 网络特征计算
│   ├── simulation/             # 仿真实验
│   │   ├── config.py           # 实验配置
│   │   └── runner.py           # 仿真运行器
│   └── visualization/          # 可视化模块
│       ├── network_plots.py    # 网络可视化
│       └── analysis_plots.py   # 结果分析图
├── experiments/                # 实验实现
│   ├── experiment1/           # 随机几何图上的合作演化
│   │   ├── exp1_runner.py     # 实验运行器
│   │   ├── exp1_analysis.py   # 数据分析
│   │   ├── exp1_visualization.py # 结果可视化
│   │   └── exp1_config.yaml   # 实验配置
│   ├── experiment2/           # 规则图上的群体动力学
│   │   ├── exp2_runner.py
│   │   ├── exp2_analysis.py
│   │   ├── exp2_visualization.py
│   │   └── exp2_config.yaml
│   └── experiment3/           # 网络结构对比研究
│       ├── exp3_runner.py
│       ├── exp3_analysis.py
│       ├── exp3_visualization.py
│       └── exp3_config.yaml
├── tests/                     # 测试套件
│   ├── models/               # 模型测试
│   │   ├── test_agent.py
│   │   └── test_game.py
│   ├── networks/             # 网络测试
│   │   ├── test_geometric.py
│   │   ├── test_regular.py
│   │   └── test_metrics.py
│   ├── simulation/           # 仿真测试
│   │   ├── test_config.py
│   │   └── test_runner.py
│   ├── visualization/        # 可视化测试
│   │   ├── test_network_plots.py
│   │   └── test_analysis_plots.py
│   └── experiments/          # 实验测试
│       ├── experiment1/
│       ├── experiment2/
│       └── experiment3/
├── data/                     # 数据目录
│   ├── experiment1/         # 实验一数据
│   │   ├── analysis/       # 分析结果
│   │   └── figures/        # 可视化图表
│   ├── experiment2/         # 实验二数据
│   │   ├── analysis/
│   │   └── figures/
│   └── experiment3/         # 实验三数据
│       ├── analysis/
│       └── figures/
├── notebooks/                # Jupyter notebooks
│   ├── analysis.ipynb       # 数据分析笔记本
│   └── exploration.ipynb    # 探索性分析
├── scripts/                 # 运行脚本
│   └── run_experiment.py    # 实验运行入口
├── requirements.txt         # 项目依赖
└── pytest.ini              # pytest配置文件
```

## 快速开始

### 安装
```bash
git clone https://github.com/username/all-or-nothing.git
cd all-or-nothing
pip install -r requirements.txt
```

### 运行实验
每个实验目录(experiment1/2/3)下包含四个主要文件：
- `exp{N}_config.yaml`: 实验配置文件
- `exp{N}_runner.py`: 实验运行脚本
- `exp{N}_analysis.py`: 数据分析脚本
- `exp{N}_visualization.py`: 结果可视化脚本

您可以按以下顺序运行实验（以实验一为例）：

```bash
# 1. 运行实验（生成原始数据）
python experiments/experiment1/exp1_runner.py

# 2. 分析数据
python experiments/experiment1/exp1_analysis.py

# 3. 生成可视化结果
python experiments/experiment1/exp1_visualization.py
```

实验结果将保存在以下位置：
- 原始数据：`data/experiment{N}/`
- 分析结果：`data/experiment{N}/analysis/`
- 可视化图表：`data/experiment{N}/figures/`

其中N为实验编号(1-3)。

### 实验内容

| 实验 | 研究内容 | 主要结果 |
|-----|---------|---------|
| 实验一 | 随机几何图上的合作演化 | - 收敛时间分布<br>- 网络密度影响<br>- 元稳态特征 |
| 实验二 | 规则图上的群体动力学 | - 邻居数量影响<br>- 同步化现象<br>- 稳定性分析 |
| 实验三 | 网络结构对比研究 | - 结构特征对比<br>- 合作水平分析<br>- 收敛性能比较 |

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/models/          # 测试模型
pytest tests/networks/        # 测试网络
pytest tests/simulation/      # 测试仿真
pytest tests/visualization/   # 测试可视化
pytest tests/experiments/     # 测试实验
```

## 项目进度

- [x] 核心模型实现
- [x] 多种网络类型实现
- [x] 三个实验完整实现
- [x] 完整测试套件
- [ ] 并行性能优化
- [ ] API文档
- [ ] 使用教程

## 许可证

[MIT License](LICENSE)
