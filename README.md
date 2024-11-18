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
│   ├── experiment2/           # 规则图上的群体动力学
│   └── experiment3/           # 网络结构对比研究
├── data/                      # 数据存储
│   ├── experiment1/          
│   ├── experiment2/          
│   └── experiment3/          
├── tests/                    # 单元测试
│   ├── test_networks/        # 网络模块测试
│   ├── experiment1/          # 实验一测试
│   ├── experiment2/          # 实验二测试
│   └── experiment3/          # 实验三测试
├── notebooks/                 # Jupyter notebooks
├── requirements.txt           # 项目依赖
└── pytest.ini                # pytest配置文件
```

## 核心组件

### 1. 模型实现 (`src/models/`)

| 组件 | 文件 | 主要功能 |
|-----|------|---------|
| Agent类 | `agent.py` | - 信念维护和更新<br>- 决策机制<br>- 历史记录 |
| Game类 | `game.py` | - 环境协调<br>- 博弈执行<br>- 收益计算 |

### 2. 网络模块 (`src/networks/`)

| 网络类型 | 文件 | 主要特点 |
|---------|------|---------|
| 随机几何图 | `geometric.py` | - r_g参数配置<br>- 确保连通性 |
| 规则图 | `regular.py` | - k-规则图生成<br>- 环形布局 |
| ER随机图 | `random.py` | - 连接概率p配置 |
| WS小世界 | `small_world.py` | - 重连概率β配置 |
| BA无标度 | `scale_free.py` | - 优先连接机制 |

### 3. 实验模块 (`experiments/`)

| 实验 | 研究内容 | 主要结果 |
|-----|---------|---------|
| 实验一 | 随机几何图研究 | - 收敛时间分布<br>- 网络密度影响 |
| 实验二 | 规则图研究 | - 邻居数量影响<br>- 同步化现象 |
| 实验三 | 网络结构对比 | - 结构特征对比<br>- 性能比较 |

## 快速开始

### 安装
```bash
git clone https://github.com/username/all-or-nothing.git
cd all-or-nothing
pip install -r requirements.txt
```

### 运行实验
```python
# 1. 配置实验
from src.simulation import ExperimentConfig
config = ExperimentConfig.from_file("experiments/exp1_config.yaml")

# 2. 运行实验
from experiments.experiment1 import ExperimentRunner
runner = ExperimentRunner(config)
runner.run_experiment()

# 3. 分析结果
from experiments.experiment1 import ExperimentAnalyzer
analyzer = ExperimentAnalyzer("data/experiment1")
analyzer.save_analysis_results()

# 4. 可视化
from experiments.experiment1 import ExperimentVisualizer
visualizer = ExperimentVisualizer("data/experiment1/analysis")
visualizer.save_all_figures("data/experiment1/figures")
```

## 项目进度

- [x] 核心模型实现
- [x] 多种网络类型实现
- [x] 三个实验完整实现
- [x] 完整测试套件
- [ ] 并行性能优化
- [ ] API文档
- [ ] 使用教程

## 贡献指南

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

[MIT License](LICENSE)
