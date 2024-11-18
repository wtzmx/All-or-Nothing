# All-or-Nothing Public Goods Game Simulation

基于网络的全或无公共品博弈多智能体学习仿真项目，研究集体行为和合作动力学。

## 项目特点

- 多种网络类型支持(随机几何图、规则图、ER随机图、WS小世界网络、BA无标度网络)
- 完整的实验流程(配置、运行、分析、可视化)
- 灵活的配置系统
- 全面的数据分析
- 丰富的可视化功能
- 完整的测试覆盖

## 核心组件

### 1. 模型实现 (`src/models/`)

#### Agent类 (`agent.py`)
智能体核心功能:
- 信念维护和更新(EMA规则)
- 基于期望效用的决策机制
- 行动和信念历史记录
- 收敛状态检测

#### PublicGoodsGame类 (`game.py`)
游戏环境管理:
- 多智能体环境协调
- λ值生成(多种分布)
- 博弈回合执行
- 收益计算
- 群体信念更新

### 2. 网络模块 (`src/networks/`)

| 网络类型 | 文件 | 主要特点 |
|---------|------|---------|
| 随机几何图 | `geometric.py` | - 支持r_g参数配置<br>- 确保连通性 |
| 规则图 | `regular.py` | - k-规则图生成<br>- 环形布局支持 |
| ER随机图 | `random.py` | - 连接概率p配置<br>- 确保连通性 |
| WS小世界 | `small_world.py` | - 重连概率β配置<br>- 小世界特征计算 |
| BA无标度 | `scale_free.py` | - 优先连接机制<br>- 幂律分布特征 |

### 3. 实验模块 (`experiments/`)

每个实验包含四个核心组件:
- 配置系统 (`exp*_config.yaml`)
- 运行器 (`exp*_runner.py`)
- 分析器 (`exp*_analysis.py`)
- 可视化器 (`exp*_visualization.py`)

#### 实验内容

| 实验 | 研究内容 | 主要结果 |
|-----|---------|---------|
| 实验一 | 随机几何图上的合作演化 | - 收敛时间分布<br>- 网络密度影响<br>- 元稳态特征 |
| 实验二 | 规则图上的群体动力学 | - 邻居数量影响<br>- 同步化现象<br>- 稳定性分析 |
| 实验三 | 网络结构对比研究 | - 结构特征对比<br>- 合作水平分析<br>- 收敛性能比较 |

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

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_networks/
pytest tests/experiment1/
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
