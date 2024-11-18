import os
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

class TestExp2Config:
    """测试实验二配置文件的测试类"""
    
    @pytest.fixture
    def config_path(self) -> str:
        """配置文件路径fixture"""
        return "experiments/experiment2/exp2_config.yaml"
    
    @pytest.fixture
    def config_data(self, config_path: str) -> Dict[str, Any]:
        """加载配置数据fixture"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def test_config_file_exists(self, config_path: str):
        """测试配置文件是否存在"""
        assert os.path.exists(config_path), f"配置文件 {config_path} 不存在"
    
    def test_config_is_valid_yaml(self, config_data: Dict[str, Any]):
        """测试配置文件是否为有效的YAML格式"""
        assert isinstance(config_data, dict), "配置文件格式无效"
        assert len(config_data) > 0, "配置文件为空"
    
    def test_required_sections_exist(self, config_data: Dict[str, Any]):
        """测试必需的配置部分是否存在"""
        required_sections = [
            "experiment_name",
            "network",
            "game",
            "simulation",
            "output",
            "visualization",
            "parallel",
            "logging",
            "analysis"
        ]
        for section in required_sections:
            assert section in config_data, f"缺少必需的配置部分: {section}"
    
    def test_network_config(self, config_data: Dict[str, Any]):
        """测试网络配置部分"""
        network = config_data["network"]
        assert network["type"] == "regular", "网络类型必须为regular"
        assert network["n_agents"] == 50, "智能体数量必须为50"
        assert isinstance(network["l_values"], list), "l_values必须为列表"
        assert network["l_values"] == [2, 4, 6, 8], "l_values必须为[2,4,6,8]"
        assert isinstance(network["seed"], int), "seed必须为整数"
    
    def test_game_config(self, config_data: Dict[str, Any]):
        """测试博弈配置部分"""
        game = config_data["game"]
        assert game["learning_rate"] == 0.3, "学习率必须为0.3"
        assert game["initial_belief"] == 0.5, "初始信念必须为0.5"
        assert game["reward_function"]["type"] == "power", "奖励函数类型必须为power"
        assert game["reward_function"]["exponent"] == 0.25, "奖励函数指数必须为0.25"
    
    def test_simulation_config(self, config_data: Dict[str, Any]):
        """测试仿真配置部分"""
        sim = config_data["simulation"]
        assert sim["max_rounds"] == 10000000, "最大轮数必须为10^7"
        assert sim["convergence_threshold"] == 0.0001, "收敛阈值必须为10^-4"
        assert sim["n_trials"] == 500, "每个l值的重复次数必须为500"
        assert isinstance(sim["save_interval"], int), "save_interval必须为整数"
    
    def test_output_paths(self, config_data: Dict[str, Any]):
        """测试输出路径配置"""
        output = config_data["output"]
        assert "base_dir" in output, "必须指定基础输出目录"
        assert output["base_dir"] == "data/experiment2", "基础输出目录必须为data/experiment2"
        assert isinstance(output["formats"], list), "formats必须为列表"
        assert set(output["formats"]) == {"csv", "pickle"}, "formats必须包含csv和pickle"
    
    def test_visualization_config(self, config_data: Dict[str, Any]):
        """测试可视化配置"""
        vis = config_data["visualization"]
        assert isinstance(vis["plot_types"], list), "plot_types必须为列表"
        required_plots = {"tail_probability", "network_state", "belief_evolution"}
        assert set(vis["plot_types"]) == required_plots, f"必须包含所有必需的图表类型: {required_plots}"
        assert vis["figure_format"] == "png", "图表格式必须为png"
        assert isinstance(vis["dpi"], int), "dpi必须为整数"
    
    def test_parallel_config(self, config_data: Dict[str, Any]):
        """测试并行计算配置"""
        parallel = config_data["parallel"]
        assert isinstance(parallel["enabled"], bool), "enabled必须为布尔值"
        assert isinstance(parallel["n_processes"], int), "n_processes必须为整数"
        assert isinstance(parallel["chunk_size"], int), "chunk_size必须为整数"
    
    def test_logging_config(self, config_data: Dict[str, Any]):
        """测试日志配置"""
        logging = config_data["logging"]
        assert logging["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "无效的日志级别"
        assert isinstance(logging["save_to_file"], bool), "save_to_file必须为布尔值"
        assert logging["file_name"] == "experiment2.log", "日志文件名必须为experiment2.log"
    
    def test_analysis_config(self, config_data: Dict[str, Any]):
        """测试分析配置"""
        analysis = config_data["analysis"]
        assert isinstance(analysis["compute_features"], list), "compute_features必须为列表"
        assert isinstance(analysis["convergence_metrics"], list), "convergence_metrics必须为列表"
        assert isinstance(analysis["statistical_tests"], list), "statistical_tests必须为列表"
        
    def test_config_paths_exist(self, config_data: Dict[str, Any]):
        """测试配置中的路径是否有效"""
        base_dir = Path(config_data["output"]["base_dir"])
        assert not base_dir.is_file(), f"{base_dir} 不应该是文件"
        # 创建输出目录（如果不存在）
        base_dir.mkdir(parents=True, exist_ok=True)
        assert base_dir.exists(), f"无法创建输出目录 {base_dir}"