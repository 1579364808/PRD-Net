"""
灵活的配置管理工具

支持:
1. 命令行参数覆盖配置文件: --vmai.temperature=0.1 --base.lr=0.0001
2. 网格搜索: --grid "vmai.temperature=[0.05,0.07,0.1]"

用法示例:
    # 单次训练，覆盖参数
    python train.py --config_file configs/train_sims.yaml --vmai.temperature=0.1 --base.lr=0.0001
    
    # 网格搜索
    python grid_search.py --config_file configs/train_sims.yaml \
        --grid "vmai.codebook_size=[256,512,1024]" \
        --grid "vmai.temperature=[0.05,0.07,0.1]" \
        --grid "vmai.commitment_weight=[0.1,0.25,0.5]"
"""

import argparse
import copy
import itertools
import ast
from typing import Any, Dict, List, Tuple


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """深度更新字典"""
    result = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_nested_value(d: dict, key_path: str, value: Any) -> dict:
    """设置嵌套字典的值
    
    Args:
        d: 目标字典
        key_path: 点分隔的路径，如 "vmai.temperature" 或 "base.lr"
        value: 要设置的值
    
    Returns:
        更新后的字典
    """
    keys = key_path.split('.')
    result = copy.deepcopy(d)
    current = result
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return result


def parse_value(value_str: str) -> Any:
    """智能解析值的类型"""
    # 尝试解析为 Python 字面量
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        pass
    
    # 布尔值
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    
    # 保持为字符串
    return value_str


def parse_override_args(args: argparse.Namespace) -> Dict[str, Any]:
    """从 argparse.Namespace 中提取覆盖参数
    
    识别形如 --xxx.yyy=value 的参数
    """
    overrides = {}
    
    # 获取所有未知参数（通过 parse_known_args）
    if hasattr(args, '_overrides'):
        for key, value in args._overrides.items():
            overrides[key] = parse_value(value)
    
    return overrides


def create_argument_parser() -> argparse.ArgumentParser:
    """创建支持动态覆盖的参数解析器"""
    parser = argparse.ArgumentParser(
        description='灵活的训练配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 单次训练
  python train.py --config_file configs/train_sims.yaml --vmai.temperature=0.1
  
  # 多个参数覆盖
  python train.py --config_file configs/train_sims.yaml \\
      --vmai.temperature=0.1 \\
      --vmai.codebook_size=512 \\
      --base.lr=0.0001
'''
    )
    parser.add_argument('--config_file', type=str, required=True, help='基础配置文件')
    parser.add_argument('--seed', type=int, default=-1, help='随机种子，-1表示使用配置文件中的值')
    parser.add_argument('--grid', action='append', default=[], 
                        help='网格搜索参数，格式: "param.path=[value1,value2,...]"')
    parser.add_argument('--key_eval', type=str, default='Mult_acc_2', help='评估指标')
    parser.add_argument('--n_runs', type=int, default=3, help='每个配置运行次数')
    
    return parser


def parse_args_with_overrides(parser: argparse.ArgumentParser = None) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """解析参数，包括动态覆盖参数
    
    Returns:
        (args, overrides): 解析后的参数和覆盖字典
    """
    if parser is None:
        parser = create_argument_parser()
    
    # 先解析已知参数
    args, unknown = parser.parse_known_args()
    
    # 解析未知参数作为覆盖
    overrides = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--') and '=' in arg:
            # 格式: --key.path=value
            key, value = arg[2:].split('=', 1)
            overrides[key] = parse_value(value)
        elif arg.startswith('--') and i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
            # 格式: --key.path value
            key = arg[2:]
            value = unknown[i + 1]
            overrides[key] = parse_value(value)
            i += 1
        i += 1
    
    return args, overrides


def apply_overrides(config: dict, overrides: Dict[str, Any]) -> dict:
    """将覆盖参数应用到配置"""
    result = copy.deepcopy(config)
    for key_path, value in overrides.items():
        result = set_nested_value(result, key_path, value)
    return result


def parse_grid_params(grid_args: List[str]) -> Dict[str, List[Any]]:
    """解析网格搜索参数
    
    Args:
        grid_args: 格式如 ["vmai.temperature=[0.05,0.07,0.1]", "vmai.codebook_size=[256,512]"]
    
    Returns:
        {"vmai.temperature": [0.05, 0.07, 0.1], "vmai.codebook_size": [256, 512]}
    """
    grid_params = {}
    for grid_arg in grid_args:
        if '=' not in grid_arg:
            raise ValueError(f"网格参数格式错误: {grid_arg}，应为 'param.path=[value1,value2,...]'")
        
        key, values_str = grid_arg.split('=', 1)
        values = parse_value(values_str)
        
        if not isinstance(values, list):
            values = [values]
        
        grid_params[key] = values
    
    return grid_params


def generate_grid_configs(base_config: dict, grid_params: Dict[str, List[Any]]) -> List[Tuple[dict, str]]:
    """生成所有网格搜索配置组合
    
    Returns:
        [(config, name), ...]: 配置和对应的名称
    """
    if not grid_params:
        return [(base_config, "base")]
    
    # 生成所有组合
    keys = list(grid_params.keys())
    values_lists = [grid_params[k] for k in keys]
    
    configs = []
    for values in itertools.product(*values_lists):
        config = copy.deepcopy(base_config)
        name_parts = []
        
        for key, value in zip(keys, values):
            config = set_nested_value(config, key, value)
            # 生成简短名称
            short_key = key.split('.')[-1]
            name_parts.append(f"{short_key}={value}")
        
        name = "_".join(name_parts)
        configs.append((config, name))
    
    return configs

