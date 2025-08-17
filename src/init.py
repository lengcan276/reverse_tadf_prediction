# src/__init__.py
"""
TADF/rTADF Prediction System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A machine learning system for predicting TADF and reverse-TADF properties
of organic molecules based on their electronic and structural features.

Modules:
    - preprocessing: Data cleaning and preparation
    - feature_engineering: Feature extraction and creation
    - models: Neural network and XGBoost model definitions
    - training: Model training utilities
    - evaluation: Model evaluation and metrics
    - interpretation: Model interpretability (SHAP, feature importance)
    - prediction: Inference on new molecules
    - utils: Utility functions and helpers
    - constants: Global constants and configurations
"""

__version__ = '1.0.0'
__author__ = 'Your Research Team'
__email__ = 'your.email@institution.edu'
__license__ = 'MIT'

import warnings
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============= 核心模块导入 =============
# 数据处理
from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

# 模型
from .models import MultiTaskTADFNet, XGBoostTADFPredictor

# 训练和评估
from .training import ModelTrainer
from .evaluation import (
    evaluate_model,
    calculate_metrics,
    plot_roc_curves,
    print_evaluation_report
)

# 解释和预测
from .interpretation import ModelInterpreter
from .prediction import predict_new_molecule

# 工具函数
from .utils import (
    load_config,
    setup_logging,
    save_results,
    load_model,
    save_model,
    set_random_seed,
    get_device,
    create_output_dirs
)

# 常量
from .constants import (
    TADF_THRESHOLD,
    OSCILLATOR_THRESHOLD,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    KEY_FEATURES,
    DATA_DIR,
    OUTPUT_DIR
)

# ============= 定义公开的API =============
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 数据处理
    'DataPreprocessor',
    'FeatureEngineer',
    
    # 模型
    'MultiTaskTADFNet',
    'XGBoostTADFPredictor',
    
    # 训练
    'ModelTrainer',
    
    # 评估
    'evaluate_model',
    'calculate_metrics',
    'plot_roc_curves',
    'print_evaluation_report',
    
    # 解释和预测
    'ModelInterpreter',
    'predict_new_molecule',
    
    # 工具函数
    'load_config',
    'setup_logging',
    'save_results',
    'load_model',
    'save_model',
    'set_random_seed',
    'get_device',
    'create_output_dirs',
    
    # 常用常量
    'TADF_THRESHOLD',
    'OSCILLATOR_THRESHOLD',
    'RANDOM_SEED',
    'KEY_FEATURES',
    
    # 便捷函数
    'get_version',
    'check_dependencies',
    'quick_setup',
    'run_pipeline'
]

# ============= 便捷函数 =============
def get_version():
    """返回包版本信息"""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__
    }

def check_dependencies():
    """检查必要的依赖是否安装"""
    dependencies = {
        'torch': ('PyTorch', '>=1.10.0'),
        'sklearn': ('scikit-learn', '>=1.0.0'),
        'xgboost': ('XGBoost', '>=1.5.0'),
        'shap': ('SHAP', '>=0.40.0'),
        'pandas': ('pandas', '>=1.3.0'),
        'numpy': ('numpy', '>=1.21.0'),
        'matplotlib': ('matplotlib', '>=3.4.0'),
        'seaborn': ('seaborn', '>=0.11.0'),
        'tqdm': ('tqdm', '>=4.62.0'),
        'yaml': ('PyYAML', '>=5.4.0'),
        'rdkit': ('RDKit', '>=2020.09.1')
    }
    
    missing = []
    installed = []
    
    for module, (name, version) in dependencies.items():
        try:
            mod = __import__(module)
            if hasattr(mod, '__version__'):
                installed.append(f"✓ {name} ({mod.__version__})")
            else:
                installed.append(f"✓ {name}")
        except ImportError:
            missing.append(f"✗ {name} {version}")
    
    print("Dependency Check:")
    print("-" * 40)
    
    for item in installed:
        print(item)
    
    if missing:
        print("\nMissing dependencies:")
        for item in missing:
            print(item)
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies satisfied! ✓")
    return True

def quick_setup(config_path='configs/config.yaml'):
    """快速设置环境"""
    import os
    
    # 加载配置
    config = load_config(config_path) if Path(config_path).exists() else {}
    
    # 设置随机种子
    seed = config.get('random_seed', RANDOM_SEED)
    set_random_seed(seed)
    
    # 创建输出目录
    output_dirs = create_output_dirs(config.get('output_dir', OUTPUT_DIR))
    
    # 设置日志
    logger = setup_logging(
        log_dir=config.get('log_dir', f"{OUTPUT_DIR}/logs")
    )
    
    # 获取设备
    device = get_device()
    
    logger.info("Environment setup completed")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directories: {output_dirs}")
    
    return {
        'config': config,
        'logger': logger,
        'device': device,
        'output_dirs': output_dirs
    }

def run_pipeline(data_path, config_path='configs/config.yaml', mode='train'):
    """运行完整的训练或预测管道"""
    from .pipeline import TADFPipeline
    
    pipeline = TADFPipeline(config_path)
    
    if mode == 'train':
        return pipeline.train(data_path)
    elif mode == 'predict':
        return pipeline.predict(data_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'predict'")

# ============= 包初始化 =============
def _initialize():
    """包初始化时的设置"""
    # 设置警告过滤
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # 检查是否在正确的环境中
    import platform
    python_version = platform.python_version()
    
    if sys.version_info < (3, 7):
        warnings.warn(
            f"Python {python_version} detected. "
            "This package requires Python 3.7 or higher.",
            UserWarning
        )
    
    # 可选：自动检查依赖
    if not Path('src/.dependencies_checked').exists():
        try:
            check_dependencies()
            # 创建标记文件
            Path('src/.dependencies_checked').touch()
        except Exception:
            pass

# 运行初始化
_initialize()

# ============= 简化的导入别名 =============
# 为常用类创建简短别名
DP = DataPreprocessor
FE = FeatureEngineer
MTN = MultiTaskTADFNet
XGB = XGBoostTADFPredictor
MT = ModelTrainer
MI = ModelInterpreter

# ============= 模块级属性 =============
def __getattr__(name):
    """动态属性访问（Python 3.7+）"""
    deprecated = {
        'DataProcessor': 'DataPreprocessor',  # 处理可能的拼写错误
        'XGBoostPredictor': 'XGBoostTADFPredictor'
    }
    
    if name in deprecated:
        warnings.warn(
            f"'{name}' is deprecated, use '{deprecated[name]}' instead",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[deprecated[name]]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# ============= 包信息 =============
def info():
    """打印包的详细信息"""
    import torch
    import sklearn
    import xgboost
    
    info_text = f"""
{'='*50}
TADF/rTADF Prediction System
{'='*50}
Version: {__version__}
Author: {__author__}
License: {__license__}

Environment:
- Python: {sys.version.split()[0]}
- PyTorch: {torch.__version__}
- Scikit-learn: {sklearn.__version__}
- XGBoost: {xgboost.__version__}
- CUDA Available: {torch.cuda.is_available()}

Modules:
- Data Processing: ✓
- Feature Engineering: ✓  
- Models (DL + XGBoost): ✓
- Training & Evaluation: ✓
- Interpretation: ✓
- Prediction: ✓

Quick Start:
>>> from src import quick_setup, run_pipeline
>>> env = quick_setup()
>>> results = run_pipeline('data.csv', mode='train')
{'='*50}
"""
    print(info_text)

# 当模块被直接运行时
if __name__ == '__main__':
    info()
    check_dependencies()