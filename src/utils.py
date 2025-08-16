# src/utils.py
import yaml
import json
import logging
import os
from datetime import datetime
import pickle
import torch
import pandas as pd
import numpy as np

def load_config(config_path='configs/config.yaml'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置默认值
    config.setdefault('random_seed', 42)
    config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    return config

def setup_logging(log_dir='outputs/logs', log_name=None):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    if log_name is None:
        log_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_file = os.path.join(log_dir, log_name)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def save_results(results, output_path, format='json'):
    """保存结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'json':
        # 转换numpy数组为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
    
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == 'csv' and isinstance(results, pd.DataFrame):
        results.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_model(model_path, model_class=None, device='cpu'):
    """加载保存的模型"""
    if model_path.endswith('.pth'):
        # PyTorch模型
        if model_class is None:
            raise ValueError("model_class required for PyTorch models")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = model_class(**checkpoint.get('model_config', {}))
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint.get('training_history', {})
        else:
            # 只有state_dict
            model = model_class()
            model.load_state_dict(checkpoint)
            return model, {}
    
    elif model_path.endswith('.pkl'):
        # Scikit-learn或XGBoost模型
        with open(model_path, 'rb') as f:
            return pickle.load(f), {}
    
    else:
        raise ValueError(f"Unknown model format: {model_path}")

def save_model(model, save_path, optimizer=None, epoch=None, metrics=None):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if hasattr(model, 'state_dict'):
        # PyTorch模型
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if hasattr(model, '__class__'):
            checkpoint['model_class'] = model.__class__.__name__
        
        torch.save(checkpoint, save_path)
    
    else:
        # Scikit-learn或XGBoost模型
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

def set_random_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def create_output_dirs(base_dir='outputs'):
    """创建输出目录结构"""
    dirs = [
        f'{base_dir}/models',
        f'{base_dir}/figures',
        f'{base_dir}/logs',
        f'{base_dir}/results'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs