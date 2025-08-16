import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models import MultiTaskTADFNet, ModelTrainer
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

def test_model_training():
    """测试模型训练流程"""
    
    print("="*50)
    print("Testing Model Training")
    print("="*50)
    
    # 1. 准备小批量数据
    print("\n1. Preparing small dataset for testing...")
    preprocessor = DataPreprocessor('data/all_conformers_data.csv')
    preprocessor.clean_data().handle_missing_values().create_labels()
    
    fe = FeatureEngineer(preprocessor.df)
    key_features = fe.select_key_features()
    preprocessor.normalize_features(key_features)
    
    (X_train, X_val, X_test), (y_tadf_train, y_tadf_val, y_tadf_test), \
    (y_rtadf_train, y_rtadf_val, y_rtadf_test) = preprocessor.split_data()
    
    # 使用小批量数据
    X_train_small = X_train[:100]
    y_tadf_train_small = y_tadf_train[:100]
    y_rtadf_train_small = y_rtadf_train[:100]
    
    print(f"   Training samples: {X_train_small.shape}")
    
    # 2. 测试模型初始化
    print("\n2. Testing model initialization...")
    model = MultiTaskTADFNet(input_dim=X_train_small.shape[1])
    print(f"   Model created with input dim: {X_train_small.shape[1]}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. 测试前向传播
    print("\n3. Testing forward pass...")
    X_test_tensor = torch.FloatTensor(X_train_small[:10])
    with torch.no_grad():
        tadf_pred, rtadf_pred = model(X_test_tensor)
    print(f"   TADF predictions shape: {tadf_pred.shape}")
    print(f"   rTADF predictions shape: {rtadf_pred.shape}")
    
    # 4. 测试训练循环
    print("\n4. Testing training loop (3 epochs)...")
    trainer = ModelTrainer(model)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_small),
        torch.FloatTensor(y_tadf_train_small),
        torch.FloatTensor(y_rtadf_train_small)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    for epoch in range(3):
        loss = trainer.train_epoch(train_loader)
        print(f"   Epoch {epoch+1}: Loss = {loss:.4f}")
    
    # 5. 测试评估
    print("\n5. Testing evaluation...")
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val[:50]),
        torch.FloatTensor(y_tadf_val[:50]),
        torch.FloatTensor(y_rtadf_val[:50])
    )
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    metrics = trainer.evaluate(val_loader)
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n✓ Model training tests completed successfully!")
    
    return model, trainer

if __name__ == "__main__":
    model, trainer = test_model_training()