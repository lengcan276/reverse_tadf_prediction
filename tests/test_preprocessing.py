import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

def test_data_preprocessing():
    """测试数据预处理流程"""
    
    print("="*50)
    print("Testing Data Preprocessing Module")
    print("="*50)
    
    # 1. 加载小批量数据进行测试
    print("\n1. Loading sample data...")
    df_sample = pd.read_csv('data/all_conformers_data.csv', nrows=100)
    print(f"   Sample shape: {df_sample.shape}")
    print(f"   Columns: {df_sample.columns.tolist()[:10]}...")
    
    # 2. 测试数据清洗
    print("\n2. Testing data cleaning...")
    preprocessor = DataPreprocessor('data/all_conformers_data.csv')
    original_shape = preprocessor.df.shape
    preprocessor.clean_data()
    print(f"   Original shape: {original_shape}")
    print(f"   After cleaning: {preprocessor.df.shape}")
    
    # 3. 测试缺失值处理
    print("\n3. Testing missing value handling...")
    missing_before = preprocessor.df.isnull().sum().sum()
    preprocessor.handle_missing_values()
    missing_after = preprocessor.df.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    print(f"   Missing values after: {missing_after}")
    
    # 4. 测试标签创建
    print("\n4. Testing label creation...")
    preprocessor.create_labels()
    print(f"   TADF molecules: {preprocessor.df['is_TADF'].sum()}")
    print(f"   rTADF molecules: {preprocessor.df['is_rTADF'].sum()}")
    
    # 5. 测试数据分割
    print("\n5. Testing data splitting...")
    (X_train, X_val, X_test), (y_tadf_train, _, _), (y_rtadf_train, _, _) = \
        preprocessor.split_data()
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")
    
    print("\n✓ Preprocessing tests completed successfully!")
    
    return preprocessor

if __name__ == "__main__":
    preprocessor = test_data_preprocessing()