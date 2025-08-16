import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
import pandas as pd

def test_feature_engineering():
    """测试特征工程"""
    
    print("="*50)
    print("Testing Feature Engineering Module")
    print("="*50)
    
    # 准备数据
    print("\n1. Preparing data...")
    preprocessor = DataPreprocessor('data/all_conformers_data.csv')
    preprocessor.clean_data().handle_missing_values()
    
    # 测试特征选择
    print("\n2. Testing feature selection...")
    fe = FeatureEngineer(preprocessor.df)
    key_features = fe.select_key_features()
    print(f"   Selected {len(key_features)} key features")
    print(f"   Sample features: {key_features[:5]}")
    
    # 测试特征创建
    print("\n3. Testing feature creation...")
    original_cols = len(preprocessor.df.columns)
    preprocessor.df = fe.create_interaction_features()
    new_cols = len(preprocessor.df.columns)
    print(f"   Original features: {original_cols}")
    print(f"   After engineering: {new_cols}")
    print(f"   New features created: {new_cols - original_cols}")
    
    # 检查新特征的分布
    print("\n4. Checking new feature distributions...")
    new_features = ['donor_acceptor_balance', 'conjugation_rigidity_ratio', 
                   'st_gap_ratio', 'molecular_complexity']
    for feat in new_features:
        if feat in preprocessor.df.columns:
            print(f"   {feat}: mean={preprocessor.df[feat].mean():.3f}, "
                  f"std={preprocessor.df[feat].std():.3f}")
    
    print("\n✓ Feature engineering tests completed successfully!")
    
    return fe, preprocessor

if __name__ == "__main__":
    fe, preprocessor = test_feature_engineering()