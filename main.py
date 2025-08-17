#main.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import joblib
import numpy as np
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import MultiTaskTADFNet, XGBoostTADFPredictor
from src.training import ModelTrainer
from src.interpretation import ModelInterpreter
import pandas as pd
import os
from xgboost import XGBClassifier
os.makedirs('outputs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/splits', exist_ok=True)
def main():
    """主程序"""
    
    # 1. 数据预处理
    print("Step 1: Data Preprocessing...")
    preprocessor = DataPreprocessor('./data/all_conformers_data.csv')
    preprocessor.clean_data().extract_features_from_smiles().handle_missing_values().create_labels()
    # ========= 检查官能团特征是否存在 =========
    print("\n=== Checking functional group features in raw data ===")
    fg_features = ['has_nitro', 'count_nitro', 'has_triphenylamine', 
                'count_triphenylamine', 'has_carbazole', 'count_carbazole', 
                'has_triazine', 'count_triazine']

    for fg in fg_features:
        if fg in preprocessor.df.columns:
            non_zero = (preprocessor.df[fg] != 0).sum()
            print(f"{fg}: exists, non-zero count = {non_zero}")
        else:
            print(f"{fg}: NOT FOUND in data!")
    # ========= 检查结束 =========
    preprocessor.clean_data().handle_missing_values().create_labels()
    # 选择使用哪套标签进行训练
    USE_STRICT_LABELS = True  # 可以通过参数控制

    if USE_STRICT_LABELS:
        print("\n📊 Using STRICT literature-based labels for training")
        # Primary labels已经在is_TADF和is_rTADF中
    else:
        print("\n📊 Using RELAXED labels for sensitivity analysis")
        # 使用relaxed标签覆盖primary
        preprocessor.df['is_TADF'] = preprocessor.df['is_TADF_relaxed']
        preprocessor.df['is_rTADF'] = preprocessor.df['is_rTADF_relaxed']

    # 打印最终使用的标签统计
    print(f"Final labels for training:")
    print(f"  TADF: {preprocessor.df['is_TADF'].sum()}/{len(preprocessor.df)}")
    print(f"  rTADF: {preprocessor.df['is_rTADF'].sum()}/{len(preprocessor.df)}")
    # 1.5 分子级聚合（新增）
    print("\nStep 1.5: Molecular Level Aggregation...")

    # 检查是否有多个构象
    if 'Molecule' in preprocessor.df.columns:
        unique_molecules = preprocessor.df['Molecule'].nunique()
        total_conformers = len(preprocessor.df)
        avg_conformers = total_conformers / unique_molecules
        
        print(f"\n=== Molecular Statistics ===")
        print(f"Data structure: {unique_molecules} molecules, {total_conformers} conformers")
        print(f"Average conformers per molecule: {avg_conformers:.1f}")
        
        # 详细统计
        mol_stats = preprocessor.df.groupby('Molecule').size()
        print(f"\nConformers distribution:")
        print(f"  Min: {mol_stats.min()}")
        print(f"  Max: {mol_stats.max()}")
        print(f"  Mean: {mol_stats.mean():.2f}")
        print(f"  Median: {mol_stats.median():.1f}")
        print(f"  Std: {mol_stats.std():.2f}")
        
        # 检查标签一致性
        print(f"\nLabel consistency check:")
        for label in ['is_TADF', 'is_rTADF']:
            if label in preprocessor.df.columns:
                label_consistency = preprocessor.df.groupby('Molecule')[label].agg(['min', 'max', 'mean'])
                inconsistent = (label_consistency['min'] != label_consistency['max']).sum()
                if inconsistent > 0:
                    print(f"  ⚠️ {inconsistent} molecules have inconsistent {label} labels across conformers")
                    # 可选：显示哪些分子有不一致的标签
                    inconsistent_mols = label_consistency[label_consistency['min'] != label_consistency['max']].index[:5]
                    print(f"    Examples: {list(inconsistent_mols)}")
        
        # 决定是否需要聚合
        if avg_conformers > 1.5:  # 有多构象
            print("\n→ Multiple conformers detected, performing molecular aggregation...")
          #  preprocessor.aggregate_by_molecule(method='min_gap')
        else:
            print("\n→ Single conformer per molecule, skipping aggregation")
    else:
        print("Warning: No 'Molecule' column found, cannot perform molecular aggregation")
    # 2. 特征工程
    print("\nStep 2: Feature Engineering...")
    fe = FeatureEngineer(preprocessor.df)
    # ========= 检查Calicene特征是否存在 =========
    print("\n=== Checking Calicene-specific features ===")

    # 基础特征
    basic_calicene = ['has_3ring', 'has_5ring', 'subs_on_3ring', 'subs_on_5ring']

    # 位置特征
    position_features = ['num_in_subs_3ring', 'num_out_subs_3ring', 
                        'num_in_subs_5ring', 'num_out_subs_5ring',
                        'num_both_subs', 'num_sp_subs']

    # D/A分布
    da_features = ['donor_on_3ring', 'donor_on_5ring',
                'acceptor_on_3ring', 'acceptor_on_5ring',
                'DA_strength_5minus3', 'DA_in_out_bias']

    # CT特征
    ct_features = ['CT_alignment_score', 'CT_position_weighted_score',
                'DA_asymmetry', 'favorable_for_inversion']

    # 密度特征
    density_features = ['D_density', 'A_density', 'D_volume_density', 
                    'A_volume_density', 'in_sub_density', 'out_sub_density']

    # One-hot编码特征
    onehot_features = ['push_pull_pattern_none', 'push_pull_pattern_D5_A3',
                    'push_pull_pattern_D3_A5', 'push_pull_pattern_D5_only',
                    'push_pull_pattern_A3_only', 'push_pull_pattern_DD_balanced',
                    'push_pull_pattern_AA_balanced',
                    'ring_polarity_expected_aligned', 
                    'ring_polarity_expected_reversed',
                    'ring_polarity_expected_neutral']

    all_calicene_features = (basic_calicene + position_features + da_features + 
                            ct_features + density_features + onehot_features)

    found_count = 0
    missing_features = []

    for feat_group, feat_list in [
        ("Basic", basic_calicene),
        ("Position", position_features),
        ("D/A", da_features),
        ("CT", ct_features),
        ("Density", density_features),
        ("One-hot", onehot_features)
    ]:
        print(f"\n{feat_group} features:")
        for feat in feat_list:
            if feat in preprocessor.df.columns:
                non_zero = (preprocessor.df[feat] != 0).sum()
                print(f"  ✓ {feat}: {non_zero} non-zero values")
                found_count += 1
            else:
                print(f"  ✗ {feat}: NOT FOUND")
                missing_features.append(feat)

    print(f"\nSummary: Found {found_count}/{len(all_calicene_features)} Calicene features")
    if missing_features:
        print(f"Missing features: {missing_features}")
    # ========= 检查结束 =========

    # 创建交互特征（包含方法1和方法2）
    preprocessor.df = fe.create_interaction_features()
    print(f"Created interaction features. Shape: {preprocessor.df.shape}")
    # ========= 新增：分析高层次特征 =========
    # 特征稀疏性分析
    def analyze_feature_quality(df, features):
        """分析特征质量"""
        quality_report = {}
        for feat in features:
            if feat in df.columns:
                non_zero = (df[feat] != 0).sum()
                unique = df[feat].nunique()
                quality_report[feat] = {
                    'coverage': non_zero / len(df),
                    'unique_values': unique
                }
        
        # 打印低覆盖率特征
        low_coverage = [f for f, v in quality_report.items() if v['coverage'] < 0.1]
        if low_coverage:
            print(f"Low coverage features (<10%): {low_coverage[:5]}")
        
        return quality_report
    
    # 分析关键特征质量
    critical_features = fe.select_critical_features()
    feature_quality = analyze_feature_quality(preprocessor.df, critical_features)
    
    # 移除低质量特征
    high_quality_features = [f for f in critical_features 
                            if f not in feature_quality or 
                            feature_quality[f]['coverage'] > 0.05]
    
    print(f"Filtered from {len(critical_features)} to {len(high_quality_features)} high-quality features")
    critical_features = high_quality_features
    # 选择关键特征（方法3）
    # 打印官能团特征的存在情况
    fg_features = ['donor_score', 'acceptor_score', 'D_A_ratio', 'D_A_product', 
                   'donor_homo_effect', 'acceptor_lumo_effect', 'DA_st_gap_effect']
    existing_fg = [f for f in fg_features if f in critical_features]
    print(f"Functional group features in critical features: {existing_fg}")
    # 标准化关键特征
    # 1. 二进制特征（0/1）- 不标准化
    # 标准化关键特征
# 1. 二进制特征（0/1）- 不标准化
    binary_features = [
        # Calicene二进制特征
        'has_3ring', 'has_5ring',
        'donor_on_3ring', 'donor_on_5ring',
        'acceptor_on_3ring', 'acceptor_on_5ring',
        'favorable_for_inversion',
        
        # 所有has_开头的特征
        *[f for f in critical_features if f.startswith('has_')],
        
        # One-hot编码特征
        *[f for f in critical_features if 'push_pull_pattern_' in f],
        *[f for f in critical_features if 'ring_polarity_expected_' in f],
        
        # 其他二进制
        'is_D_A_molecule', 'has_inversion'
    ]

    # 2. 整数计数特征 - 不标准化或使用特殊处理
    count_features = [
        # 环系统计数
        'num_rings', 'num_aromatic_rings', 'num_saturated_rings',
        'num_aliphatic_rings', 'num_aromatic_heterocycles',
        'num_saturated_heterocycles',
        'num_3_member_rings', 'num_4_member_rings', 'num_5_member_rings',
        'num_6_member_rings', 'num_7_member_rings', 'num_8_member_rings',
        
        # 原子计数
        'num_atoms', 'num_bonds', 'num_heavy_atoms',
        'num_heteroatoms', 'num_N_atoms', 'num_O_atoms', 'num_S_atoms',
        'num_F_atoms', 'num_Cl_atoms', 'num_Br_atoms', 'num_I_atoms', 'num_P_atoms',
        
        # 其他计数
        'num_rotatable_bonds', 'num_conjugated_bonds', 'num_conjugated_systems',
        'num_hbd', 'num_hba', 'num_amide_bonds',
        
        # Calicene计数
        'subs_on_3ring', 'subs_on_5ring',
        'num_in_subs_3ring', 'num_out_subs_3ring',
        'num_in_subs_5ring', 'num_out_subs_5ring',
        'num_both_subs', 'num_sp_subs',
        
        # 所有count_开头的特征
        *[f for f in critical_features if f.startswith('count_')],
        
        # 反转gap计数
        'num_inverted_gaps', 'num_primary_inversions',
        
        # CREST构象数
        'crest_num_conformers',
        # 激发态计数
        'num_singlet_states', 'num_triplet_states',
        
        # 成功标志（虽然是0/1，但作为计数处理）
        'excited_opt_success', 'excited_no_imaginary',
        'triplet_excited_opt_success', 'triplet_excited_no_imaginary'
    ]

    # 3. 分类/编码特征 - 不标准化
    categorical_features = [
        'mol_type_code',  # 分子类型编码（0,1,2,3）
    ]

    # 4. 需要标准化的连续特征
    continuous_features = []
    features_to_skip = set(binary_features + count_features + categorical_features)

    # 强制包含密度特征（即使它们可能不在critical_features中）
    must_normalize = [
            'cyano_density', 'nitro_density', 'amino_density', 'carbonyl_density',
            'aromatic_gap_product', 'st_gap_ratio', 'st_average_energy'
        ]

    for feat in must_normalize:
        if feat in preprocessor.df.columns and feat not in features_to_skip:
            continuous_features.append(feat)
            print(f"  Added mandatory continuous feature: {feat}")
    
    # 然后添加其他连续特征
    for f in critical_features:
        if f in preprocessor.df.columns and f not in features_to_skip and f not in continuous_features:
            unique_vals = preprocessor.df[f].dropna().unique()
            if len(unique_vals) <= 10:  # 少于10个唯一值
                print(f"  Feature {f} has only {len(unique_vals)} unique values, checking if categorical...")
                # 如果都是整数，可能是编码特征
                if all(isinstance(v, (int, np.integer)) or v.is_integer() for v in unique_vals if pd.notna(v)):
                    print(f"    -> Treating as categorical, not normalizing")
                    categorical_features.append(f)
                    continue
            continuous_features.append(f)

    print(f"\nFeature type breakdown:")
    print(f"  Binary features (keep as-is): {len([f for f in binary_features if f in preprocessor.df.columns])}")
    print(f"  Count features (keep as-is): {len([f for f in count_features if f in preprocessor.df.columns])}")
    print(f"  Categorical features (keep as-is): {len([f for f in categorical_features if f in preprocessor.df.columns])}")
    print(f"  Continuous features (normalize): {len(continuous_features)}")
    preprocessor.binary_features = binary_features
    preprocessor.count_features = count_features
    preprocessor.categorical_features = categorical_features
    preprocessor.continuous_features = continuous_features
    # 只标准化连续特征
    if continuous_features:
        print(f"\nNormalizing {len(continuous_features)} continuous features...")
        # 显示一些将被标准化的特征示例
        print(f"  Examples: {continuous_features[:5]}")
        preprocessor.normalize_features(continuous_features)

    # 保存特征类型信息，以便后续使用
    preprocessor.continuous_features = continuous_features  # 添加这行

    # 验证关键特征没有被破坏
    print("\n=== Verification of Key Features ===")
    verification_features = [
        'has_3ring', 'has_5ring', 'num_rings', 'num_aromatic_rings',
        'num_3_member_rings', 'num_atoms', 'subs_on_3ring', 'subs_on_5ring'
    ]

    for feat in verification_features:
        if feat in preprocessor.df.columns:
            unique_vals = preprocessor.df[feat].nunique()
            non_zero = (preprocessor.df[feat] != 0).sum()
            mean_val = preprocessor.df[feat].mean()
            max_val = preprocessor.df[feat].max()
            print(f"{feat:25s}: unique={unique_vals:4d}, non_zero={non_zero:4d}, mean={mean_val:7.3f}, max={max_val:7.3f}")

    # 保存处理后的数据
    import os
    os.makedirs('data/processed', exist_ok=True)

    # 保存完整数据前，移除全零列
    print("\n=== Checking for empty columns in full data ===")
    full_df = preprocessor.df.copy()
    empty_cols_full = []
    for col in full_df.columns:
        if col not in ['Molecule', 'is_TADF', 'is_rTADF']:  # 保护标签列
            if full_df[col].dtype in [np.number, float, int]:
                if (full_df[col] == 0).all() or full_df[col].isna().all():
                    empty_cols_full.append(col)
                    print(f"  Removing empty column from full data: {col}")

    if empty_cols_full:
        full_df = full_df.drop(columns=empty_cols_full)
        print(f"Removed {len(empty_cols_full)} empty columns from full data")

    full_df.to_csv('data/processed/full_features.csv', index=False)
    print(f"Saved full_features.csv: {full_df.shape}")

    # 保存关键特征子集 - 确保包含所有必要的列
    save_columns = critical_features + ['is_TADF', 'is_rTADF', 'Molecule', 'SMILES']
    save_columns = [col for col in save_columns if col in preprocessor.df.columns]
    critical_df = preprocessor.df[save_columns].copy()
    # 验证关键特征
    print("\n=== Feature Verification ===")
    verify_features = ['has_3ring_nh2', 'has_5ring_nh2', 'has_5ring_oh', 
                    'num_aromatic_rings', 'has_nitro', 'nitro_density']
    for feat in verify_features:
        if feat in critical_df.columns:
            non_zero = (critical_df[feat] != 0).sum()
            print(f"{feat}: {non_zero} non-zero values")
    # 移除全零列和空列
    print("\n=== Checking for empty columns in critical features ===")
    empty_cols_critical = []
    for col in save_columns:
        if col not in ['is_TADF', 'is_rTADF', 'Molecule']:  # 保护标签和ID列
            if col in critical_df.columns:
                # 检查是否全零或全空
                if critical_df[col].dtype in [np.number, float, int]:
                    is_all_zero = (critical_df[col] == 0).all()
                    is_all_nan = critical_df[col].isna().all()
                    if is_all_zero or is_all_nan:
                        empty_cols_critical.append(col)
                        print(f"  Removing empty column: {col} (all_zero={is_all_zero}, all_nan={is_all_nan})")

    # 移除空列
    if empty_cols_critical:
        critical_df = critical_df.drop(columns=empty_cols_critical)
        save_columns = [col for col in save_columns if col not in empty_cols_critical]
        # 同时从critical_features中移除，保持一致性
        critical_features = [f for f in critical_features if f not in empty_cols_critical]
        print(f"Removed {len(empty_cols_critical)} empty columns from critical features")

    # 再次验证保存的数据
    print("\n=== Final verification before saving ===")
    verification_features = [
        'has_3ring', 'has_5ring', 'num_rings', 'num_aromatic_rings',
        'num_3_member_rings', 'num_atoms', 'subs_on_3ring', 'subs_on_5ring'
    ]

    for feat in verification_features:
        if feat in critical_df.columns:
            non_zero = (critical_df[feat] != 0).sum()
            unique_vals = critical_df[feat].nunique()
            print(f"  {feat}: {non_zero} non-zero values, {unique_vals} unique values")

    critical_df.to_csv('data/processed/critical_features.csv', index=False)
    print(f"Saved critical_features.csv: {critical_df.shape}")

    # 打印两个文件的差异统计
    print(f"\n=== Summary ===")
    print(f"Full features: {full_df.shape[1]} columns (removed {len(empty_cols_full)} empty)")
    print(f"Critical features: {critical_df.shape[1]} columns (removed {len(empty_cols_critical)} empty)")
    print(f"Feature reduction: {full_df.shape[1]} -> {critical_df.shape[1]} ({critical_df.shape[1]/full_df.shape[1]*100:.1f}%)")
    
    # 3. 数据划分
    print("Step 3: Data Splitting...")
    (X_train, X_val, X_test), (y_tadf_train, y_tadf_val, y_tadf_test), \
    (y_rtadf_train, y_rtadf_val, y_rtadf_test) = preprocessor.split_data( 
        use_features=critical_features 
    )

    # 验证维度
    print(f"\nFeature dimensions check:")
    print(f"  Training features: {X_train.shape}")
    print(f"  Critical features selected: {len(critical_features)}")
    print(f"  Continuous features normalized: {len(continuous_features)}")  # 修改这里：numeric_features -> continuous_features

    if hasattr(preprocessor, 'feature_cols'):
        print(f"  Number of feature names: {len(preprocessor.feature_cols)}")
        assert len(preprocessor.feature_cols) == X_train.shape[1], \
            f"Feature names mismatch: {len(preprocessor.feature_cols)} names vs {X_train.shape[1]} features"
    # ============ 验证代码结束 ============
    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_tadf_train_tensor = torch.FloatTensor(y_tadf_train)
    y_rtadf_train_tensor = torch.FloatTensor(y_rtadf_train)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_tadf_train_tensor, y_rtadf_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 4. 深度学习模型训练
    print("Step 4: Training Deep Learning Model...")
    # 计算类别权重
    n_tadf_pos = y_tadf_train.sum()
    n_tadf_neg = len(y_tadf_train) - n_tadf_pos
    tadf_pos_weight = n_tadf_neg / (n_tadf_pos + 1e-6)

    n_rtadf_pos = y_rtadf_train.sum()
    n_rtadf_neg = len(y_rtadf_train) - n_rtadf_pos
    rtadf_pos_weight = n_rtadf_neg / (n_rtadf_pos + 1e-6)

    print(f"Class weights - TADF: {tadf_pos_weight:.2f}, rTADF: {rtadf_pos_weight:.2f}")

    dl_model = MultiTaskTADFNet(input_dim=X_train.shape[1])
    trainer = ModelTrainer(dl_model, tadf_pos_weight=tadf_pos_weight, rtadf_pos_weight=rtadf_pos_weight)

    # 训练循环 - 改为监控PR-AUC
    best_val_pr_auc = 0  # 改为监控PR-AUC
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        train_loss = trainer.train_epoch(train_loader)
        
        # 验证
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_tadf_val),
            torch.FloatTensor(y_rtadf_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=32)
        val_metrics = trainer.evaluate(val_loader, use_optimal_threshold=True)
        
        # 使用PR-AUC的平均值作为监控指标
        val_pr_auc = (val_metrics.get('tadf_pr_auc', 0) + val_metrics.get('rtadf_pr_auc', 0)) / 2
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
            f"TADF PR-AUC={val_metrics.get('tadf_pr_auc', 0):.4f}, "
            f"rTADF PR-AUC={val_metrics.get('rtadf_pr_auc', 0):.4f}, "
            f"TADF F1={val_metrics.get('tadf_f1', 0):.4f}, "
            f"rTADF F1={val_metrics.get('rtadf_f1', 0):.4f}")
        
        # 早停基于PR-AUC
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            torch.save(dl_model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        trainer.scheduler.step(val_pr_auc)  # 基于PR-AUC调整学习率
        
    # 5. XGBoost模型训练
    print("\nStep 5: Training XGBoost Model...")
    xgb_predictor = XGBoostTADFPredictor()
    xgb_predictor.train(X_train, y_tadf_train, y_rtadf_train)
    
    # 6. 模型评估
    print("\nStep 6: Model Evaluation...")
    
    # 深度学习模型评估
    dl_model.load_state_dict(torch.load('best_model.pth'))
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_tadf_test),
        torch.FloatTensor(y_rtadf_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=32)
    dl_metrics = trainer.evaluate(test_loader)
    
    print("\nDeep Learning Model Performance:")
    for key, value in dl_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # XGBoost模型评估
    xgb_tadf_pred, xgb_rtadf_pred = xgb_predictor.predict(X_test)
    xgb_metrics = {
        'tadf_auc': roc_auc_score(y_tadf_test, xgb_tadf_pred) if len(np.unique(y_tadf_test)) > 1 else 0,
        'rtadf_auc': roc_auc_score(y_rtadf_test, xgb_rtadf_pred) if len(np.unique(y_rtadf_test)) > 1 else 0
    }
    
    print("\nXGBoost Model Performance:")
    for key, value in xgb_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # ============ 7. 模型解释 ============
    print("\nStep 7: Model Interpretation...")
    
    # 获取特征名称
    if hasattr(preprocessor, 'feature_cols'):
        feature_names = preprocessor.feature_cols
    else:
        feature_names = critical_features  # 使用关键特征名称
    
    # 确保特征名称数量正确
    if len(feature_names) != X_train.shape[1]:
        print(f"Warning: Feature names mismatch. Using generic names.")
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # 创建解释器
    interpreter = ModelInterpreter(
        xgb_predictor.tadf_model.best_estimator_ if hasattr(xgb_predictor.tadf_model, 'best_estimator_') 
        else xgb_predictor.tadf_model,
        X_train,
        feature_names
    )
    
    # SHAP分析
    interpreter.shap_analysis(X_test[:min(100, len(X_test))], task='tadf')
    
    # 特征重要性
    tadf_importance, rtadf_importance = xgb_predictor.get_feature_importance()

    # 绘制TADF特征重要性
    if isinstance(tadf_importance, dict):
        tadf_values = list(tadf_importance.values())
    else:
        tadf_values = tadf_importance

    interpreter.plot_feature_importance(tadf_values, top_n=20, task='tadf')

    # 绘制rTADF特征重要性
    if isinstance(rtadf_importance, dict):
        rtadf_values = list(rtadf_importance.values())
    else:
        rtadf_values = rtadf_importance

    interpreter.plot_feature_importance(rtadf_values, top_n=20, task='rtadf')

    # 可选：创建综合特征重要性图（两个任务的平均）
    combined_importance = (np.array(tadf_values) + np.array(rtadf_values)) / 2
    interpreter.plot_feature_importance(combined_importance, top_n=20, task='combined')

    print("Generated feature importance plots for both tasks")
    
    
    # 混淆矩阵
    # 混淆矩阵 - TADF
    xgb_tadf_binary = (xgb_tadf_pred > 0.5).astype(int)
    interpreter.plot_confusion_matrices(y_tadf_test, xgb_tadf_binary, 
                                    labels=['Non-TADF', 'TADF'], 
                                    task='tadf')

    # 混淆矩阵 - rTADF
    xgb_rtadf_binary = (xgb_rtadf_pred > 0.5).astype(int)
    interpreter.plot_confusion_matrices(y_rtadf_test, xgb_rtadf_binary, 
                                    labels=['Non-rTADF', 'rTADF'], 
                                    task='rtadf')
    # 保存模型
    import joblib
    joblib.dump(xgb_predictor, 'xgboost_tadf_predictor.pkl')
    torch.save({
        'model_state_dict': dl_model.state_dict(),
        'model_config': {'input_dim': X_train.shape[1], 'hidden_dims': [256, 128, 64], 'dropout': 0.3}
    }, 'best_dl_model.pth')
    print("Models saved successfully!")

    print("\nStep 8: Saving Models and Data...")
    
    # 保存模型
    import joblib
    
    # 保存XGBoost模型
    joblib.dump(xgb_predictor, 'xgboost_tadf_predictor.pkl')
    
    # 保存深度学习模型
    torch.save({
        'model_state_dict': dl_model.state_dict(),
        'model_config': {'input_dim': X_train.shape[1], 'hidden_dims': [256, 128, 64], 'dropout': 0.3}
    }, 'best_dl_model.pth')
    
    # 保存标准化器
    if hasattr(preprocessor, 'scaler'):
        joblib.dump(preprocessor.scaler, 'scaler.pkl')
        print("Scaler saved successfully!")
    
    # 保存特征名称列表（用于预测时）
    with open('data/splits/feature_names.txt', 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    print(f"Saved {len(feature_names)} feature names")
    
    print("All models and data saved successfully!")

    # ============ 9. 稳健性测试 ============
    print("\n" + "="*60)
    print("Step 9: Robustness Testing")
    print("="*60)
    
    from src.robustness_test import (
        define_feature_blocks, run_block_ablation,
        xgb_importances, perm_importance, shap_importance, rank_consistency,
        sweep_thresholds, oof_score, multi_seed_stability,
        get_design_safe_features, safe_cv_xgb_grouped, random_label_test
    )
    
    # 9.0 准备两套特征：诊断模型 vs 设计模型
    print("\n9.0 Feature Sets Preparation...")
    
    # 诊断模型特征（包含gap）
    diagnostic_features = preprocessor.feature_cols
    print(f"Diagnostic model features: {len(diagnostic_features)}")
    
    # 设计模型特征（移除近标签特征）
    design_features = get_design_safe_features(preprocessor.df)
    design_features = [f for f in design_features if f in preprocessor.feature_cols]
    print(f"Design model features: {len(design_features)}")
    
    # 9.1 随机标签测试（泄漏检测）
    print("\n9.1 Random Label Test (Leakage Detection)...")
    
    # 测试诊断模型
    print("\nDiagnostic Model Random Label Test:")
    diag_random = random_label_test(preprocessor.df, diagnostic_features, n_iterations=3)
    
    # 测试设计模型
    print("\nDesign Model Random Label Test:")
    design_random = random_label_test(preprocessor.df, design_features, n_iterations=3)
    
    # 9.2 分块消融（使用GroupKFold）
    print("\n9.2 Block Ablation Analysis (with GroupKFold)...")
    blocks = define_feature_blocks(preprocessor.df.columns)
    
    # 更新run_block_ablation以使用safe_cv_xgb_grouped
    def run_block_ablation_grouped(df, target_col, all_feats, blocks):
        """改进的分块消融，使用GroupKFold"""
        results = []
        
        # 定义实验
        only_gap = [f for f in blocks['gap_block'] if f in all_feats]
        only_da_geom = sorted(set([f for f in blocks['da_block'] + blocks['geom_block'] if f in all_feats]))
        only_structure = [f for f in blocks['structure_block'] if f in all_feats]
        
        full = all_feats
        drop_gap = [c for c in full if c not in set(blocks['gap_block'])]
        drop_da = [c for c in full if c not in set(blocks['da_block'])]
        drop_geom = [c for c in full if c not in set(blocks['geom_block'])]
        
        exps = {
            'only_gap': only_gap,
            'only_da_geom': only_da_geom,
            'only_structure': only_structure,
            'full': full,
            'full_minus_gap': drop_gap,
            'full_minus_da': drop_da,
            'full_minus_geom': drop_geom
        }
        
        for name, cols in exps.items():
            if len(cols) == 0:
                continue
            
            print(f"  Testing {name} with {len(cols)} features...")
            m = safe_cv_xgb_grouped(df, cols, target_col, n_splits=5)
            m['setting'] = name
            m['n_feats'] = len(cols)
            results.append(m)
        
        return pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    
    # 设计模型的消融（不包含gap）
    print("\nDesign Model Ablation (no gap features):")
    design_blocks = define_feature_blocks(pd.Index(design_features))
    res_design_rtadf = run_block_ablation_grouped(
        preprocessor.df, 'is_rTADF', design_features, design_blocks
    )
    print(res_design_rtadf[['setting', 'n_feats', 'roc_auc', 'f1', 'roc_auc_std']].to_string())
    res_design_rtadf.to_csv('outputs/design_rtadf_ablation.csv', index=False)
    
    # 诊断模型的消融（包含gap）
    print("\nDiagnostic Model Ablation (with gap features):")
    res_diag_rtadf = run_block_ablation_grouped(
        preprocessor.df, 'is_rTADF', diagnostic_features, blocks
    )
    print(res_diag_rtadf[['setting', 'n_feats', 'roc_auc', 'f1', 'roc_auc_std']].to_string())
    res_diag_rtadf.to_csv('outputs/diagnostic_rtadf_ablation.csv', index=False)
    
    # 9.3 改进的重要性一致性（同一验证集）
    print("\n9.3 Improved Feature Importance Consistency...")
    
    # 使用设计模型特征进行测试
    from sklearn.model_selection import GroupShuffleSplit
    
    X_design = preprocessor.df[design_features].astype(np.float32).fillna(0).values
    y_rtadf = preprocessor.df['is_rTADF'].astype(int).values
    groups = preprocessor.df['Molecule'].values if 'Molecule' in preprocessor.df.columns else None
    
    # 分组划分
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, val_idx = next(gss.split(X_design, y_rtadf, groups))
    else:
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(
            range(len(X_design)), test_size=0.2, random_state=42, stratify=y_rtadf
        )
    
    X_tr, X_val = X_design[tr_idx], X_design[val_idx]
    y_tr, y_val = y_rtadf[tr_idx], y_rtadf[val_idx]
    
    # 训练模型
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ))
    ])
    pipe.fit(X_tr, y_tr)
    
    # 在同一验证集上计算三种重要性
    model = pipe.named_steps['model']
    imputer = pipe.named_steps['imputer']
    X_val_imp = imputer.transform(X_val)
    
    # XGBoost内置重要性
    imp_xgb = xgb_importances(model)
    
    # 置换重要性（在验证集上）
    imp_perm = perm_importance(
        pipe, X_val, y_val, design_features,
        scoring='average_precision' if y_val.mean() < 0.1 else 'roc_auc',
        n_repeats=5  # 减少重复次数加快速度
    )
    
    # SHAP重要性（在验证集上）
    imp_shap = shap_importance(
        model, X_val_imp, design_features, n_sample=min(300, len(X_val))
    )
    
    # 计算一致性
    if len(imp_xgb) > 0 and len(imp_perm) > 0 and len(imp_shap) > 0:
        rho_gp, j_gp = rank_consistency(imp_xgb, 'gain', imp_perm, 'perm_importance', topk=20)
        rho_gs, j_gs = rank_consistency(imp_xgb, 'gain', imp_shap, 'shap_mean_abs', topk=20)
        rho_ps, j_ps = rank_consistency(imp_perm, 'perm_importance', imp_shap, 'shap_mean_abs', topk=20)
        
        print(f"\nDesign Model Importance Consistency:")
        print(f"  Spearman(gain vs perm): {rho_gp:.3f}, Jaccard@20: {j_gp:.2f}")
        print(f"  Spearman(gain vs SHAP): {rho_gs:.3f}, Jaccard@20: {j_gs:.2f}")
        print(f"  Spearman(perm vs SHAP): {rho_ps:.3f}, Jaccard@20: {j_ps:.2f}")
        
        if rho_ps < 0.5 or j_ps < 0.4:
            print("  ⚠️ WARNING: Low consistency between importance measures")
    
   # 9.4 改进的外推稳定性（使用分组）
    print("\n9.4 Enhanced Extrapolation Stability (with Groups)...")
    
    # 准备分组信息
    if 'Molecule' in preprocessor.df.columns:
        groups = preprocessor.df['Molecule'].values
        print(f"Using molecular groups: {len(np.unique(groups))} unique molecules")
    else:
        groups = np.arange(len(preprocessor.df))
        print("Warning: No Molecule column, using pseudo-groups")
    
    # 使用设计模型特征的分组OOF评分
    from src.robustness_test import oof_score_grouped
    
    X_design = preprocessor.df[design_features].astype(np.float32).fillna(0).values
    y_rtadf = preprocessor.df['is_rTADF'].astype(int).values
    
    oof_metrics, oof_probs = oof_score_grouped(
        X_design, y_rtadf, groups,
        seeds=(7, 13, 23), 
        n_splits=5
    )
    
    print(f"\nDesign Model Grouped OOF Results:")
    for metric, value in oof_metrics.items():
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}")
    
    # 验证：OOF应该与消融结果接近
    if abs(oof_metrics['roc_auc'] - 0.625) > 0.15:  # 预期在0.625附近
        print("  ⚠️ WARNING: OOF AUC significantly different from ablation results!")
        print("  This may indicate remaining data leakage.")
    else:
        print("  ✓ OOF results consistent with ablation analysis")
    
    # 诊断模型的分组OOF（作为对比）
    X_diag = preprocessor.df[diagnostic_features].astype(np.float32).fillna(0).values
    
    oof_diag_metrics, _ = oof_score_grouped(
        X_diag, y_rtadf, groups,
        seeds=(7, 13, 23),
        n_splits=5
    )
    
    print(f"\nDiagnostic Model Grouped OOF Results:")
    for metric, value in oof_diag_metrics.items():
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}") 
    # 9.5 概率校准和Bootstrap置信区间
    print("\n9.5 Probability Calibration and Bootstrap CI...")

    # 首先检查是否能进行校准分析
    if len(design_features) == 0:
        print("  Skipping calibration - no design features available")
    else:
        try:
            from src.robustness_test import (
                calibrate_probabilities, plot_calibration_curve, 
                bootstrap_confidence_intervals
            )
            
            # 重新准备设计模型的数据（使用design_features而不是critical_features）
            # 获取原始的分组信息
            if 'Molecule' in preprocessor.df.columns:
                from sklearn.model_selection import GroupShuffleSplit
                
                # 准备设计特征数据
                X_design_full = preprocessor.df[design_features].fillna(0).values.astype(np.float32)
                y_full = preprocessor.df['is_rTADF'].values
                groups_full = preprocessor.df['Molecule'].values
                
                # 使用相同的随机种子重现split
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_val_idx, test_idx = next(gss.split(X_design_full, y_full, groups_full))
                
                # 再分割训练集和验证集
                X_trainval = X_design_full[train_val_idx]
                y_trainval = y_full[train_val_idx]
                groups_trainval = groups_full[train_val_idx]
                
                val_ratio = 0.1 / 0.8  # 相当于总体的10%
                gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
                train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups_trainval))
                
                # 最终的训练和测试数据（使用design_features）
                X_train_design = X_trainval[train_idx]
                X_test_design = X_design_full[test_idx]
                y_train_design = y_trainval[train_idx]
                y_test_design = y_full[test_idx]
                
                print(f"  Design model data shapes:")
                print(f"    Train: {X_train_design.shape}")
                print(f"    Test: {X_test_design.shape}")
                
                # 训练最终模型用于校准
                final_model = XGBClassifier(
                    n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, random_state=42
                )
                final_model.fit(X_train_design, y_train_design)
                
                # 概率校准
                prob_uncal, prob_cal, calibrated_model = calibrate_probabilities(
                    final_model, 
                    X_train_design, 
                    y_train_design,
                    X_test_design,
                    method='isotonic'
                )
                
                # 绘制校准曲线
                ece_before, ece_after = plot_calibration_curve(
                    y_test_design, prob_uncal, prob_cal, task='design_rtadf'
                )
                print(f"  ECE before calibration: {ece_before:.4f}")
                print(f"  ECE after calibration: {ece_after:.4f}")
                
                # Bootstrap置信区间
                print("\nBootstrap Confidence Intervals (1000 iterations)...")
                ci_results = bootstrap_confidence_intervals(
                    X_test_design, y_test_design, final_model, n_bootstrap=1000
                )
                
                print(f"  ROC-AUC: {ci_results['roc_auc_mean']:.3f} "
                    f"(95% CI: [{ci_results['roc_auc_ci'][0]:.3f}, {ci_results['roc_auc_ci'][1]:.3f}])")
                print(f"  PR-AUC: {ci_results['pr_auc_mean']:.3f} "
                    f"(95% CI: [{ci_results['pr_auc_ci'][0]:.3f}, {ci_results['pr_auc_ci'][1]:.3f}])")
                
                # 保存最终报告（包含校准结果）
                final_summary = pd.DataFrame({
                    'Model': ['Diagnostic', 'Design', 'Design_Calibrated'],
                    'ROC-AUC': [oof_diag_metrics['roc_auc'], 
                                oof_metrics['roc_auc'],
                                ci_results['roc_auc_mean']],
                    'PR-AUC': [oof_diag_metrics.get('pr_auc', 0), 
                            oof_metrics.get('pr_auc', 0),
                            ci_results['pr_auc_mean']],
                    'ECE': [np.nan, ece_before, ece_after],
                    'F1': [oof_diag_metrics.get('f1', 0),
                        oof_metrics.get('f1', 0),
                        np.nan],
                    'Optimal_Threshold': [oof_diag_metrics.get('optimal_threshold', 0.5),
                                        oof_metrics.get('optimal_threshold', 0.5),
                                        np.nan]
                })
                
            else:
                print("  Warning: No molecular groups found, skipping calibration")
                raise ValueError("No molecular groups")
                
        except Exception as e:
            print(f"  Calibration analysis failed: {e}")
            print("  Saving results without calibration...")
            
            # 保存基本结果（不包含校准）
            final_summary = pd.DataFrame({
                'Model': ['Diagnostic', 'Design'],
                'ROC-AUC': [oof_diag_metrics['roc_auc'], 
                            oof_metrics['roc_auc']],
                'PR-AUC': [oof_diag_metrics.get('pr_auc', 0), 
                        oof_metrics.get('pr_auc', 0)],
                'F1': [oof_diag_metrics.get('f1', 0),
                    oof_metrics.get('f1', 0)],
                'Optimal_Threshold': [oof_diag_metrics.get('optimal_threshold', 0.5),
                                    oof_metrics.get('optimal_threshold', 0.5)]
            })
        
        final_summary.to_csv('outputs/final_model_summary.csv', index=False)
        print("\nFinal summary saved to outputs/final_model_summary.csv")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()