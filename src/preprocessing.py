# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer, SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        print(f"Loaded data shape: {self.df.shape}")
        
    def clean_data(self):
        """数据清洗"""
        print("Cleaning data...")
        initial_shape = self.df.shape
        
        # 处理异常值（使用IQR方法）
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 扩展保护特征列表 - 包含所有稀疏特征和计数特征
        protected_features = [
            # 原有的Calicene特征
            'num_in_subs_3ring', 'num_in_subs_5ring', 'num_both_subs',
            'donor_on_3ring', 'donor_on_5ring', 'acceptor_on_3ring', 'acceptor_on_5ring',
            'in_sub_density', 'out_sub_density',
            
            # one-hot特征
            'push_pull_pattern_none', 'push_pull_pattern_D5_A3',
            'push_pull_pattern_D3_A5', 'push_pull_pattern_D5_only',
            'push_pull_pattern_A3_only', 'push_pull_pattern_DD_balanced',
            'push_pull_pattern_AA_balanced',
            'ring_polarity_expected_aligned', 'ring_polarity_expected_reversed',
            'ring_polarity_expected_neutral',
            
            # === 新增：保护所有3ring和5ring的计数特征 ===
            'count_3ring_nh2', 'count_3ring_cn', 'count_3ring_cf3', 
            'count_3ring_oh', 'count_3ring_ome',
            'count_5ring_nh2', 'count_5ring_cn', 'count_5ring_cf3',
            'count_5ring_oh', 'count_5ring_ome',
            
            # === 新增：保护所有has_前缀的二值特征 ===
            'has_3ring', 'has_5ring',
            'has_3ring_nh2', 'has_3ring_cn', 'has_3ring_cf3', 'has_3ring_oh',
            'has_5ring_nh2', 'has_5ring_cn', 'has_5ring_cf3', 'has_5ring_oh',
            
            # === 新增：保护所有count_前缀的计数特征 ===
            'count_cyano', 'count_nitro', 'count_amino', 'count_carbonyl',
            'count_sulfone', 'count_triazine', 'count_boron', 'count_phosphorus',
            'count_halogen', 'count_methyl', 'count_trifluoromethyl', 'count_phenyl',
            'count_pyridine', 'count_thiophene', 'count_furan', 'count_pyrrole',
            'count_carbazole', 'count_triphenylamine',
            
            # === 新增：保护所有has_前缀的存在性特征 ===
            'has_cyano', 'has_nitro', 'has_amino', 'has_carbonyl', 'has_sulfone',
            'has_triazine', 'has_boron', 'has_phosphorus', 'has_halogen',
            'has_methyl', 'has_trifluoromethyl', 'has_phenyl', 'has_pyridine',
            'has_thiophene', 'has_furan', 'has_pyrrole', 'has_carbazole',
            'has_triphenylamine',
            
            # === 新增：保护反转相关特征 ===
            'num_inverted_gaps', 'has_inversion', 'has_s1_t2_inversion',
            'favorable_for_inversion',
            
            # === 新增：保护小环计数 ===
            'num_3_member_rings', 'num_4_member_rings', 'num_5_member_rings',
            'num_6_member_rings', 'num_7_member_rings', 'num_8_member_rings',
        ]
        
        # 动态识别稀疏特征（非零值比例小于10%的特征）
        for col in numeric_cols:
            if col in self.df.columns and col not in protected_features:
                non_zero_ratio = (self.df[col] != 0).sum() / len(self.df)
                
                # 如果是稀疏特征（非零值少于10%）或二值特征，加入保护列表
                unique_values = self.df[col].nunique()
                is_binary = unique_values <= 2
                is_sparse = non_zero_ratio < 0.1
                is_count_feature = col.startswith(('count_', 'num_', 'has_'))
                
                if is_binary or is_sparse or is_count_feature:
                    protected_features.append(col)
                    print(f"  Protecting sparse/binary feature: {col} (non-zero ratio: {non_zero_ratio:.2%})")
        
        # 执行IQR异常值处理（只对非保护特征）
        processed_count = 0
        for col in numeric_cols:
            if col in self.df.columns and col not in protected_features:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # 额外检查：如果IQR为0，跳过该特征
                if IQR == 0:
                    print(f"  Skipping {col} - IQR is 0")
                    continue
                    
                lower = Q1 - 3 * IQR
                upper = Q3 + 3 * IQR
                
                # 记录处理前后的变化
                before_unique = self.df[col].nunique()
                self.df[col] = self.df[col].clip(lower, upper)
                after_unique = self.df[col].nunique()
                
                if before_unique != after_unique:
                    print(f"  Processed {col}: unique values {before_unique} -> {after_unique}")
                    processed_count += 1
        
        print(f"Data cleaned: {initial_shape} -> {self.df.shape}")
        print(f"Applied IQR to {processed_count} features, protected {len(protected_features)} features")
        
        # 验证关键稀疏特征是否保持不变
        sparse_features_to_check = [
            'count_3ring_nh2', 'count_3ring_cn', 'count_5ring_nh2', 'count_5ring_cn',
            'has_3ring', 'has_5ring', 'num_inverted_gaps'
        ]
        
        print("\nVerifying sparse features preservation:")
        for feat in sparse_features_to_check:
            if feat in self.df.columns:
                non_zero = (self.df[feat] != 0).sum()
                unique_vals = self.df[feat].nunique()
                print(f"  {feat}: {non_zero} non-zero values, {unique_vals} unique values")
        
        return self
    
    def handle_missing_values(self):
        """处理缺失值"""
        print("Handling missing values...")
        
        # 首先统计缺失值
        missing_stats = self.df.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]
        if len(missing_cols) > 0:
            print(f"Found {len(missing_cols)} columns with missing values")
            print(f"Top columns with missing values: {missing_cols.head(10).to_dict()}")
        sparse_features = [
            'has_3ring_nh2', 'has_3ring_cn', 'has_3ring_cf3', 'has_3ring_oh',
            'has_5ring_nh2', 'has_5ring_cn', 'has_5ring_cf3', 'has_5ring_oh',
            'count_3ring_nh2', 'count_3ring_cn', 'count_3ring_cf3',
            'count_5ring_nh2', 'count_5ring_cn', 'count_5ring_cf3'
        ]
        
        # 保存非缺失值的位置和值
        sparse_data = {}
        for feat in sparse_features:
            if feat in self.df.columns:
                # 记录非缺失值的位置
                non_missing_mask = ~self.df[feat].isna()
                sparse_data[feat] = {
                    'values': self.df[feat][non_missing_mask].copy(),
                    'mask': non_missing_mask
                }
                print(f"  Saving sparse feature {feat}: {(self.df[feat] == 1).sum()} ones, {self.df[feat].isna().sum()} missing")
        # ========= 修改结束 =========
        # 分离数值列和分类列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # 处理数值列
        if numeric_cols:
            # 移除全为NaN的列
            cols_to_drop = [col for col in numeric_cols 
                          if self.df[col].isnull().all()]
            if cols_to_drop:
                print(f"Dropping columns with all NaN values: {cols_to_drop}")
                self.df = self.df.drop(columns=cols_to_drop)
                numeric_cols = [col for col in numeric_cols if col not in cols_to_drop]
            
            # 对于缺失值太多的列（>80%），用均值填充
            high_missing_cols = []
            for col in numeric_cols:
                if col in self.df.columns:
                    missing_ratio = self.df[col].isnull().sum() / len(self.df)
                    if missing_ratio > 0.8:
                        high_missing_cols.append(col)
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
            
            if high_missing_cols:
                print(f"Filled {len(high_missing_cols)} columns with >80% missing values using mean")
            
            # 对于其他列，使用KNN填充
            low_missing_cols = [col for col in numeric_cols 
                               if col in self.df.columns and col not in high_missing_cols 
                               and self.df[col].isnull().sum() > 0]
            
            if low_missing_cols:
                try:
                    # 使用KNNImputer
                    imputed_values = self.imputer.fit_transform(self.df[low_missing_cols])
                    self.df[low_missing_cols] = imputed_values
                    print(f"Applied KNN imputation to {len(low_missing_cols)} columns")
                except Exception as e:
                    print(f"KNN imputation failed: {e}. Using mean imputation instead.")
                    # 如果KNN失败，使用均值填充
                    for col in low_missing_cols:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        # 处理分类变量
        if categorical_cols:
            for col in categorical_cols:
                if col in self.df.columns:
                    mode_value = self.df[col].mode()
                    fill_value = mode_value[0] if not mode_value.empty else 'unknown'
                    self.df[col].fillna(fill_value, inplace=True)
            print(f"Filled categorical columns with mode values")
        
        # 最终检查
        remaining_missing = self.df.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: Still have {remaining_missing} missing values after imputation")
            # 用0填充剩余的缺失值
            self.df.fillna(0, inplace=True)
        for feat, data_dict in sparse_data.items():
            # 先将整列设为0
            self.df[feat] = 0
            # 然后恢复原始的非缺失值
            self.df.loc[data_dict['mask'], feat] = data_dict['values'].values
            non_zero = (self.df[feat] != 0).sum()
            total = len(self.df[feat])
            print(f"  Restored sparse feature {feat}: {non_zero} non-zero values out of {total} total")
        # ========= 修改结束 =========
        
        
        print("Missing value handling completed")
        return self
    
    def normalize_features(self, feature_cols):
        """特征标准化"""
        print(f"Normalizing {len(feature_cols)} features...")
        
        # 只标准化存在的列
        valid_cols = [col for col in feature_cols if col in self.df.columns]
        
        if valid_cols:
            # 确保数据是数值型
            for col in valid_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # 标准化
            self.df[valid_cols] = self.scaler.fit_transform(self.df[valid_cols])
            print(f"Normalized {len(valid_cols)} columns")
        else:
            print("Warning: No valid columns to normalize")
        
        return self
    def aggregate_by_molecule(self, method='mean'):
        """
        分子级聚合 - 将同一分子的多个构象聚合成一条记录
        
        Args:
            method: 聚合方法 ('mean', 'boltzmann', 'min_gap')
        """
        print(f"\n=== Molecular Aggregation ===")
        print(f"Before: {len(self.df)} conformers")
        print(f"Unique molecules: {self.df['Molecule'].nunique()}")
        
        # 分离不同类型的列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = ['Molecule', 'smiles', 'SMILES']
        label_cols = ['is_TADF', 'is_rTADF']
        
        # 移除标签和ID列
        numeric_cols = [c for c in numeric_cols if c not in label_cols]
        
        # 定义聚合规则
        agg_rules = {}
        
        if method == 'mean':
            # 简单平均
            for col in numeric_cols:
                if col in self.df.columns:
                    agg_rules[col] = 'mean'
        
        elif method == 'boltzmann':
            # 玻尔兹曼加权平均
            def boltzmann_weights(energies, T=300):
                """计算玻尔兹曼权重"""
                kT = 8.617333262e-5 * T  # eV
                if 'energy' in energies.name or 'crest_energy' in energies.name:
                    de = energies - energies.min()
                    w = np.exp(-de / kT)
                    return w / (w.sum() + 1e-12)
                return np.ones(len(energies)) / len(energies)
            
            # 使用加权平均
            for col in numeric_cols:
                if col in self.df.columns:
                    if 'gap' in col:
                        # 对gap特征取最小绝对值
                        agg_rules[col] = lambda x: x.loc[x.abs().idxmin()] if len(x) > 0 else 0
                    else:
                        agg_rules[col] = 'mean'
        
        elif method == 'min_gap':
            # 对gap取最小，其他取平均
            for col in numeric_cols:
                if col in self.df.columns:
                    if 'gap' in col:
                        # 对gap特征取绝对值最小的那个值（保留符号）
                        agg_rules[col] = lambda x: x.loc[x.abs().idxmin()] if len(x) > 0 else 0
                    elif 'crest' in col:
                        # CREST特征保留多样性信息
                        if 'num_conformers' in col:
                            agg_rules[col] = 'first'  # 保持原值
                        elif 'std' in col or 'diversity' in col:
                            agg_rules[col] = 'max'  # 取最大多样性
                        else:
                            agg_rules[col] = 'mean'
                    else:
                        agg_rules[col] = 'mean'
        
        # 标签聚合规则
        if 'is_TADF' in self.df.columns:
            agg_rules['is_TADF'] = lambda x: 1 if x.max() > 0 else 0  # 任一构象满足即为真
        if 'is_rTADF' in self.df.columns:
            agg_rules['is_rTADF'] = lambda x: 1 if x.max() > 0 else 0
        
        # 执行聚合
        print(f"Aggregating {len(agg_rules)} features...")
        self.df_molecule = self.df.groupby('Molecule').agg(agg_rules).reset_index()
        
        # 添加构象计数（聚合后添加）
        conformer_counts = self.df.groupby('Molecule').size().reset_index(name='num_conformers_total')
        self.df_molecule = self.df_molecule.merge(conformer_counts, on='Molecule', how='left')
        
        # 添加构象多样性特征（如果原始数据中存在这些列）
        diversity_features = []
        for col in ['planarity_ratio', 'gaussian_eccentricity', 'crest_avg_rmsd']:
            if col in self.df.columns:
                # 计算标准差作为多样性指标
                std_col = f'{col}_std'
                std_values = self.df.groupby('Molecule')[col].std().reset_index()
                std_values.columns = ['Molecule', std_col]
                self.df_molecule = self.df_molecule.merge(std_values, on='Molecule', how='left')
                diversity_features.append(std_col)
        
        if diversity_features:
            print(f"Added {len(diversity_features)} diversity features: {diversity_features}")
        
        print(f"After: {len(self.df_molecule)} molecules")
        print(f"Aggregation method: {method}")
        
        # 统计标签分布
        if 'is_TADF' in self.df_molecule.columns:
            tadf_mol = self.df_molecule['is_TADF'].sum()
            print(f"TADF molecules: {tadf_mol}/{len(self.df_molecule)} ({tadf_mol/len(self.df_molecule)*100:.1f}%)")
        
        if 'is_rTADF' in self.df_molecule.columns:
            rtadf_mol = self.df_molecule['is_rTADF'].sum()
            print(f"rTADF molecules: {rtadf_mol}/{len(self.df_molecule)} ({rtadf_mol/len(self.df_molecule)*100:.1f}%)")
        
        # 保存原始构象级数据
        self.df_conformers = self.df.copy()
        # 使用分子级数据
        self.df = self.df_molecule
        
        return self
    def create_labels(self):
        """创建TADF和rTADF标签 - 基于严格的文献标准"""
        print("\n=== Creating labels based on literature criteria ===")
        
        # 检查必要的列是否存在
        required_cols = ['s1_t1_gap']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"Warning: Missing required columns for label creation: {missing_cols}")
            self.df['is_TADF'] = 0
            self.df['is_rTADF'] = 0
            self.df['is_TADF_relaxed'] = 0
            self.df['is_rTADF_relaxed'] = 0
            self.df['is_near_degenerate'] = 0
        else:
            # ========= PRIMARY LABELS (严格的文献标准) =========
            print("\n1. Primary labels (strict literature criteria):")
            
            # TADF Primary: 基于文献的保守阈值
            # Ref: ΔE_ST ≤ 0.1 eV is the empirical target for efficient TADF
            tadf_s1t1 = self.df['s1_t1_gap'].abs() <= 0.10  # 主要条件
            
            # 考虑高三重态介导的TADF
            tadf_s1t2 = False
            tadf_s1t3 = False
            if 's1_t2_gap' in self.df.columns:
                tadf_s1t2 = self.df['s1_t2_gap'].abs() <= 0.15
            if 's1_t3_gap' in self.df.columns:
                tadf_s1t3 = self.df['s1_t3_gap'].abs() <= 0.20
            
            # 振子强度过滤（非常宽松，只排除完全暗态）
            if 's1_oscillator' in self.df.columns:
                has_oscillator = self.df['s1_oscillator'] >= 1e-4  # 极宽松阈值
            else:
                has_oscillator = True  # 如果没有数据，不过滤
            
            # 组合TADF条件
            self.df['is_TADF'] = ((tadf_s1t1 | tadf_s1t2 | tadf_s1t3) & has_oscillator).astype(int)
            
            # rTADF Primary: 严格要求负gap（真正的能级反转）
            # Ref: Singlet-triplet inversion requires E(S1) < E(T1)
            rtadf_s1t1 = self.df['s1_t1_gap'] <= -0.02  # 允许-0.02 eV的计算噪声
            
            rtadf_s1t2 = False
            rtadf_s1t3 = False
            if 's1_t2_gap' in self.df.columns:
                rtadf_s1t2 = self.df['s1_t2_gap'] <= -0.05
            if 's1_t3_gap' in self.df.columns:
                rtadf_s1t3 = self.df['s1_t3_gap'] <= -0.05
            
            # 高单重态与低三重态的反转（S2 < T1等）
            rtadf_s2t1 = False
            if 's2_t1_gap' in self.df.columns:
                rtadf_s2t1 = self.df['s2_t1_gap'] <= -0.10  # S2明显低于T1
            
            # 组合rTADF条件
            self.df['is_rTADF'] = (rtadf_s1t1 | rtadf_s1t2 | rtadf_s1t3 | rtadf_s2t1).astype(int)
            
            # ========= SENSITIVITY LABELS (敏感性分析用的宽松标准) =========
            print("\n2. Sensitivity labels (relaxed criteria for sensitivity analysis):")
            
            # TADF Relaxed: 放宽阈值
            tadf_relax_s1t1 = self.df['s1_t1_gap'].abs() <= 0.20  # 放宽到0.2 eV
            tadf_relax_s1t2 = False
            tadf_relax_s1t3 = False
            if 's1_t2_gap' in self.df.columns:
                tadf_relax_s1t2 = self.df['s1_t2_gap'].abs() <= 0.25
            if 's1_t3_gap' in self.df.columns:
                tadf_relax_s1t3 = self.df['s1_t3_gap'].abs() <= 0.30
            
            self.df['is_TADF_relaxed'] = ((tadf_relax_s1t1 | tadf_relax_s1t2 | tadf_relax_s1t3) & has_oscillator).astype(int)
            
            # rTADF Relaxed: 稍微放宽负gap要求
            rtadf_relax_s1t1 = self.df['s1_t1_gap'] <= -0.01  # 放宽到-0.01 eV
            rtadf_relax_s1t2 = False
            rtadf_relax_s1t3 = False
            if 's1_t2_gap' in self.df.columns:
                rtadf_relax_s1t2 = self.df['s1_t2_gap'] <= -0.02
            if 's1_t3_gap' in self.df.columns:
                rtadf_relax_s1t3 = self.df['s1_t3_gap'] <= -0.02
            
            self.df['is_rTADF_relaxed'] = (rtadf_relax_s1t1 | rtadf_relax_s1t2 | rtadf_relax_s1t3).astype(int)
            
            # ========= NEAR-DEGENERATE (近简并但未反转，不是rTADF!) =========
            # 这些分子有很小的正gap，可能是TADF候选，但绝不是rTADF
            near_degenerate = (self.df['s1_t1_gap'] > -0.01) & (self.df['s1_t1_gap'] <= 0.05)
            self.df['is_near_degenerate'] = near_degenerate.astype(int)
            
            # ========= ADDITIONAL CATEGORIZATION =========
            # rTADF类型细分
            self.df['rTADF_type'] = 'none'
            
            # 标记主要反转类型
            mask_s1t1 = self.df['s1_t1_gap'] <= -0.02
            self.df.loc[mask_s1t1, 'rTADF_type'] = 'S1<T1'
            
            if 's1_t2_gap' in self.df.columns:
                mask_s1t2 = (self.df['s1_t2_gap'] <= -0.05) & (self.df['rTADF_type'] == 'none')
                self.df.loc[mask_s1t2, 'rTADF_type'] = 'S1<T2'
            
            if 's2_t1_gap' in self.df.columns:
                mask_s2t1 = (self.df['s2_t1_gap'] <= -0.10) & (self.df['rTADF_type'] == 'none')
                self.df.loc[mask_s2t1, 'rTADF_type'] = 'S2<T1'
            
            # 多重反转
            num_inversions = 0
            for i in range(1, 6):
                gap_col = f's1_t{i}_gap'
                if gap_col in self.df.columns:
                    num_inversions += (self.df[gap_col] <= -0.02).astype(int)
            
            mask_multi = (num_inversions > 1)
            self.df.loc[mask_multi, 'rTADF_type'] = 'multiple'
            
            # TADF强度分级（基于gap大小）
            self.df['TADF_strength'] = 'none'
            
            # 强TADF：非常小的gap
            strong_mask = (self.df['s1_t1_gap'].abs() <= 0.05) & (self.df['is_TADF'] == 1)
            self.df.loc[strong_mask, 'TADF_strength'] = 'strong'
            
            # 中等TADF
            medium_mask = (self.df['s1_t1_gap'].abs() > 0.05) & (self.df['s1_t1_gap'].abs() <= 0.10) & (self.df['is_TADF'] == 1)
            self.df.loc[medium_mask, 'TADF_strength'] = 'medium'
            
            # 弱TADF（使用relaxed标准）
            weak_mask = (self.df['s1_t1_gap'].abs() > 0.10) & (self.df['is_TADF_relaxed'] == 1)
            self.df.loc[weak_mask, 'TADF_strength'] = 'weak'
        
        # ========= 统计输出 =========
        # Primary标签统计
        tadf_count = self.df['is_TADF'].sum()
        rtadf_count = self.df['is_rTADF'].sum()
        total_count = len(self.df)
        
        print(f"\n=== Primary Labels Statistics ===")
        print(f"TADF molecules (ΔE_ST ≤ 0.1 eV): {tadf_count}/{total_count} ({tadf_count/total_count*100:.1f}%)")
        print(f"rTADF molecules (negative gap): {rtadf_count}/{total_count} ({rtadf_count/total_count*100:.1f}%)")
        
        # Relaxed标签统计
        tadf_relax_count = self.df['is_TADF_relaxed'].sum()
        rtadf_relax_count = self.df['is_rTADF_relaxed'].sum()
        near_deg_count = self.df['is_near_degenerate'].sum()
        
        print(f"\n=== Sensitivity Labels Statistics ===")
        print(f"TADF relaxed (ΔE_ST ≤ 0.2 eV): {tadf_relax_count}/{total_count} ({tadf_relax_count/total_count*100:.1f}%)")
        print(f"rTADF relaxed (less strict): {rtadf_relax_count}/{total_count} ({rtadf_relax_count/total_count*100:.1f}%)")
        print(f"Near-degenerate (0 < ΔE_ST ≤ 0.05 eV): {near_deg_count}/{total_count} ({near_deg_count/total_count*100:.1f}%)")
        
        # 详细的gap分布
        if 's1_t1_gap' in self.df.columns:
            gap = self.df['s1_t1_gap']
            print(f"\n=== S1-T1 Gap Distribution ===")
            print(f"Negative gap (< -0.02 eV): {(gap <= -0.02).sum()}")
            print(f"Near zero (-0.02 to 0.02 eV): {((gap > -0.02) & (gap <= 0.02)).sum()}")
            print(f"Small positive (0.02 to 0.1 eV): {((gap > 0.02) & (gap <= 0.1)).sum()}")
            print(f"Moderate (0.1 to 0.3 eV): {((gap > 0.1) & (gap <= 0.3)).sum()}")
            print(f"Large (> 0.3 eV): {(gap > 0.3).sum()}")
            
            print(f"\nGap statistics:")
            print(f"  Min: {gap.min():.3f} eV")
            print(f"  Max: {gap.max():.3f} eV")
            print(f"  Mean: {gap.mean():.3f} eV")
            print(f"  Median: {gap.median():.3f} eV")
        
        # rTADF类型分布
        if 'rTADF_type' in self.df.columns and rtadf_count > 0:
            rtadf_types = self.df[self.df['is_rTADF'] == 1]['rTADF_type'].value_counts()
            print(f"\n=== rTADF Types (Primary) ===")
            for rtype, count in rtadf_types.items():
                print(f"  {rtype}: {count}")
        
        # TADF强度分布
        if 'TADF_strength' in self.df.columns and tadf_count > 0:
            tadf_strength = self.df[self.df['is_TADF'] == 1]['TADF_strength'].value_counts()
            print(f"\n=== TADF Strength Distribution ===")
            for strength, count in tadf_strength.items():
                print(f"  {strength}: {count}")
        
        # 警告
        if tadf_count < 5:
            print(f"\n⚠️ Warning: Very few TADF samples ({tadf_count}) with strict criteria")
            print(f"   Consider using relaxed criteria for initial exploration")
        if rtadf_count < 5:
            print(f"⚠️ Warning: Very few rTADF samples ({rtadf_count}) with negative gap")
            print(f"   This is expected - true singlet-triplet inversion is rare")
        
        # 建议
        print(f"\n=== Recommendations ===")
        print("1. Use primary labels for main results (physically meaningful)")
        print("2. Report sensitivity analysis with relaxed labels")
        print("3. Clearly state that 'near-degenerate' are NOT rTADF")
        print("4. Consider reporting results stratified by gap ranges")
        
        return self
    def split_data(self, test_size=0.2, val_size=0.1, use_features=None, use_molecular_split=True):
        """
        数据集划分 - 支持分子级别的分组
        
        Args:
            test_size: 测试集比例
            val_size: 验证集比例
            use_features: 指定要使用的特征列表
            use_molecular_split: 是否使用分子级别切分（防止泄漏）
        """
        print("Splitting data...")
        
        # ========= 原有的特征选择代码（保持不变）=========
        # 如果指定了特征列表，使用指定的特征
        if use_features is not None:
            feature_cols = [f for f in use_features if f in self.df.columns]
            print(f"Using specified {len(feature_cols)} features")
        else:
            # 特征列选择（排除标签和非数值列）
            exclude_cols = [
                'Molecule', 'State', 'conformer', 
                'is_TADF', 'is_rTADF', 'inverted_gaps',
                'singlet_states', 'triplet_states',
                'smiles', 'SMILES',
                'primary_inversion_type',
                'rTADF_type', 'TADF_strength'  # 添加新的分类列
            ]
            
            # 获取所有数值列作为特征
            feature_cols = []
            for col in self.df.columns:
                if col not in exclude_cols:
                    # 尝试转换为数值类型
                    try:
                        # 对于布尔类型，转换为整数
                        if self.df[col].dtype == bool:
                            self.df[col] = self.df[col].astype(int)
                            feature_cols.append(col)
                        # 创建临时列来测试转换
                        elif pd.api.types.is_numeric_dtype(self.df[col]):
                            feature_cols.append(col)
                        else:
                            temp_col = pd.to_numeric(self.df[col], errors='coerce')
                            # 如果转换后非空值比例大于50%，则认为是有效数值列
                            if temp_col.notna().sum() / len(temp_col) > 0.5:
                                self.df[col] = temp_col  # 实际转换
                                feature_cols.append(col)
                    except:
                        continue
        
        print(f"Selected {len(feature_cols)} feature columns")
        
        # ========= One-hot编码处理（保持不变）=========
        push_pull_onehot = ['push_pull_pattern_none', 'push_pull_pattern_D5_A3',
                        'push_pull_pattern_D3_A5', 'push_pull_pattern_D5_only',
                        'push_pull_pattern_A3_only', 'push_pull_pattern_DD_balanced',
                        'push_pull_pattern_AA_balanced']
        
        polarity_onehot = ['ring_polarity_expected_aligned',
                        'ring_polarity_expected_reversed',
                        'ring_polarity_expected_neutral']
        
        # 检查是否需要编码
        has_onehot = any(col in self.df.columns for col in push_pull_onehot + polarity_onehot)
        
        if not has_onehot:
            # 如果没有one-hot特征，说明需要编码原始字符串特征
            string_features_to_encode = ['push_pull_pattern', 'ring_polarity_expected']
            for feat in string_features_to_encode:
                if feat in self.df.columns and feat in feature_cols:
                    print(f"One-hot encoding {feat}...")
                    dummies = pd.get_dummies(self.df[feat], prefix=feat)
                    for col in dummies.columns:
                        self.df[col] = dummies[col]
                        feature_cols.append(col)
                    feature_cols.remove(feat)
                    print(f"  Created {len(dummies.columns)} encoded features from {feat}")
        else:
            print("One-hot encoded features already exist, skipping encoding")
        
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found!")
        
        # ========= 准备特征和标签 =========
        # 提取特征 - 确保完全是数值型
        X_df = self.df[feature_cols].copy()
        
        # 填充缺失值
        X_df = X_df.fillna(0)
        
        # 强制转换为float64，确保没有object类型
        X = X_df.astype(np.float64).values
        
        # 检查是否有无效值
        if np.any(np.isnan(X)):
            print("Warning: Found NaN values after conversion, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isinf(X)):
            print("Warning: Found infinite values, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 提取标签
        y_tadf = self.df['is_TADF'].astype(np.float32).values
        y_rtadf = self.df['is_rTADF'].astype(np.float32).values
        
        print(f"Feature matrix shape: {X.shape}, dtype: {X.dtype}")
        print(f"TADF labels shape: {y_tadf.shape}, dtype: {y_tadf.dtype}")
        print(f"rTADF labels shape: {y_rtadf.shape}, dtype: {y_rtadf.dtype}")
        
        # 验证数据类型
        assert X.dtype in [np.float64, np.float32], f"X has wrong dtype: {X.dtype}"
        assert y_tadf.dtype in [np.float64, np.float32, np.int32, np.int64], f"y_tadf has wrong dtype: {y_tadf.dtype}"
        assert y_rtadf.dtype in [np.float64, np.float32, np.int32, np.int64], f"y_rtadf has wrong dtype: {y_rtadf.dtype}"
        
        # ========= 新增：分子级别切分 =========
        if use_molecular_split and 'Molecule' in self.df.columns:
            from sklearn.model_selection import GroupShuffleSplit
            
            print("\n=== Using molecular-level split to prevent leakage ===")
            
            groups = self.df['Molecule'].values
            unique_molecules = len(np.unique(groups))
            print(f"  Total samples: {len(self.df)}")
            print(f"  Unique molecules: {unique_molecules}")
            
            # 打印标签分布
            print(f"  TADF positive rate: {y_tadf.mean():.2%}")
            print(f"  rTADF positive rate: {y_rtadf.mean():.2%}")
            
            # 第一次分割：训练+验证 vs 测试
            gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_val_idx, test_idx = next(gss1.split(X, y_tadf, groups))
            
            # 第二次分割：训练 vs 验证
            X_trainval = X[train_val_idx]
            y_tadf_trainval = y_tadf[train_val_idx]
            y_rtadf_trainval = y_rtadf[train_val_idx]
            groups_trainval = groups[train_val_idx]
            
            val_ratio = val_size / (1 - test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
            train_idx, val_idx = next(gss2.split(X_trainval, y_tadf_trainval, groups_trainval))
            
            # 最终的数据集
            X_train = X_trainval[train_idx]
            X_val = X_trainval[val_idx]
            X_test = X[test_idx]
            
            y_tadf_train = y_tadf_trainval[train_idx]
            y_tadf_val = y_tadf_trainval[val_idx]
            y_tadf_test = y_tadf[test_idx]
            
            y_rtadf_train = y_rtadf_trainval[train_idx]
            y_rtadf_val = y_rtadf_trainval[val_idx]
            y_rtadf_test = y_rtadf[test_idx]
            
            # 打印分子分布
            train_molecules = len(np.unique(groups_trainval[train_idx]))
            val_molecules = len(np.unique(groups_trainval[val_idx]))
            test_molecules = len(np.unique(groups[test_idx]))
            
            print(f"\n  Train: {len(X_train)} samples from {train_molecules} molecules")
            print(f"    TADF: {y_tadf_train.sum()}/{len(y_tadf_train)} positives")
            print(f"    rTADF: {y_rtadf_train.sum()}/{len(y_rtadf_train)} positives")
            
            print(f"  Val: {len(X_val)} samples from {val_molecules} molecules")
            print(f"    TADF: {y_tadf_val.sum()}/{len(y_tadf_val)} positives")
            print(f"    rTADF: {y_rtadf_val.sum()}/{len(y_rtadf_val)} positives")
            
            print(f"  Test: {len(X_test)} samples from {test_molecules} molecules")
            print(f"    TADF: {y_tadf_test.sum()}/{len(y_tadf_test)} positives")
            print(f"    rTADF: {y_rtadf_test.sum()}/{len(y_rtadf_test)} positives")
            
            # 检查是否有分子泄漏
            train_mols = set(groups_trainval[train_idx])
            val_mols = set(groups_trainval[val_idx])
            test_mols = set(groups[test_idx])
            
            overlap_train_val = train_mols & val_mols
            overlap_train_test = train_mols & test_mols
            overlap_val_test = val_mols & test_mols
            
            if overlap_train_val or overlap_train_test or overlap_val_test:
                print(f"\n  ⚠️ WARNING: Molecular overlap detected!")
                print(f"    Train-Val overlap: {len(overlap_train_val)} molecules")
                print(f"    Train-Test overlap: {len(overlap_train_test)} molecules")
                print(f"    Val-Test overlap: {len(overlap_val_test)} molecules")
            else:
                print(f"\n  ✓ No molecular overlap - split is valid")
                
        else:
            # ========= 原有的随机切分（不推荐）=========
            print("\n⚠️ Using standard random split (may have molecular leakage)")
            
            # 训练集/测试集划分
            try:
                # 如果TADF标签都相同，不使用stratify
                if len(np.unique(y_tadf)) > 1:
                    X_temp, X_test, y_tadf_temp, y_tadf_test, y_rtadf_temp, y_rtadf_test = \
                        train_test_split(X, y_tadf, y_rtadf, test_size=test_size, 
                                    random_state=42, stratify=y_tadf)
                else:
                    X_temp, X_test, y_tadf_temp, y_tadf_test, y_rtadf_temp, y_rtadf_test = \
                        train_test_split(X, y_tadf, y_rtadf, test_size=test_size, 
                                    random_state=42)
            except Exception as e:
                print(f"Stratified split failed: {e}. Using random split.")
                X_temp, X_test, y_tadf_temp, y_tadf_test, y_rtadf_temp, y_rtadf_test = \
                    train_test_split(X, y_tadf, y_rtadf, test_size=test_size, 
                                random_state=42)
            
            # 训练集/验证集划分
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_tadf_train, y_tadf_val, y_rtadf_train, y_rtadf_val = \
                train_test_split(X_temp, y_tadf_temp, y_rtadf_temp, 
                            test_size=val_ratio, random_state=42)
        
        # ========= 确保返回float32类型 =========
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        print(f"\nData split complete:")
        print(f"  Train: {X_train.shape[0]} samples, dtype: {X_train.dtype}")
        print(f"  Val: {X_val.shape[0]} samples, dtype: {X_val.dtype}")
        print(f"  Test: {X_test.shape[0]} samples, dtype: {X_test.dtype}")
        
        # 保存特征列名称
        self.feature_cols = feature_cols
        
        return (X_train, X_val, X_test), \
            (y_tadf_train, y_tadf_val, y_tadf_test), \
            (y_rtadf_train, y_rtadf_val, y_rtadf_test)