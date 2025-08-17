#src/feature_engineering.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    """特征工程"""
    
    def __init__(self, df):
        self.df = df
    
    def create_interaction_features(self):
        """创建交互特征"""
        # ========= 提前创建密度特征（避免标准化问题）=========
        # 官能团密度特征
        if 'num_heavy_atoms' in self.df.columns and self.df['num_heavy_atoms'].min() > 0:
            # 基础官能团密度
            for fg in ['cyano', 'nitro', 'amino', 'carbonyl']:
                count_col = f'count_{fg}'
                if count_col in self.df.columns:
                    self.df[f'{fg}_density'] = self.df[count_col] / self.df['num_heavy_atoms']
            
            # 从名称提取的硝基密度
            if 'count_nitro' not in self.df.columns and 'has_nitro' in self.df.columns:
                self.df['count_nitro'] = self.df['has_nitro']
            if 'count_nitro' in self.df.columns:
                self.df['nitro_density'] = self.df['count_nitro'] / self.df['num_heavy_atoms']
        
        # 芳香环-能隙乘积
        if 'num_aromatic_rings' in self.df.columns and 's1_t1_gap' in self.df.columns:
            self.df['aromatic_gap_product'] = self.df['num_aromatic_rings'] * self.df['s1_t1_gap'].abs()
        # ========= 基础交互特征 =========
        # 电子效应特征
        if 'has_amino' in self.df.columns and 'has_nitro' in self.df.columns:
            self.df['donor_acceptor_balance'] = self.df['has_amino'] - self.df['has_nitro']
        
        # 共轭-刚性比
        if 'num_conjugated_bonds' in self.df.columns and 'num_rotatable_bonds' in self.df.columns:
            self.df['conjugation_rigidity_ratio'] = (
                self.df['num_conjugated_bonds'] / 
                (self.df['num_rotatable_bonds'] + 1)
            )
        
        # ========= 官能团组合特征 =========
        # 电子给体得分
        donor_cols = ['has_amino', 'has_triphenylamine', 'count_methyl', 'has_phenyl']
        donor_score = 0
        for col in donor_cols:
            if col in self.df.columns:
                weight = 3 if 'triphenylamine' in col else 2 if 'amino' in col else 0.5
                donor_score += self.df[col] * weight
        self.df['donor_score'] = donor_score
        
        # 电子受体得分
        acceptor_cols = ['has_cyano', 'has_nitro', 'has_carbonyl', 'has_triazine', 'has_sulfone']
        acceptor_score = 0
        for col in acceptor_cols:
            if col in self.df.columns:
                weight = 3 if 'cyano' in col else 2 if 'nitro' in col or 'triazine' in col else 1
                acceptor_score += self.df[col] * weight
        self.df['acceptor_score'] = acceptor_score
        
        # D-A平衡特征
        self.df['D_A_ratio'] = donor_score / (acceptor_score + 1)
        self.df['D_A_product'] = donor_score * acceptor_score
        self.df['is_D_A_molecule'] = ((donor_score > 0) & (acceptor_score > 0)).astype(int)
        
        # D-A强度特征
        self.df['DA_strength'] = self.df['donor_score'] * self.df['acceptor_score']
        self.df['DA_balance'] = np.abs(self.df['donor_score'] - self.df['acceptor_score'])
        
        # 分子类型编码
        self.df['mol_type_code'] = 0  # 默认
        self.df.loc[(self.df['donor_score'] > 2) & (self.df['acceptor_score'] > 2), 'mol_type_code'] = 3  # 强D-A
        self.df.loc[(self.df['donor_score'] > 2) & (self.df['acceptor_score'] <= 2), 'mol_type_code'] = 1  # 给体主导
        self.df.loc[(self.df['donor_score'] <= 2) & (self.df['acceptor_score'] > 2), 'mol_type_code'] = 2  # 受体主导
        
        # ========= 官能团-能级交互特征 =========
        if 'homo' in self.df.columns:
            if 'has_amino' in self.df.columns:
                self.df['amino_homo_effect'] = self.df['has_amino'] * self.df['homo']
            if 'donor_score' in self.df.columns:
                self.df['donor_homo_effect'] = self.df['donor_score'] * self.df['homo']
        
        if 'lumo' in self.df.columns:
            if 'has_cyano' in self.df.columns:
                self.df['cyano_lumo_effect'] = self.df['has_cyano'] * self.df['lumo']
            if 'acceptor_score' in self.df.columns:
                self.df['acceptor_lumo_effect'] = self.df['acceptor_score'] * self.df['lumo']
        
        # 官能团对S-T gap的影响
        if 's1_t1_gap' in self.df.columns:
            if 'has_triphenylamine' in self.df.columns:
                self.df['tpa_st_gap_effect'] = self.df['has_triphenylamine'] * self.df['s1_t1_gap'].abs()
            self.df['DA_st_gap_effect'] = self.df['D_A_product'] * self.df['s1_t1_gap'].abs()
        
        # ========= S-T能隙相关特征 =========
        if 's1_t1_gap' in self.df.columns and 'homo_lumo_gap' in self.df.columns:
            self.df['st_gap_ratio'] = np.where(
                self.df['homo_lumo_gap'] != 0,
                self.df['s1_t1_gap'] / self.df['homo_lumo_gap'],
                0
            )
        
        if 's1_energy_ev' in self.df.columns and 't1_energy_ev' in self.df.columns:
            self.df['st_average_energy'] = (self.df['s1_energy_ev'] + self.df['t1_energy_ev']) / 2
        
        # ========= 分子复杂度特征 =========
        if 'num_rings' in self.df.columns and 'num_heteroatoms' in self.df.columns:
            mol_weight = self.df.get('molecular_weight', self.df.get('mol_weight', 100))
            self.df['molecular_complexity'] = (
                self.df['num_rings'] * self.df['num_heteroatoms'] * np.log1p(mol_weight)
            )
        
        # 环系统复杂度
        if 'num_rings' in self.df.columns and 'avg_ring_size' in self.df.columns:
            self.df['ring_complexity'] = self.df['num_rings'] * self.df['avg_ring_size']
        
        # 小环应力指标
        small_ring_cols = ['num_3_member_rings', 'num_4_member_rings', 'num_5_member_rings']
        if all(col in self.df.columns for col in small_ring_cols):
            self.df['small_ring_strain'] = (
                self.df['num_3_member_rings'] * 3 +
                self.df['num_4_member_rings'] * 2 +
                self.df['num_5_member_rings'] * 1
            )
        
        # ========= 3D形状特征 =========
        if all(col in self.df.columns for col in ['gaussian_mol_length', 'gaussian_mol_width']):
            self.df['aspect_ratio'] = self.df['gaussian_mol_length'] / (self.df['gaussian_mol_width'] + 0.001)
            
        if all(col in self.df.columns for col in ['gaussian_mol_length', 'gaussian_mol_width', 'gaussian_mol_height']):
            self.df['shape_anisotropy'] = (self.df['gaussian_mol_length'] - self.df['gaussian_mol_width']) / \
                                         (self.df['gaussian_mol_length'] + self.df['gaussian_mol_width'] + 0.001)
        
        # ========= 激发态特征 =========
        if 'excited_homo' in self.df.columns and 'homo' in self.df.columns:
            self.df['homo_shift'] = self.df['excited_homo'] - self.df['homo']
        
        if 'excited_lumo' in self.df.columns and 'lumo' in self.df.columns:
            self.df['lumo_shift'] = self.df['excited_lumo'] - self.df['lumo']
        
        # ========= 反转gap特征 =========
        if 'num_inverted_gaps' in self.df.columns:
            self.df['has_inversion'] = (self.df['num_inverted_gaps'] > 0).astype(int)
        
        # ========= CREST特征 =========
        if 'crest_energy_range' in self.df.columns and 'crest_num_conformers' in self.df.columns:
            self.df['energy_per_conformer'] = self.df['crest_energy_range'] / (self.df['crest_num_conformers'] + 1)
        
        # ========= 氢键特征 =========
        if 'num_hbd' in self.df.columns and 'num_hba' in self.df.columns:
            self.df['h_bonding_capacity'] = self.df['num_hbd'] + self.df['num_hba']
            self.df['h_bond_balance'] = self.df['num_hbd'] / (self.df['num_hba'] + 1)
        
        # ========= Calicene特征交互 =========
        if 'CT_alignment_score' in self.df.columns and 'DA_strength' in self.df.columns:
            self.df['ct_da_interaction'] = self.df['CT_alignment_score'] * self.df['DA_strength']
        
        if 'DA_strength_5minus3' in self.df.columns and 'num_aromatic_rings' in self.df.columns:
            self.df['calicene_aromatic_interaction'] = self.df['DA_strength_5minus3'] * self.df['num_aromatic_rings']
        
        if 'favorable_for_inversion' in self.df.columns and 's1_t1_gap' in self.df.columns:
            self.df['inversion_gap_product'] = self.df['favorable_for_inversion'] * self.df['s1_t1_gap'].abs()
        
        # ========= 验证Calicene特征 =========
        print("\n=== Verifying Calicene features ===")
        calicene_features = [
            'has_3ring_nh2', 'has_3ring_cn', 'has_3ring_cf3', 'has_3ring_oh',
            'has_5ring_nh2', 'has_5ring_cn', 'has_5ring_cf3', 'has_5ring_oh',
            'count_3ring_nh2', 'count_3ring_cn', 'count_3ring_cf3',
            'count_5ring_nh2', 'count_5ring_cn', 'count_5ring_cf3'
        ]
        
        for feat in calicene_features:
            if feat in self.df.columns:
                non_zero = (self.df[feat] != 0).sum()
                unique = self.df[feat].nunique()
                print(f"  {feat}: {non_zero} non-zero values, {unique} unique values")
            else:
                print(f"  Warning: {feat} not found in data")
        
        return self.df
    
    def select_critical_features(self):
        """基于重要性选择关键特征 - 统一去重版本"""
        critical_features = [
            # === 最重要的能级特征 ===
            's1_t1_gap', 's1_t2_gap', 's1_t3_gap',
            's2_t1_gap', 's2_t2_gap',
            
            # === 基础电子特征 ===
            'homo', 'lumo', 'homo_lumo_gap',
            
            # === 分子结构特征 ===
            'num_atoms', 'num_bonds',
            'num_rings', 'num_aromatic_rings', 'num_saturated_rings',
            'num_aromatic_heterocycles',
            'num_3_member_rings', 'num_4_member_rings',
            'num_heteroatoms', 'num_N_atoms', 'num_O_atoms',
            'num_rotatable_bonds',
            'num_hbd', 'num_hba', 'tpsa',
            'planarity_ratio',
            
            # === 官能团存在性特征 ===
            'has_cyano', 'has_nitro', 'has_amino', 'has_carbonyl',
            'has_triphenylamine', 'has_carbazole', 'has_triazine',
            
            # === 官能团计数特征 ===
            'count_cyano', 'count_nitro', 'count_amino', 'count_carbonyl',
            'count_triphenylamine', 'count_carbazole', 'count_triazine',
            
            # === 官能团密度特征 ===
            'cyano_density', 'nitro_density', 'amino_density', 'carbonyl_density',
            
            # === D-A组合特征 ===
            'donor_score', 'acceptor_score',
            'D_A_ratio', 'D_A_product', 'is_D_A_molecule',
            'DA_strength', 'DA_balance', 'mol_type_code',
            
            # === 交互特征 ===
            'donor_homo_effect', 'acceptor_lumo_effect', 'DA_st_gap_effect',
            'st_gap_ratio', 'st_average_energy',
            'aromatic_gap_product',
            'molecular_complexity',
            'ring_complexity',
            'small_ring_strain',
            
            # === 3D形状特征 ===
            'gaussian_mol_volume',
            'gaussian_asphericity', 'gaussian_eccentricity',
            'aspect_ratio',
            
            # === CREST构象特征 ===
            'crest_min_rmsd', 'crest_std_rmsd', 'crest_avg_radius_gyration',
            'crest_num_conformers', 'crest_conformer_diversity',
            'energy_per_conformer',
            
            # === 氢键特征 ===
            'h_bonding_capacity', 'h_bond_balance',
            
            # === 反转gap特征 ===
            'num_inverted_gaps', 'primary_inversion_gap',
            
            # === Calicene体系特征 ===
            # 环存在与取代
            'has_3ring', 'has_5ring',
            'subs_on_3ring', 'subs_on_5ring',
            'num_in_subs_3ring', 'num_out_subs_3ring',
            'num_in_subs_5ring', 'num_out_subs_5ring',
            'num_both_subs', 'num_sp_subs',
            
            # 环上D/A分布
            'donor_on_3ring', 'donor_on_5ring',
            'acceptor_on_3ring', 'acceptor_on_5ring',
            
            # D/A强度与位置
            'DA_strength_5minus3',
            'DA_in_out_bias',
            'CT_alignment_score',
            'CT_position_weighted_score',
            'DA_asymmetry',
            
            # 反演可能性
            'favorable_for_inversion',
            
            # Calicene密度特征
            'D_density', 'A_density',
            'D_volume_density', 'A_volume_density',
            'in_sub_density', 'out_sub_density',
            
            # Push-pull模式one-hot
            'push_pull_pattern_none',
            'push_pull_pattern_D5_A3',
            'push_pull_pattern_D3_A5',
            'push_pull_pattern_D5_only',
            'push_pull_pattern_A3_only',
            'push_pull_pattern_DD_balanced',
            'push_pull_pattern_AA_balanced',
            
            # Ring polarity one-hot
            'ring_polarity_expected_aligned',
            'ring_polarity_expected_reversed',
            'ring_polarity_expected_neutral',
            
            # Calicene具体取代基
            'has_3ring_nh2', 'has_3ring_cn', 'has_3ring_cf3', 'has_3ring_oh',
            'has_5ring_nh2', 'has_5ring_cn', 'has_5ring_oh',
            'count_3ring_nh2', 'count_3ring_cn', 'count_3ring_cf3',
            'count_5ring_nh2', 'count_5ring_cn',
            
            # CT-ST交互
            'ct_st_gap_interaction', 'da_asymmetry_st_interaction',
        ]
        
        # 去重（确保没有重复）
        critical_features = list(dict.fromkeys(critical_features))
        
        # 只选择存在的特征
        available_features = [f for f in critical_features if f in self.df.columns]
        
        # 移除全零列（但保留重要的稀疏特征）
        important_sparse_features = [
            'count_3ring_nh2', 'count_3ring_cn', 'count_3ring_cf3',
            'count_5ring_nh2', 'count_5ring_cn',
            'has_3ring_nh2', 'has_3ring_cn', 'has_5ring_nh2', 'has_5ring_cn',
            'num_inverted_gaps', 'has_s1_t2_inversion'
        ]
        
        non_zero_features = []
        for feat in available_features:
            if feat in self.df.columns:
                if feat in important_sparse_features:
                    non_zero_features.append(feat)
                    print(f"  Keeping important sparse feature: {feat}")
                elif (self.df[feat] != 0).any():
                    non_zero_features.append(feat)
                else:
                    print(f"  Removing all-zero feature: {feat}")
        
        available_features = non_zero_features
        
        # 统计各类特征
        print(f"\nSelected {len(available_features)} critical features")
        print(f"Feature breakdown:")
        print(f"  Gap features: {len([f for f in available_features if 'gap' in f])}")
        print(f"  Structure: {len([f for f in available_features if 'num_' in f or 'ring' in f])}")
        print(f"  Functional groups: {len([f for f in available_features if 'has_' in f or 'count_' in f])}")
        print(f"  Density features: {len([f for f in available_features if 'density' in f])}")
        print(f"  Interaction features: {len([f for f in available_features if 'effect' in f or 'ratio' in f or 'product' in f])}")
        print(f"  Geometry: {len([f for f in available_features if 'gaussian' in f])}")
        print(f"  CREST: {len([f for f in available_features if 'crest' in f])}")
        print(f"  Calicene: {len([f for f in available_features if '3ring' in f or '5ring' in f])}")
        
        return available_features