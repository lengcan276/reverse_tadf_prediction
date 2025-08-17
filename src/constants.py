# src/constants.py
"""
全局常量定义
"""

# ============= 物理常量 =============
# TADF判定阈值
TADF_THRESHOLD = 0.3  # eV, S1-T1能隙阈值
OSCILLATOR_THRESHOLD = 0.01  # 振子强度阈值
INVERTED_GAP_THRESHOLD = 0.0  # eV, 反转能隙判定

# 能量转换
HARTREE_TO_EV = 27.2114  # Hartree到eV的转换系数
KCAL_TO_EV = 0.0433641  # kcal/mol到eV的转换

# ============= 模型参数 =============
# 随机种子
RANDOM_SEED = 42

# 数据分割
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# 深度学习参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
DROPOUT_RATE = 0.3

# 模型架构
HIDDEN_DIMS = [256, 128, 64]
TASK_HEAD_DIM = 32

# ============= XGBoost参数 =============
XGB_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

# ============= 特征工程 =============
# 关键特征列表
# 关键特征列表 - 更新为扩展版本
# 关键特征列表 - 与feature_engineering.py保持一致
KEY_FEATURES = [
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
    
    # === 官能团特征 ===
    'has_cyano', 'has_nitro', 'has_amino', 'has_carbonyl',
    'has_triphenylamine', 'has_carbazole', 'has_triazine',
    'count_cyano', 'count_nitro', 'count_amino', 'count_carbonyl',
    'count_triphenylamine', 'count_carbazole', 'count_triazine',
    
    # === 密度特征 ===
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
    
    # === CREST特征 ===
    'crest_min_rmsd', 'crest_std_rmsd', 'crest_avg_radius_gyration',
    'crest_num_conformers', 'crest_conformer_diversity',
    'energy_per_conformer',
    
    # === 氢键特征 ===
    'h_bonding_capacity', 'h_bond_balance',
    
    # === 反转gap特征 ===
    'num_inverted_gaps', 'primary_inversion_gap',
    
    # === Calicene体系完整特征 ===
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

# 排除的列
EXCLUDE_COLUMNS = [
    'Molecule', 'State', 'conformer', 'is_TADF', 
    'is_rTADF', 'inverted_gaps', 'singlet_states', 'triplet_states'
]

# ============= 文件路径 =============
# 数据路径
DATA_DIR = 'data'
RAW_DATA_FILE = 'all_conformers_data.csv'
PROCESSED_DATA_DIR = 'data/processed'

# 输出路径
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'outputs/models'
FIGURE_DIR = 'outputs/figures'
LOG_DIR = 'outputs/logs'
RESULT_DIR = 'outputs/results'

# 模型文件名
BEST_DL_MODEL = 'best_dl_model.pth'
BEST_XGB_MODEL = 'best_xgb_model.pkl'

# ============= 可视化 =============
# 图表参数
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)
COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# ============= 日志 =============
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'