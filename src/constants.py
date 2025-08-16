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
KEY_FEATURES = [
    # 电子结构
    'homo', 'lumo', 'homo_lumo_gap', 'dipole',
    'excited_homo', 'excited_lumo', 'excited_homo_lumo_gap', 'excited_dipole',
    's1_energy_ev', 's1_oscillator', 's1_wavelength_nm',
    't1_energy_ev', 't1_wavelength_nm',
    
    # S-T能隙
    's1_t1_gap', 's1_t2_gap', 's1_t3_gap', 's1_t4_gap', 's1_t5_gap',
    's2_t1_gap', 's2_t2_gap', 's2_t3_gap', 's2_t4_gap', 's2_t5_gap',
    's3_t1_gap', 's3_t2_gap', 's3_t3_gap', 's3_t4_gap', 's3_t5_gap',
    
    # 分子结构
    'num_rings', 'num_aromatic_rings', 'num_saturated_rings',
    'num_conjugated_bonds', 'max_conjugated_system_size',
    'planarity_ratio', 'torsion_std',
    'num_heteroatoms', 'num_heavy_atoms',
    
    # 3D特征
    'gaussian_mol_volume', 'gaussian_radius_of_gyration',
    'gaussian_asphericity', 'gaussian_eccentricity',
    
    # 反转能隙
    'num_inverted_gaps', 'primary_inversion_gap',
    
    # CREST特征
    'crest_num_conformers', 'crest_energy_range',
    # 新增高层次特征
    'donor_score', 'acceptor_score', 'D_A_ratio', 'D_A_product', 'is_D_A_molecule',
    'DA_strength', 'DA_balance', 'mol_type_code',
    'cyano_density', 'nitro_density', 'amino_density',
    'aromatic_gap_product',
    'ring_complexity', 'small_ring_strain',
    'h_bonding_capacity', 'h_bond_balance',
    'donor_homo_effect', 'acceptor_lumo_effect', 'DA_st_gap_effect',
    'st_gap_ratio', 'st_average_energy', 'molecular_complexity',
    'aspect_ratio', 'energy_per_conformer'
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