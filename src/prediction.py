import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import joblib
import os
import re # 导入re模块，用于正则表达式匹配

def extract_features_from_smiles(smiles, molecule_name=None, quantum_features=None):
    """
    从SMILES提取特征
    
    Args:
        smiles: SMILES字符串
        molecule_name: 分子命名（可选），如 '5ring_nh2_3ring_cn_both'
        quantum_features: 字典，包含量子化学计算的特征（必需）
                         如 {'homo': -5.8, 'lumo': -2.1, 's1_t1_gap': 0.15, 
                             's1_t2_gap': 0.25, 's1_t3_gap': 0.35,...}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # 基础分子特征（从SMILES直接计算）
    features = {
        'molecular_weight': Descriptors.MolWt(mol),
        'num_heavy_atoms': Lipinski.HeavyAtomCount(mol),
        'num_heteroatoms': Lipinski.NumHeteroatoms(mol),
        'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
        'num_hbd': Lipinski.NumHDonors(mol),
        'num_hba': Lipinski.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
        'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': Descriptors.RingCount(mol),
        
        # 元素计数
        'num_N_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'),
        'num_O_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'),
        'num_S_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S'),
        'num_F_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'),
        
        # 小环特征
        'num_3_member_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) == 3),
        'num_4_member_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) == 4),
        'num_5_member_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) == 5),
        'num_6_member_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) == 6),
        
        # 杂环特征
        'num_aromatic_heterocycles': sum(1 for ring in mol.GetRingInfo().AtomRings() 
                                        if any(mol.GetAtomWithIdx(idx).GetSymbol()!= 'C' for idx in ring) 
                                        and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)),
    }
    
    # 官能团检测和计数
    features.update({
        'has_cyano': int('[C-]#N' in smiles or 'C#N' in smiles),
        'has_nitro': int('[N+](=O)[O-]' in smiles or 'N(=O)=O' in smiles),
        'has_amino': int(any(atom.GetSymbol() == 'N' and atom.GetDegree() <= 3 
                            and atom.GetTotalNumHs() > 0 for atom in mol.GetAtoms())),
        'has_carbonyl': int('C=O' in smiles),
        'has_sulfone': int('S(=O)(=O)' in smiles or 'S(=O)=O' in smiles),
        'has_triazine': int('c1ncncn1' in smiles or 'C1=NC=NC=N1' in smiles),
        'has_boron': int('B' in smiles),
        'has_phosphorus': int('P' in smiles),
        'has_triphenylamine': int('N(c1ccccc1)(c2ccccc2)c3ccccc3' in smiles),
        'has_carbazole': int('c1ccc2c(c1)[nH]c1ccccc12' in smiles),
        
        'count_cyano': smiles.count('C#N'),
        'count_nitro': smiles.count('[N+](=O)[O-]') + smiles.count('N(=O)=O'),
        'count_amino': sum(1 for atom in mol.GetAtoms() 
                          if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0),
        'count_carbonyl': smiles.count('C=O'),
        'count_triphenylamine': 1 if 'N(c1ccccc1)(c2ccccc2)c3ccccc3' in smiles else 0,
        'count_carbazole': 1 if 'c1ccc2c(c1)[nH]c1ccccc12' in smiles else 0,
        'count_triazine': 1 if 'c1ncncn1' in smiles or 'C1=NC=NC=N1' in smiles else 0,
    })
    
    # ========= 处理量子化学特征 =========
    if quantum_features is None:
        raise ValueError(
            "Quantum chemical features are required for accurate prediction.\n"
            "Please provide a dictionary with these keys:\n"
            "- Energy levels: homo, lumo, homo_lumo_gap\n"
            "- Excitation energies: s1_energy_ev, t1_energy_ev, s1_oscillator\n"
            "- Energy gaps: s1_t1_gap, s1_t2_gap, s1_t3_gap, s2_t1_gap, s2_t2_gap\n"
            "These values should be calculated using quantum chemistry software (Gaussian, ORCA, etc.)"
        )
    
    # 使用提供的量子化学特征
    features.update(quantum_features)
    
    # 验证必要的量子化学特征
    required_quantum = ['homo', 'lumo', 's1_t1_gap']
    missing = [k for k in required_quantum if k not in quantum_features]
    if missing:
        raise ValueError(f"Missing required quantum features: {missing}")
    
    # 计算衍生的量子化学特征
    if 'homo_lumo_gap' not in quantum_features:
        features['homo_lumo_gap'] = quantum_features.get('lumo', 0) - quantum_features.get('homo', 0)
    
    if 'planarity_ratio' not in quantum_features:
        features['planarity_ratio'] = 0.9  # 默认值
    
    features['num_inverted_gaps'] = 1 if quantum_features.get('s1_t1_gap', 0) < 0 else 0
    features['primary_inversion_gap'] = quantum_features.get('s1_t1_gap', 0) if quantum_features.get('s1_t1_gap', 0) < 0 else 0
    
    # ========= 处理Calicene特征 =========
    calicene_features = {}
    
    if molecule_name:
        # 从命名中提取Calicene特征
        name_lower = molecule_name.lower()
        
        # 检测环类型
        calicene_features['has_3ring'] = int('3ring' in name_lower)
        calicene_features['has_5ring'] = int('5ring' in name_lower)
        
        # 检测取代基位置
        calicene_features['num_in_subs_3ring'] = name_lower.count('3ring_') * int('_in' in name_lower)
        calicene_features['num_out_subs_3ring'] = name_lower.count('3ring_') * int('_out' in name_lower)
        calicene_features['num_in_subs_5ring'] = name_lower.count('5ring_') * int('_in' in name_lower)
        calicene_features['num_out_subs_5ring'] = name_lower.count('5ring_') * int('_out' in name_lower)
        calicene_features['num_both_subs'] = int('_both' in name_lower)
        calicene_features['num_sp_subs'] = int('_sp' in name_lower)
        
        # 检测具体取代基
        calicene_features['has_3ring_nh2'] = int('3ring_nh2' in name_lower)
        calicene_features['has_3ring_cn'] = int('3ring_cn' in name_lower)
        calicene_features['has_3ring_cf3'] = int('3ring_cf3' in name_lower)
        calicene_features['has_3ring_oh'] = int('3ring_oh' in name_lower)
        calicene_features['has_5ring_nh2'] = int('5ring_nh2' in name_lower)
        calicene_features['has_5ring_cn'] = int('5ring_cn' in name_lower or '5ring_nme2' in name_lower)
        calicene_features['has_5ring_oh'] = int('5ring_oh' in name_lower)
        
        calicene_features['count_3ring_nh2'] = name_lower.count('3ring_nh2')
        calicene_features['count_3ring_cn'] = name_lower.count('3ring_cn')
        calicene_features['count_3ring_cf3'] = name_lower.count('3ring_cf3')
        calicene_features['count_5ring_nh2'] = name_lower.count('5ring_nh2')
        calicene_features['count_5ring_cn'] = name_lower.count('5ring_cn') + name_lower.count('5ring_nme2')
        
        # 计算取代基数量
        subs_on_3ring = sum([calicene_features[f'count_3ring_{sub}'] 
                            for sub in ['nh2', 'cn', 'cf3'] 
                            if f'count_3ring_{sub}' in calicene_features])
        subs_on_5ring = sum([calicene_features[f'count_5ring_{sub}'] 
                            for sub in ['nh2', 'cn'] 
                            if f'count_5ring_{sub}' in calicene_features])
        
        calicene_features['subs_on_3ring'] = subs_on_3ring
        calicene_features['subs_on_5ring'] = subs_on_5ring
        
        # 判断D/A分布
        donor_3ring = calicene_features['has_3ring_nh2']
        donor_5ring = calicene_features['has_5ring_nh2'] or int('5ring_nph3' in name_lower)
        acceptor_3ring = calicene_features['has_3ring_cn'] + calicene_features['has_3ring_cf3']
        acceptor_5ring = calicene_features['has_5ring_cn']
        
        calicene_features['donor_on_3ring'] = donor_3ring
        calicene_features['donor_on_5ring'] = donor_5ring
        calicene_features['acceptor_on_3ring'] = acceptor_3ring
        calicene_features['acceptor_on_5ring'] = acceptor_5ring
        
        # 计算D/A相关特征
        calicene_features = (donor_5ring - donor_3ring) * 2 + (acceptor_5ring - acceptor_3ring)
        calicene_features = calicene_features['num_in_subs_3ring'] - calicene_features['num_out_subs_3ring']
        
    else:
        # 没有命名信息时，使用默认值
        calicene_defaults = {
            'has_3ring': 0, 'has_5ring': 0,
            'subs_on_3ring': 0, 'subs_on_5ring': 0,
            'num_in_subs_3ring': 0, 'num_out_subs_3ring': 0,
            'num_in_subs_5ring': 0, 'num_out_subs_5ring': 0,
            'num_both_subs': 0, 'num_sp_subs': 0,
            'donor_on_3ring': 0, 'donor_on_5ring': 0,
            'acceptor_on_3ring': 0, 'acceptor_on_5ring': 0,
            'DA_strength_5minus3': 0, 'DA_in_out_bias': 0,
            'has_3ring_nh2': 0, 'has_3ring_cn': 0, 'has_3ring_cf3': 0, 'has_3ring_oh': 0,
            'has_5ring_nh2': 0, 'has_5ring_cn': 0, 'has_5ring_oh': 0,
            'count_3ring_nh2': 0, 'count_3ring_cn': 0, 'count_3ring_cf3': 0,
            'count_5ring_nh2': 0, 'count_5ring_cn': 0,
        }
        calicene_features = calicene_defaults
    
    features.update(calicene_features)
    
    # ========= 计算D-A特征 =========
    donor_score = (
        features.get('has_amino', 0) * 2 + 
        features.get('has_triphenylamine', 0) * 3 +
        features.get('count_carbazole', 0) * 2.5 +
        features.get('donor_on_5ring', 0) * 2 +
        features.get('donor_on_3ring', 0) * 1.5
    )
    acceptor_score = (
        features.get('has_cyano', 0) * 3 + 
        features.get('has_nitro', 0) * 2 + 
        features.get('has_carbonyl', 0) * 1 +
        features.get('has_triazine', 0) * 2 +
        features.get('acceptor_on_5ring', 0) * 2 +
        features.get('acceptor_on_3ring', 0) * 1.5
    )
    
    features.update({
        'donor_score': donor_score,
        'acceptor_score': acceptor_score,
        'D_A_ratio': donor_score / (acceptor_score + 1),
        'D_A_product': donor_score * acceptor_score,
        'is_D_A_molecule': int(donor_score > 0 and acceptor_score > 0),
        'DA_strength': donor_score * acceptor_score,
        'DA_balance': abs(donor_score - acceptor_score),
        'mol_type_code': 3 if (donor_score > 2 and acceptor_score > 2) else 
                         1 if (donor_score > 2) else 
                         2 if (acceptor_score > 2) else 0,
    })
    
    # ========= 密度特征 =========
    num_heavy = max(features['num_heavy_atoms'], 1)
    features.update({
        'cyano_density': features.get('count_cyano', 0) / num_heavy,
        'nitro_density': features.get('count_nitro', 0) / num_heavy,
        'amino_density': features.get('count_amino', 0) / num_heavy,
        'carbonyl_density': features.get('count_carbonyl', 0) / num_heavy,
        'D_density': donor_score / num_heavy,
        'A_density': acceptor_score / num_heavy,
    })
    
    # ========= 交互特征（使用真实的量子化学值）=========
    features.update({
        'aromatic_gap_product': features['num_aromatic_rings'] * abs(quantum_features['s1_t1_gap']),
        'donor_homo_effect': donor_score * quantum_features['homo'],
        'acceptor_lumo_effect': acceptor_score * quantum_features['lumo'],
        'DA_st_gap_effect': features * abs(quantum_features['s1_t1_gap']),
        'st_gap_ratio': quantum_features['s1_t1_gap'] / max(features['homo_lumo_gap'], 0.1),
        'st_average_energy': (quantum_features.get('s1_energy_ev', 3.0) + 
                             quantum_features.get('t1_energy_ev', 2.8)) / 2,
    })
    
    # ========= 其他复杂特征 =========
    features.update({
        'molecular_complexity': (features['num_rings'] * features['num_heteroatoms'] * 
                                np.log1p(features['molecular_weight'])),
        'ring_complexity': features['num_rings'] * 6,
        'small_ring_strain': (features['num_3_member_rings'] * 3 +
                              features['num_4_member_rings'] * 2 +
                              features['num_5_member_rings'] * 1),
        'h_bonding_capacity': features['num_hbd'] + features['num_hba'],
        'h_bond_balance': features['num_hbd'] / max(features['num_hba'], 1),
    })
    
    # ========= 3D和CREST特征（需要提供或使用默认值）=========
    # 这些特征理想情况下应该从CREST计算得到
    features.update({
        'gaussian_mol_volume': quantum_features.get('gaussian_mol_volume', 500),
        'gaussian_asphericity': quantum_features.get('gaussian_asphericity', 0.1),
        'gaussian_eccentricity': quantum_features.get('gaussian_eccentricity', 0.2),
        'aspect_ratio': quantum_features.get('aspect_ratio', 1.5),
        'crest_min_rmsd': quantum_features.get('crest_min_rmsd', 0.5),
        'crest_std_rmsd': quantum_features.get('crest_std_rmsd', 0.1),
        'crest_avg_radius_gyration': quantum_features.get('crest_avg_radius_gyration', 5.0),
        'crest_num_conformers': quantum_features.get('crest_num_conformers', 1),
        'crest_conformer_diversity': quantum_features.get('crest_conformer_diversity', 0.1),
        'energy_per_conformer': quantum_features.get('energy_per_conformer', 0.1),
    })
    
    # ========= 其他Calicene相关特征 =========
    features.update({
        'CT_alignment_score': calicene_features.get('DA_strength_5minus3', 0) * 0.1,
        'CT_position_weighted_score': 0,
        'DA_asymmetry': abs(calicene_features.get('DA_strength_5minus3', 0)),
        'favorable_for_inversion': 1 if quantum_features['s1_t1_gap'] < 0.1 else 0,
        'D_volume_density': donor_score / max(features['gaussian_mol_volume'], 1),
        'A_volume_density': acceptor_score / max(features['gaussian_mol_volume'], 1),
        'in_sub_density': calicene_features.get('num_in_subs_3ring', 0) / num_heavy,
        'out_sub_density': calicene_features.get('num_out_subs_3ring', 0) / num_heavy,
        
        # One-hot编码
        'push_pull_pattern_none': 0 if features else 1,
        'push_pull_pattern_D5_A3': int(calicene_features.get('donor_on_5ring', 0) > 0 and 
                                       calicene_features.get('acceptor_on_3ring', 0) > 0),
        'push_pull_pattern_D3_A5': int(calicene_features.get('donor_on_3ring', 0) > 0 and 
                                       calicene_features.get('acceptor_on_5ring', 0) > 0),
        'push_pull_pattern_D5_only': 0,
        'push_pull_pattern_A3_only': 0,
        'push_pull_pattern_DD_balanced': 0,
        'push_pull_pattern_AA_balanced': 0,
        
        'ring_polarity_expected_aligned': 0,
        'ring_polarity_expected_reversed': 0,
        'ring_polarity_expected_neutral': 1,
        
        'ct_st_gap_interaction': 0,
        'da_asymmetry_st_interaction': 0,
    })
    
    return features


def predict_new_molecule(smiles, quantum_features, molecule_name=None, model_path='best_dl_model.pth', 
                         xgb_model_path='xgboost_tadf_predictor.pkl',
                         scaler_path='scaler.pkl', use_xgboost=True):
    """
    预测新分子的TADF/rTADF性质
    
    Args:
        smiles: SMILES字符串
        quantum_features: 字典，包含量子化学计算的特征（必需）。
                          如 {'homo': -5.8, 'lumo': -2.1, 's1_t1_gap': 0.15,...}
        model_path: 深度学习模型路径
        xgb_model_path: XGBoost模型路径
        scaler_path: 特征标准化器路径
        use_xgboost: 是否使用XGBoost（True）还是深度学习模型
    """
    
    if quantum_features is None:
        raise ValueError(
            "Quantum features are required for prediction.\n"
            "Please provide: homo, lumo, s1_t1_gap, s1_t2_gap, etc."
        )
    
    # 提取特征 - 使用修改后的函数
    features_dict = extract_features_from_smiles(
        smiles, 
        molecule_name=molecule_name,
        quantum_features=quantum_features
    )
    
    # 加载已保存的特征列表和标准化器
    try:
        # 读取训练时使用的特征列表
        with open('data/splits/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
    except FileNotFoundError:
        # 如果没有保存的特征列表，使用从当前提取的特征字典中获取的键作为特征名
        print("Warning: 'data/splits/feature_names.txt' not found. Using keys from extracted features as feature names.")
        feature_names = list(features_dict.keys())
    
    # 创建特征向量（按照训练时的顺序）
    features_list =
    for feat_name in feature_names:
        if feat_name in features_dict:
            features_list.append(features_dict[feat_name])
        else:
            # 缺失的特征用0填充，并发出警告
            # print(f"Warning: Feature '{feat_name}' missing from extracted features. Filling with 0.")
            features_list.append(0)
    
    features = np.array(features_list).reshape(1, -1)
    
    # 标准化特征（如果有保存的scaler）
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)
    
    if use_xgboost and os.path.exists(xgb_model_path):
        # 使用XGBoost模型
        xgb_model = joblib.load(xgb_model_path)
        # XGBoost模型可能直接返回概率，或者需要sigmoid转换
        # 假设模型返回的是原始分数，需要转换为概率
        raw_predictions = xgb_model.predict(features)
        tadf_prob = 1 / (1 + np.exp(-raw_predictions)) # 假设第一个输出是TADF
        rtadf_prob = 1 / (1 + np.exp(-raw_predictions)) # 假设第二个输出是rTADF
        
        result = {
            'TADF_probability': float(tadf_prob),
            'rTADF_probability': float(rtadf_prob),
            'is_TADF': tadf_prob > 0.5,
            'is_rTADF': rtadf_prob > 0.5,
            'model_used': 'XGBoost'
        }
    else:
        # 使用深度学习模型
        # 确保 MultiTaskTADFNet 在当前环境中可用
        try:
            from.models import MultiTaskTADFNet
        except ImportError:
            # 如果作为独立脚本运行，可能需要调整导入路径
            print("Warning: Could not import MultiTaskTADFNet from.models. Assuming it's defined elsewhere or not used.")
            # 这里的处理取决于 MultiTaskTADFNet 的实际定义位置
            # 为了让示例运行，这里可以添加一个简单的占位符类，或者要求用户确保模型类可用
            class MultiTaskTADFNet(torch.nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_dim, 10)
                    self.fc_tadf = torch.nn.Linear(10, 1)
                    self.fc_rtadf = torch.nn.Linear(10, 1)
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    tadf = torch.sigmoid(self.fc_tadf(x))
                    rtadf = torch.sigmoid(self.fc_rtadf(x))
                    return tadf, rtadf

        
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 创建模型实例
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model = MultiTaskTADFNet(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设是直接的state_dict，需要知道input_dim
            # 这里的 input_dim 应该与训练时模型的输入维度一致
            # 实际应用中，model_config 应该被保存和加载
            print("Warning: Model config not found in checkpoint. Assuming input_dim from features.")
            model = MultiTaskTADFNet(input_dim=features.shape[1])
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            tadf_prob, rtadf_prob = model(features_tensor)
        
        result = {
            'TADF_probability': float(tadf_prob.item()),
            'rTADF_probability': float(rtadf_prob.item()),
            'is_TADF': tadf_prob.item() > 0.5,
            'is_rTADF': rtadf_prob.item() > 0.5,
            'model_used': 'Deep Learning'
        }
    
    # 添加置信度评估
    result['confidence'] = 'high' if (
        (result > 0.8 or result < 0.2) and
        (result > 0.8 or result < 0.2)
    ) else 'medium' if (
        (result > 0.7 or result < 0.3) and
        (result > 0.7 or result < 0.3)
    ) else 'low'
    
    return result

def batch_predict(smiles_list, quantum_features_list, **kwargs):
    """
    批量预测多个分子的TADF/rTADF性质
    
    Args:
        smiles_list: SMILES字符串列表
        quantum_features_list: 包含量子化学特征字典的列表，与 smiles_list 一一对应。
                               每个字典如 {'homo': -5.8, 'lumo': -2.1, 's1_t1_gap': 0.15,...}
        **kwargs: 传递给 predict_new_molecule 的额外参数（如 model_path, use_xgboost 等）
    """
    results =
    if len(smiles_list)!= len(quantum_features_list):
        raise ValueError("smiles_list and quantum_features_list must have the same length.")

    for i, smiles in enumerate(smiles_list):
        try:
            # 调用 predict_new_molecule 时传入对应的 quantum_features
            result = predict_new_molecule(smiles, quantum_features_list[i], **kwargs)
            result['smiles'] = smiles
            results.append(result)
        except Exception as e:
            results.append({
                'smiles': smiles,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# 使用示例
if __name__ == "__main__":
    # 示例量子化学特征数据
    # 在实际应用中，这些数据应通过量子化学计算获得
    sample_quantum_features_single = {
        'homo': -5.5, 'lumo': -2.0, 's1_t1_gap': 0.1,
        's1_energy_ev': 3.0, 't1_energy_ev': 2.9, 's1_oscillator': 0.5,
        's1_t2_gap': 0.2, 's1_t3_gap': 0.3, 's2_t1_gap': 0.4, 's2_t2_gap': 0.5,
        'gaussian_mol_volume': 600, 'gaussian_asphericity': 0.15,
        'gaussian_eccentricity': 0.25, 'aspect_ratio': 1.6,
        'crest_min_rmsd': 0.4, 'crest_std_rmsd': 0.08,
        'crest_avg_radius_gyration': 5.2, 'crest_num_conformers': 2,
        'crest_conformer_diversity': 0.12, 'energy_per_conformer': 0.08,
        'planarity_ratio': 0.85 # 确保所有可能被 extract_features_from_smiles 访问的键都存在
    }

    sample_quantum_features_rtadf = {
        'homo': -5.0, 'lumo': -2.5, 's1_t1_gap': -0.05, # 负能隙表示rTADF
        's1_energy_ev': 2.8, 't1_energy_ev': 2.85, 's1_oscillator': 0.6,
        's1_t2_gap': 0.1, 's1_t3_gap': 0.2, 's2_t1_gap': 0.3, 's2_t2_gap': 0.4,
        'gaussian_mol_volume': 550, 'gaussian_asphericity': 0.12,
        'gaussian_eccentricity': 0.22, 'aspect_ratio': 1.4,
        'crest_min_rmsd': 0.3, 'crest_std_rmsd': 0.05,
        'crest_avg_radius_gyration': 4.8, 'crest_num_conformers': 3,
        'crest_conformer_diversity': 0.15, 'energy_per_conformer': 0.05,
        'planarity_ratio': 0.92
    }
    
    # 测试单个分子
    test_smiles_single = "c1ccc(N(c2ccccc2)c2ccccc2)cc1"  # 三苯胺
    
    print("--- 测试单个分子预测 ---")
    try:
        # 调用 predict_new_molecule 时传入 quantum_features
        result_single = predict_new_molecule(test_smiles_single, sample_quantum_features_single, use_xgboost=False) # 假设使用DL模型
        print(f"Prediction for {test_smiles_single}:")
        print(f"  TADF probability: {result_single:.3f}")
        print(f"  rTADF probability: {result_single:.3f}")
        print(f"  Is TADF: {result_single}")
        print(f"  Is rTADF: {result_single}")
        print(f"  Confidence: {result_single['confidence']}")
    except Exception as e:
        print(f"Error predicting single molecule: {e}")

    print("\n--- 测试批量分子预测 ---")
    # 批量测试
    test_smiles_list = [
        "c1ccc(N(c2ccccc2)c2ccccc2)cc1", # 三苯胺
        "N#Cc1ccc(-c2ccc(N)cc2)cc1", # 氰基-苯-苯-胺
        "O=C(C)c1ccc(N(C)C)cc1" # 乙酰基-苯-二甲胺
    ]
    
    # 为每个SMILES提供对应的量子化学特征
    test_quantum_features_list =

    try:
        # 调用 batch_predict 时传入 smiles_list 和 quantum_features_list
        batch_results = batch_predict(test_smiles_list, test_quantum_features_list, use_xgboost=False) # 假设使用DL模型
        print("Batch prediction results:")
        print(batch_results])
    except Exception as e:
        print(f"Error predicting batch molecules: {e}")