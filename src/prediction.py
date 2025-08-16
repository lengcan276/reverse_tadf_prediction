import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import joblib
import os

def extract_features_from_smiles(smiles):
    """从SMILES提取特征"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # 基础分子特征
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
        
        # 官能团检测
        'has_cyano': int('[C-]#N' in smiles or 'C#N' in smiles),
        'has_nitro': int('[N+](=O)[O-]' in smiles),
        'has_amino': int('N' in smiles and 'N(' in smiles),
        'has_carbonyl': int('C=O' in smiles),
        'has_sulfone': int('S(=O)(=O)' in smiles),
        'has_triazine': int('c1ncncn1' in smiles),
        'has_boron': int('B' in smiles),
        'has_phosphorus': int('P' in smiles),
        'has_triphenylamine': int('N(c1ccccc1)(c2ccccc2)c3ccccc3' in smiles),
    }
    
    # 注意：这里只能提取RDKit可以计算的特征
    # 能级特征（HOMO/LUMO等）需要量子化学计算，这里用占位值
    features.update({
        'homo': -6.0,  # 需要实际计算
        'lumo': -2.0,  # 需要实际计算
        'homo_lumo_gap': 4.0,  # 需要实际计算
        's1_energy_ev': 3.0,  # 需要实际计算
        's1_oscillator': 0.1,  # 需要实际计算
        't1_energy_ev': 2.8,  # 需要实际计算
        's1_t1_gap': 0.2,  # 需要实际计算
        # ... 添加其他必要特征的占位值
    })
    # ============ 在这里添加高层次特征计算 ============
    # 官能团组合特征
    donor_score = (
        features.get('has_amino', 0) * 2 + 
        features.get('has_triphenylamine', 0) * 3
    )
    acceptor_score = (
        features.get('has_cyano', 0) * 3 + 
        features.get('has_nitro', 0) * 2 + 
        features.get('has_carbonyl', 0) * 1
    )
    
    features.update({
        'donor_score': donor_score,
        'acceptor_score': acceptor_score,
        'D_A_ratio': donor_score / (acceptor_score + 1),
        'D_A_product': donor_score * acceptor_score,
        'is_D_A_molecule': int(donor_score > 0 and acceptor_score > 0),
        
        # 高层次特征
        'DA_strength': donor_score * acceptor_score,
        'DA_balance': abs(donor_score - acceptor_score),
        'mol_type_code': 3 if (donor_score > 2 and acceptor_score > 2) else 
                         1 if (donor_score > 2) else 
                         2 if (acceptor_score > 2) else 0,
        
        # 需要添加更多占位值
        's1_t2_gap': 0.3,
        's1_t3_gap': 0.4,
        's2_t1_gap': 0.5,
        's2_t2_gap': 0.6,
        'st_gap_ratio': 0.05,
        'st_average_energy': 2.9,
        'aspect_ratio': 1.5,
        'gaussian_mol_volume': 500,
        'gaussian_asphericity': 0.1,
        'gaussian_eccentricity': 0.2,
        'crest_min_rmsd': 0.5,
        'crest_std_rmsd': 0.1,
        'crest_avg_radius_gyration': 5.0,
        'num_inverted_gaps': 0,
        'primary_inversion_gap': 0,
        'planarity_ratio': 0.9,
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': Descriptors.RingCount(mol),
        'num_N_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'),
        'num_O_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'),
    })
    # ============ 高层次特征计算结束 ============
    
   
    
    return features

def predict_new_molecule(smiles, model_path='best_dl_model.pth', 
                         xgb_model_path='xgboost_tadf_predictor.pkl',
                         scaler_path='scaler.pkl', use_xgboost=True):
    """
    预测新分子的TADF/rTADF性质
    
    Args:
        smiles: SMILES字符串
        model_path: 深度学习模型路径
        xgb_model_path: XGBoost模型路径
        scaler_path: 特征标准化器路径
        use_xgboost: 是否使用XGBoost（True）还是深度学习模型
    """
    
    # 提取特征
    features_dict = extract_features_from_smiles(smiles)
    
    # 加载已保存的特征列表和标准化器
    try:
        # 读取训练时使用的特征列表
        with open('data/splits/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
    except:
        # 如果没有保存的特征列表，使用默认的
        print("Warning: Using default feature list")
        feature_names = list(features_dict.keys())
    
    # 创建特征向量（按照训练时的顺序）
    features_list = []
    for feat_name in feature_names:
        if feat_name in features_dict:
            features_list.append(features_dict[feat_name])
        else:
            # 缺失的特征用0填充
            features_list.append(0)
    
    features = np.array(features_list).reshape(1, -1)
    
    # 标准化特征（如果有保存的scaler）
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)
    
    if use_xgboost and os.path.exists(xgb_model_path):
        # 使用XGBoost模型
        xgb_model = joblib.load(xgb_model_path)
        tadf_prob, rtadf_prob = xgb_model.predict(features)
        
        result = {
            'TADF_probability': float(tadf_prob[0]),
            'rTADF_probability': float(rtadf_prob[0]),
            'is_TADF': tadf_prob[0] > 0.5,
            'is_rTADF': rtadf_prob[0] > 0.5,
            'model_used': 'XGBoost'
        }
    else:
        # 使用深度学习模型
        from .models import MultiTaskTADFNet
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 创建模型实例
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model = MultiTaskTADFNet(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设是直接的state_dict
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
        (result['TADF_probability'] > 0.8 or result['TADF_probability'] < 0.2) and
        (result['rTADF_probability'] > 0.8 or result['rTADF_probability'] < 0.2)
    ) else 'medium' if (
        (result['TADF_probability'] > 0.7 or result['TADF_probability'] < 0.3) and
        (result['rTADF_probability'] > 0.7 or result['rTADF_probability'] < 0.3)
    ) else 'low'
    
    return result

def batch_predict(smiles_list, **kwargs):
    """批量预测多个分子"""
    results = []
    for smiles in smiles_list:
        try:
            result = predict_new_molecule(smiles, **kwargs)
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
    # 测试单个分子
    test_smiles = "c1ccc(N(c2ccccc2)c2ccccc2)cc1"  # 三苯胺
    
    try:
        result = predict_new_molecule(test_smiles)
        print(f"Prediction for {test_smiles}:")
        print(f"  TADF probability: {result['TADF_probability']:.3f}")
        print(f"  rTADF probability: {result['rTADF_probability']:.3f}")
        print(f"  Is TADF: {result['is_TADF']}")
        print(f"  Is rTADF: {result['is_rTADF']}")
        print(f"  Confidence: {result['confidence']}")
    except Exception as e:
        print(f"Error: {e}")