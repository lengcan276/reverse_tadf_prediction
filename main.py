# main.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XGBOOST_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import random
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from xgboost import XGBClassifier
import joblib

# ----------------- Project modules -----------------
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import MultiTaskTADFNet, XGBoostTADFPredictor
from src.training import ModelTrainer
from src.interpretation import ModelInterpreter

# 关键：使用“设计安全特征”并保证训练/OOF/测试一致
from src.robustness_test import get_design_safe_features

# 统一的 OOF-stacking（已在你项目中封装）
from src.ensemble import (
    run_design_oof_stacking_pipeline,
    apply_stacking_to_test,
)

# ----------------- Global setup -----------------
os.makedirs('outputs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/splits', exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ========= 辅助函数：统一且自洽的标签与间隙分布汇总 =========
def summarize_labels(df, strict=True):
    """
    基于当前 df 的 is_TADF / is_rTADF 和 gap 列做自洽汇总。
    仅报告我们“实际用于训练的那一套标签”的统计，避免口径不一致。
    """
    print("\n=== Creating labels summary (coherent with training labels) ===")
    label_tag = "PRIMARY (Strict)" if strict else "RELAXED (Sensitivity)"
    print(f"Using label set: {label_tag}")

    # 直接用 is_TADF / is_rTADF 计数（训练就是用这两列）
    if 'is_TADF' in df.columns and 'is_rTADF' in df.columns:
        n = len(df)
        t_pos = int(df['is_TADF'].sum())
        r_pos = int(df['is_rTADF'].sum())
        print(f"TADF: {t_pos}/{n} ({t_pos / max(n,1) * 100:.1f}%)")
        print(f"rTADF: {r_pos}/{n} ({r_pos / max(n,1) * 100:.1f}%)")

    # ΔE_ST（S1-T1 gap）分布（如果存在）
    gap_col_candidates = ['s1_t1_gap', 'delta_e_st', 'DE_ST', 'DeltaE_ST']
    gap_col = None
    for c in gap_col_candidates:
        if c in df.columns:
            gap_col = c
            break

    if gap_col is not None:
        g = df[gap_col].astype(float).dropna().values
        if len(g) > 0:
            print(f"\n=== S1-T1 Gap Distribution (from `{gap_col}`) ===")
            bins = [(-1e9, -0.02), (-0.02, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 1e9)]
            names = ["Negative (<-0.02)", "Near zero (-0.02~0.02)", "Small (+0.02~0.10)",
                     "Moderate (0.10~0.30)", "Large (>0.30)"]
            for (lo, hi), nm in zip(bins, names):
                cnt = int(((g > lo) & (g <= hi)).sum())
                print(f"  {nm}: {cnt}")
            print("\nGap statistics:")
            print(f"  Min: {np.min(g):.3f} eV")
            print(f"  Max: {np.max(g):.3f} eV")
            print(f"  Mean: {np.mean(g):.3f} eV")
            print(f"  Median: {np.median(g):.3f} eV")
    else:
        print("\nWarning: No S1-T1 gap column found; skipped gap distribution.")


def main():
    """主程序：已对齐‘可投稿版’的关键要求"""

    # ----------------- Step 1: 数据预处理 -----------------
    print("Step 1: Data Preprocessing...")
    preprocessor = DataPreprocessor('./data/all_conformers_data.csv')
    preprocessor.clean_data().extract_features_from_smiles().handle_missing_values().create_labels()

    # 按严格口径训练（建议作为主结果）
    USE_STRICT_LABELS = True
    if USE_STRICT_LABELS:
        print("\n📊 Using STRICT literature-based labels for training (PRIMARY)")
        # is_TADF / is_rTADF 已由 create_labels 写入
    else:
        print("\n📊 Using RELAXED labels for sensitivity analysis")
        # 使用 relaxed 覆盖
        for col_src, col_dst in [('is_TADF_relaxed', 'is_TADF'),
                                 ('is_rTADF_relaxed', 'is_rTADF')]:
            if col_src in preprocessor.df.columns:
                preprocessor.df[col_dst] = preprocessor.df[col_src]
            else:
                raise ValueError(f"Relaxed label column `{col_src}` not found!")

    # 统一、可复现的标签与间隙汇总
    summarize_labels(preprocessor.df, strict=USE_STRICT_LABELS)

    # ----------------- Step 1.5: 分子级统计 -----------------
    print("\nStep 1.5: Molecular Level Statistics...")
    if 'Molecule' in preprocessor.df.columns:
        unique_molecules = preprocessor.df['Molecule'].nunique()
        total_conformers = len(preprocessor.df)
        avg_conf = total_conformers / max(unique_molecules, 1)
        print(f"Data structure: {unique_molecules} molecules, {total_conformers} conformers")
        print(f"Average conformers per molecule: {avg_conf:.2f}")

        for label in ['is_TADF', 'is_rTADF']:
            if label in preprocessor.df.columns:
                cons = preprocessor.df.groupby('Molecule')[label].agg(['min', 'max'])
                inconsistent = (cons['min'] != cons['max']).sum()
                if inconsistent > 0:
                    print(f"  ⚠️ {inconsistent} molecules have inconsistent `{label}` among conformers")
    else:
        print("Warning: column `Molecule` not found; grouped validation still works with pseudo-groups.")

    # ----------------- Step 2: 特征工程 -----------------
    print("\nStep 2: Feature Engineering...")
    fe = FeatureEngineer(preprocessor.df)

    # 你的 create_interaction_features + 关键特征选择
    preprocessor.df = fe.create_interaction_features()
    critical_features = fe.select_critical_features()

    # 质量过滤（低覆盖率剔除）
    def _coverage(df, cols, thr=0.05):
        keep = []
        for c in cols:
            if c in df.columns:
                nz = int((df[c] != 0).sum())
                if nz / max(len(df), 1) > thr:
                    keep.append(c)
        return keep

    critical_features = _coverage(preprocessor.df, critical_features, thr=0.05)

    # 归一化仅作用于连续列（与项目原逻辑一致）
    # —— 这里直接委托 DataPreprocessor.normalize_features
    #    且只对“连续特征”进行标准化
    # 自动识别连续特征：非 has_/count_/onehot_、非整数少分类
    binary_like = [c for c in critical_features if c.startswith('has_') or 'push_pull_pattern_' in c
                   or 'ring_polarity_expected_' in c]
    count_like = [c for c in critical_features if c.startswith('count_') or c.startswith('num_')]
    cat_like = ['mol_type_code'] if 'mol_type_code' in critical_features else []
    skip = set(binary_like + count_like + cat_like)

    continuous_features = []
    for f in critical_features:
        if f in skip:
            continue
        if f in preprocessor.df.columns:
            vals = preprocessor.df[f].dropna().unique()
            # <=10 个唯一值且近似整数 -> 认为是分类/编码
            if len(vals) <= 10:
                vals = np.array(vals, dtype=float)
                if np.allclose(vals, np.round(vals)):
                    continue
            continuous_features.append(f)

    if continuous_features:
        print(f"Normalizing {len(continuous_features)} continuous features...")
        preprocessor.normalize_features(continuous_features)

    # 全量数据保存（剔除全零列）
    full_df = preprocessor.df.copy()
    drop_cols = []
    for c in full_df.columns:
        if c not in ['Molecule', 'SMILES', 'is_TADF', 'is_rTADF']:
            if np.issubdtype(full_df[c].dtype, np.number):
                if (full_df[c] == 0).all() or full_df[c].isna().all():
                    drop_cols.append(c)
    if drop_cols:
        full_df = full_df.drop(columns=drop_cols)
    full_df.to_csv('data/processed/full_features.csv', index=False)
    print(f"Saved full_features.csv: {full_df.shape}")

    # 关键：**统一特征集** —— 使用“设计安全特征”（去近标签）
    # 使 Step3/4/5 训练、Step6 OOF-stacking 与 Step6 测试评估完全一致
    design_features = get_design_safe_features(full_df)
    design_features = [f for f in design_features if f in full_df.columns]

    # 若设计特征过少，兜底回退到 critical_features（极少见）
    used_features = design_features if len(design_features) > 0 else [
        c for c in critical_features if c in full_df.columns
    ]
    print(f"\n[Feature Set] Using DESIGN-SAFE features: {len(used_features)} columns")

    # 仅保存用于训练/评估的子集（便于复现）
    cols_to_save = used_features + ['is_TADF', 'is_rTADF', 'Molecule', 'SMILES']
    cols_to_save = [c for c in cols_to_save if c in full_df.columns]
    critical_df = full_df[cols_to_save].copy()
    critical_df.to_csv('data/processed/critical_features.csv', index=False)
    print(f"Saved critical_features.csv with USED features: {critical_df.shape}")
    print("\n=== Removing constant features ===")
    numeric_critical = []
    non_numeric_critical = []
    for col in list(critical_features):
        if col in preprocessor.df.columns and pd.api.types.is_numeric_dtype(preprocessor.df[col]):
            numeric_critical.append(col)
        else:
            non_numeric_critical.append(col)

    if non_numeric_critical:
        print(f"[Sanity] Dropping {len(non_numeric_critical)} non-numeric features before split:")
        print(f"  Examples: {non_numeric_critical[:10]}")

    critical_features = numeric_critical
    # 让后续模块（ensemble 等）也用同一套最终列
    preprocessor.feature_cols = critical_features
    print(f"Using {len(critical_features)} numeric features for modeling.")
    # ----------------- Step 3: 数据划分（分子分组防泄漏） -----------------
    print("\nStep 3: Data Splitting...")
    # 将 used_features 回写，供内部 split 使用
    preprocessor.df = critical_df.copy()
    preprocessor.feature_cols = used_features  # 统一给后续解释/保存用

    (X_train, X_val, X_test), (y_tadf_train, y_tadf_val, y_tadf_test), \
    (y_rtadf_train, y_rtadf_val, y_rtadf_test) = preprocessor.split_data(
        use_features=used_features
    )

    print("\nFeature dimensions check:")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  #features used: {len(used_features)}  (design-safe)")

    # ----------------- Step 4: 深度学习模型训练（与 OOF 设定一致） -----------------
    print("\nStep 4: Training Deep Learning Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 类别权重（与 OOF 一致）
    n_t_pos = y_tadf_train.sum(); n_t_neg = len(y_tadf_train) - n_t_pos
    n_r_pos = y_rtadf_train.sum(); n_r_neg = len(y_rtadf_train) - n_r_pos
    tadf_pos_weight = float(n_t_neg / (n_t_pos + 1e-6))
    rtadf_pos_weight = float(n_r_neg / (n_r_pos + 1e-6))
    print(f"Class weights - TADF: {tadf_pos_weight:.2f}, rTADF: {rtadf_pos_weight:.2f}")

    dl_model = MultiTaskTADFNet(input_dim=X_train.shape[1]).to(device)
    trainer = ModelTrainer(dl_model, tadf_pos_weight=tadf_pos_weight, rtadf_pos_weight=rtadf_pos_weight)

    # 构建 DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_tadf_train),
                                  torch.FloatTensor(y_rtadf_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    best_val_pr_auc, patience, patience_counter = 0.0, 20, 0
    for epoch in range(100):
        train_loss = trainer.train_epoch(train_loader)
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_tadf_val),
                                    torch.FloatTensor(y_rtadf_val))
        val_loader = DataLoader(val_dataset, batch_size=32)
        val_metrics = trainer.evaluate(val_loader, use_optimal_threshold=True)
        val_pr_auc = (val_metrics.get('tadf_pr_auc', 0.0) + val_metrics.get('rtadf_pr_auc', 0.0)) / 2.0

        print(f"Epoch {epoch+1:02d} | Loss={train_loss:.4f} | "
              f"TADF PR-AUC={val_metrics.get('tadf_pr_auc',0):.4f} | "
              f"rTADF PR-AUC={val_metrics.get('rtadf_pr_auc',0):.4f} | "
              f"TADF F1={val_metrics.get('tadf_f1',0):.4f} | "
              f"rTADF F1={val_metrics.get('rtadf_f1',0):.4f}")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            torch.save(dl_model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        trainer.scheduler.step(val_pr_auc)

    # ----------------- Step 5: XGBoost 模型训练（同一特征集） -----------------
    print("\nStep 5: Training XGBoost Model...")
    xgb_predictor = XGBoostTADFPredictor()
    xgb_predictor.train(X_train, y_tadf_train, y_rtadf_train)

    # ----------------- Step 6: OOF-stacking（同一特征集 + 同一设定） -----------------
    print("\nStep 6: Model Evaluation with OOF-based Ensemble (Design Features, unified)...")
    # 重要：在 OOF 中也使用 **相同的 used_features**，并在折内给 DL 类权重训练
    design_oof = run_design_oof_stacking_pipeline(
        preprocessor,
        feature_cols=used_features,     # 强制统一
        n_splits=5,
        dl_epochs=20,
        seed=42,
        device="cpu",                   # OOF 建议 CPU（跨折一致性更好）
        save_dir="outputs"
    )

    stacker_tadf = design_oof['stacker_tadf']
    stacker_rtadf = design_oof['stacker_rtadf']
    thr_tadf_dict = design_oof['thresholds_tadf']  # {'f1', 'prevalence', 'fbeta2', 'fixed'}
    thr_rtadf_dict = design_oof['thresholds_rtadf']

    # 用主训练得到的 dl_model / xgb_predictor 在测试集算概率，再用 OOF 学到的 stacker 融合
    ens_probs_tadf, ens_probs_rtadf = apply_stacking_to_test(
        dl_model, xgb_predictor,
        stacker_tadf, stacker_rtadf,
        X_test, device=str(device)
    )

    # 采用 OOF 学得阈值：TADF->max-F1；rTADF->max-Fβ(β=2)
    best_thr_t = float(thr_tadf_dict['f1'])
    best_thr_r = float(thr_rtadf_dict['fbeta2'])

    ensemble_metrics = {
        'tadf_auc': roc_auc_score(y_tadf_test, ens_probs_tadf) if len(np.unique(y_tadf_test)) > 1 else 0.0,
        'tadf_pr_auc': average_precision_score(y_tadf_test, ens_probs_tadf) if len(np.unique(y_tadf_test)) > 1 else 0.0,
        'rtadf_auc': roc_auc_score(y_rtadf_test, ens_probs_rtadf) if len(np.unique(y_rtadf_test)) > 1 else 0.0,
        'rtadf_pr_auc': average_precision_score(y_rtadf_test, ens_probs_rtadf) if len(np.unique(y_rtadf_test)) > 1 else 0.0,
        'tadf_f1': f1_score(y_tadf_test, (ens_probs_tadf > best_thr_t).astype(int)),
        'rtadf_f1': f1_score(y_rtadf_test, (ens_probs_rtadf > best_thr_r).astype(int)),
    }

    print("\nStacking Ensemble Performance (Design features / OOF-learned threshold):")
    for k, v in ensemble_metrics.items():
        print(f"{k}: {v:.4f}")

    # 供解释/绘图使用
    ensemble_threshold_tadf = best_thr_t
    ensemble_threshold_rtadf = best_thr_r

    # ----------------- Step 7: 模型解释（与 used_features 对齐） -----------------
    print("\nStep 7: Model Interpretation...")
    feature_names = list(used_features)
    if len(feature_names) != X_train.shape[1]:
        print("Warning: Feature names mismatch; regenerating generic names.")
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

    # 用 TADF 的 XGB 模型做 SHAP/重要性展示（你项目的解释器是这么设计的）
    interpreter = ModelInterpreter(
        xgb_predictor.tadf_model.best_estimator_ if hasattr(xgb_predictor.tadf_model, 'best_estimator_')
        else xgb_predictor.tadf_model,
        X_train,
        feature_names
    )
    interpreter.shap_analysis(X_test[:min(100, len(X_test))], task='tadf')

    tadf_importance, rtadf_importance = xgb_predictor.get_feature_importance()
    tadf_values = list(tadf_importance.values()) if isinstance(tadf_importance, dict) else tadf_importance
    rtadf_values = list(rtadf_importance.values()) if isinstance(rtadf_importance, dict) else rtadf_importance

    interpreter.plot_feature_importance(tadf_values, top_n=20, task='tadf')
    interpreter.plot_feature_importance(rtadf_values, top_n=20, task='rtadf')

    combined_importance = (np.array(tadf_values) + np.array(rtadf_values)) / 2.0
    interpreter.plot_feature_importance(combined_importance, top_n=20, task='combined')

    # 混淆矩阵（使用 OOF 学到的阈值）
    ensemble_tadf_binary = (ens_probs_tadf > ensemble_threshold_tadf).astype(int)
    interpreter.plot_confusion_matrices(
        y_tadf_test, ensemble_tadf_binary,
        labels=['Non-TADF', 'TADF'],
        task=f'tadf_threshold_{ensemble_threshold_tadf:.3f}'
    )

    ensemble_rtadf_binary = (ens_probs_rtadf > ensemble_threshold_rtadf).astype(int)
    interpreter.plot_confusion_matrices(
        y_rtadf_test, ensemble_rtadf_binary,
        labels=['Non-rTADF', 'rTADF'],
        task=f'rtadf_threshold_{ensemble_threshold_rtadf:.3f}'
    )
    print("Note: Confusion matrices use OOF-learned thresholds")
    print(f"  TADF threshold: {ensemble_threshold_tadf:.3f}")
    print(f"  rTADF threshold: {ensemble_threshold_rtadf:.3f}")

    # ----------------- Step 8: 保存模型与工件 -----------------
    print("\nStep 8: Saving Models and Data...")
    joblib.dump(xgb_predictor, 'xgboost_tadf_predictor.pkl')
    torch.save({
        'model_state_dict': dl_model.state_dict(),
        'model_config': {'input_dim': X_train.shape[1], 'hidden_dims': [256, 128, 64], 'dropout': 0.3}
    }, 'best_dl_model.pth')

    if hasattr(preprocessor, 'scaler'):
        joblib.dump(preprocessor.scaler, 'scaler.pkl')
        print("Scaler saved successfully!")

    with open('data/splits/feature_names.txt', 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    print(f"Saved {len(feature_names)} feature names")
    print("All models and data saved successfully!")

    # ----------------- Step 9: 稳健性测试（保留你原流程） -----------------
    print("\n" + "="*60)
    print("Step 9: Robustness Testing")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("="*60)
    # 复用你项目里的函数；此处不再赘述（与前一个版本一致）
    from src.robustness_test import (
        define_feature_blocks, xgb_importances, perm_importance, shap_importance,
        rank_consistency, safe_cv_xgb_grouped, random_label_test, oof_score_grouped,
        calibrate_probabilities, plot_calibration_curve, bootstrap_confidence_intervals
    )

    # 诊断（含 gap）/设计（去近标签）
    diagnostic_features = list([c for c in preprocessor.df.columns
                                if c not in ['is_TADF', 'is_rTADF', 'Molecule', 'SMILES']])
    design_features = list(used_features)  # 与训练一致

    print("\n9.0 Feature Sets Preparation...")
    print(f"Diagnostic model features: {len(diagnostic_features)}")
    print(f"Design model features   : {len(design_features)}")

    print("\n9.1 Random Label Test (Leakage Detection)...")
    print("\nDiagnostic Model Random Label Test:")
    _ = random_label_test(preprocessor.df, diagnostic_features, n_iterations=3)
    print("\nDesign Model Random Label Test:")
    _ = random_label_test(preprocessor.df, design_features, n_iterations=3)

    print("\n9.2 Block Ablation Analysis (with GroupKFold)...")
    blocks = define_feature_blocks(pd.Index(diagnostic_features))

    def run_block_ablation_grouped(df, target_col, all_feats, blocks):
        results = []
        only_gap = [f for f in blocks.get('gap_block', []) if f in all_feats]
        only_da_geom = sorted(set([f for f in blocks.get('da_block', []) + blocks.get('geom_block', []) if f in all_feats]))
        only_structure = [f for f in blocks.get('structure_block', []) if f in all_feats]

        full = [f for f in all_feats]
        drop_gap = [c for c in full if c not in set(blocks.get('gap_block', []))]
        drop_da = [c for c in full if c not in set(blocks.get('da_block', []))]
        drop_geom = [c for c in full if c not in set(blocks.get('geom_block', []))]

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

    print("\nDesign Model Ablation (no gap features):")
    design_blocks = define_feature_blocks(pd.Index(design_features))
    res_design_rtadf = run_block_ablation_grouped(preprocessor.df, 'is_rTADF', design_features, design_blocks)
    print(res_design_rtadf[['setting', 'n_feats', 'roc_auc', 'f1', 'roc_auc_std']].to_string())
    res_design_rtadf.to_csv('outputs/design_rtadf_ablation.csv', index=False)

    print("\nDiagnostic Model Ablation (with gap features):")
    res_diag_rtadf = run_block_ablation_grouped(preprocessor.df, 'is_rTADF', diagnostic_features, blocks)
    print(res_diag_rtadf[['setting', 'n_feats', 'roc_auc', 'f1', 'roc_auc_std']].to_string())
    res_diag_rtadf.to_csv('outputs/diagnostic_rtadf_ablation.csv', index=False)

    print("\n9.3 Improved Feature Importance Consistency...")

    # 使用设计模型特征进行测试
    from sklearn.model_selection import GroupShuffleSplit
    
    # 关键修复：只保留数值列
    design_features_numeric = preprocessor.df[design_features].select_dtypes(include=[np.number]).columns.tolist()
    dropped = [f for f in design_features if f not in design_features_numeric]
    if dropped:
        print(f"  Dropping {len(dropped)} non-numeric features from design set: {dropped[:5]}")
    
   
    X_design = preprocessor.df[design_features_numeric].fillna(0).values.astype(np.float32)
    y_rtadf = preprocessor.df['is_rTADF'].astype(int).values
    groups = preprocessor.df['Molecule'].values if 'Molecule' in preprocessor.df.columns else None
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

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                eval_metric='logloss', n_jobs=1
            ))
        ])
    pipe.fit(X_tr, y_tr)

    model = pipe.named_steps['model']
    imputer = pipe.named_steps['imputer']
    X_val_imp = imputer.transform(X_val)

    imp_xgb = xgb_importances(model)
    imp_perm = perm_importance(
        pipe, X_val, y_val, design_features_numeric,
        scoring='roc_auc', n_repeats=5
    )
    imp_shap = shap_importance(
        model, X_val_imp, design_features_numeric, n_sample=min(300, len(X_val))  # 使用过滤后的数值特征列表
    )
    if len(imp_xgb) > 0 and len(imp_perm) > 0 and len(imp_shap) > 0:
        rho_gp, j_gp = rank_consistency(imp_xgb, 'gain', imp_perm, 'perm_importance', topk=20)
        rho_gs, j_gs = rank_consistency(imp_xgb, 'gain', imp_shap, 'shap_mean_abs', topk=20)
        rho_ps, j_ps = rank_consistency(imp_perm, 'perm_importance', imp_shap, 'shap_mean_abs', topk=20)
        print(f"  Spearman(gain vs perm): {rho_gp:.3f}, Jaccard@20: {j_gp:.2f}")
        print(f"  Spearman(gain vs SHAP): {rho_gs:.3f}, Jaccard@20: {j_gs:.2f}")
        print(f"  Spearman(perm vs SHAP): {rho_ps:.3f}, Jaccard@20: {j_ps:.2f}")
    # 9.3.5 增强的特征重要性稳定性分析（可选）
    print("\n9.3.5 Enhanced Feature Importance Stability (Filtered)...")
    
    from sklearn.model_selection import GroupKFold
    import warnings
    warnings.filterwarnings('ignore')
    
    # 重新使用design_features_numeric
    X_design_stable = preprocessor.df[design_features_numeric].fillna(0).values.astype(np.float32)
    
    # 先过滤掉低重要性特征
    preliminary_model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=1)
    preliminary_model.fit(X_design_stable, y_rtadf)
    
    # 获取初步重要性
    prelim_importance = preliminary_model.feature_importances_
    importance_threshold = np.percentile(prelim_importance, 25)  # 保留前75%
    
    # 过滤特征
    important_indices = prelim_importance > importance_threshold
    filtered_features = [design_features_numeric[i] for i in range(len(design_features_numeric)) if important_indices[i]]
    X_design_filtered = X_design_stable[:, important_indices]
    
    print(f"Filtered from {len(design_features_numeric)} to {len(filtered_features)} important features")
    
    # 在过滤后的特征上进行稳定性分析
    stability_results = []
    n_repeats = 5
    gkf = GroupKFold(n_splits=5)
    
    for seed in range(42, 42 + n_repeats):
        np.random.seed(seed)
        fold_importances = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_design_filtered, y_rtadf, groups)):
            X_fold_train = X_design_filtered[train_idx]
            y_fold_train = y_rtadf[train_idx]
            
            # 训练模型
            fold_model = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=1
            )
            fold_model.fit(X_fold_train, y_fold_train)
            fold_importances.append(fold_model.feature_importances_)
        
        avg_importance = np.mean(fold_importances, axis=0)
        stability_results.append(avg_importance)
    
    # 计算稳定性指标
    stability_matrix = np.array(stability_results)
    mean_importance = stability_matrix.mean(axis=0)
    std_importance = stability_matrix.std(axis=0)
    cv_importance = std_importance / (mean_importance + 1e-10)
    
    # 找出最稳定的特征
    stable_features_idx = np.argsort(cv_importance)[:20]
    
    print(f"\nTop 20 Most Stable Features (after filtering):")
    for i, idx in enumerate(stable_features_idx[:20]):
        if idx < len(filtered_features):  # 添加边界检查
            feat_name = filtered_features[idx]
            print(f"  {i+1:2d}. {feat_name:30s} CV={cv_importance[idx]:.3f} Importance={mean_importance[idx]:.4f}")
    print("\n9.4 Enhanced Extrapolation Stability (with Groups)...")
    if 'Molecule' in preprocessor.df.columns:
        groups = preprocessor.df['Molecule'].values
    else:
        groups = np.arange(len(preprocessor.df))

    oof_metrics, _ = oof_score_grouped(
        X_design, y_rtadf, groups,
        seeds=(7, 13, 23),
        n_splits=5
    )
    print(f"\nDesign Model Grouped OOF Results:")
    for metric, value in oof_metrics.items():
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}")

    # 概率校准 + Bootstrap CI（rTADF，以设计特征为准）
    print("\n9.5 Probability Calibration and Bootstrap CI...")
    if len(design_features_numeric) > 0 and 'Molecule' in preprocessor.df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        trv_idx, test_idx = next(gss.split(X_design, y_rtadf, groups))
        X_trainval, X_test_d = X_design[trv_idx], X_design[test_idx]
        y_trainval, y_test_d = y_rtadf[trv_idx], y_rtadf[test_idx]
        groups_trv = groups[trv_idx]

        # 从 trainval 中再切出一小部分作为校准的内部验证
        val_ratio = 0.1 / 0.8
        tr_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
                               .split(X_trainval, y_trainval, groups_trv))
        X_train_d, y_train_d = X_trainval[tr_idx], y_trainval[tr_idx]

        final_model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=1
        )
        final_model.fit(X_train_d, y_train_d)

        prob_uncal, prob_cal, _ = calibrate_probabilities(
            final_model, X_train_d, y_train_d, X_test_d, method='isotonic'
        )
        ece_before, ece_after = plot_calibration_curve(y_test_d, prob_uncal, prob_cal, task='design_rtadf')
        print(f"  ECE before calibration: {ece_before:.4f}")
        print(f"  ECE after  calibration: {ece_after:.4f}")

        ci_results = bootstrap_confidence_intervals(
            X_test_d, y_test_d, final_model, n_bootstrap=1000
        )
        print(f"  ROC-AUC: {ci_results['roc_auc_mean']:.3f} "
              f"(95% CI: [{ci_results['roc_auc_ci'][0]:.3f}, {ci_results['roc_auc_ci'][1]:.3f}])")
        print(f"  PR-AUC : {ci_results['pr_auc_mean']:.3f} "
              f"(95% CI: [{ci_results['pr_auc_ci'][0]:.3f}, {ci_results['pr_auc_ci'][1]:.3f}])")

        final_summary = pd.DataFrame({
            'Model': ['Design_OOF', 'Design_Calibrated'],
            'ROC-AUC': [oof_metrics.get('roc_auc', np.nan), ci_results['roc_auc_mean']],
            'PR-AUC': [oof_metrics.get('pr_auc', np.nan), ci_results['pr_auc_mean']],
            'ECE': [np.nan, ece_after],
            'F1': [oof_metrics.get('f1', np.nan), np.nan],
            'Optimal_Threshold': [oof_metrics.get('optimal_threshold', np.nan), np.nan]
        })
    else:
        print("  Skipping calibration/CI (no design features or no molecular groups).")
        final_summary = pd.DataFrame({
            'Model': ['Design_OOF'],
            'ROC-AUC': [oof_metrics.get('roc_auc', np.nan)],
            'PR-AUC': [oof_metrics.get('pr_auc', np.nan)],
            'ECE': [np.nan],
            'F1': [oof_metrics.get('f1', np.nan)],
            'Optimal_Threshold': [oof_metrics.get('optimal_threshold', np.nan)]
        })

    final_summary.to_csv('outputs/final_model_summary.csv', index=False)
    print("\nFinal summary saved to outputs/final_model_summary.csv")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
