# src/robustness_test.py
"""稳健性测试模块"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def _select_numeric_features(df: pd.DataFrame, features):
    """Return the intersection of `features` and numeric columns in df; log dropped ones."""
    inter = [c for c in features if c in df.columns]
    if not inter:
        raise ValueError("[robustness] No overlap between requested features and df columns.")
    numeric = df[inter].select_dtypes(include=[np.number]).columns.tolist()
    dropped = [c for c in inter if c not in numeric]
    if dropped:
        print(f"[robustness] Dropping {len(dropped)} non-numeric features:")
        print(f"  Examples: {dropped[:10]}")
    return numeric

def _to_numeric_matrix(df: pd.DataFrame, cols):
    """Coerce to numeric, replace inf/NaN with 0, and return float32 numpy array."""
    X_df = df[cols].copy()
    X_df = X_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_df.to_numpy(dtype=np.float32)
# ============ 分块消融（Block Ablation）============
def define_feature_blocks(df_cols):
    """定义特征块"""
    cols = list(df_cols)
    
    # Gap特征块
    gap_patterns = [
        r'^s\d+_t\d+_gap$',
        r'st_.*gap',
        r'num_inverted_gaps',
        r'primary_inversion_gap'
    ]
    gap_block = []
    for col in cols:
        if any(re.match(pattern, col) for pattern in gap_patterns):
            gap_block.append(col)
    
    # D/A电子特征块
    da_block = [c for c in cols if c in [
        'D_A_ratio', 'DA_balance', 'D_A_product', 'donor_score', 'acceptor_score',
        'homo', 'lumo', 'homo_lumo_gap', 'CT_position_weighted_score', 
        'A_density', 'D_density', 'out_sub_density', 'in_sub_density',
        'DA_strength_5minus3', 'donor_homo_effect', 'acceptor_lumo_effect'
    ]]
    
    # 几何形状特征块
    geom_block = []
    geom_patterns = ['gaussian_', 'crest_', 'planarity', 'aspect_ratio', 'molecular_complexity']
    for col in cols:
        if any(pattern in col for pattern in geom_patterns):
            geom_block.append(col)
    
    # 结构特征块
    structure_block = []
    struct_patterns = ['num_', 'count_', 'has_', '_density']
    for col in cols:
        if any(pattern in col for pattern in struct_patterns):
            # 排除已分类的特征
            if col not in set(gap_block + da_block + geom_block):
                structure_block.append(col)
    
    return {
        'gap_block': gap_block,
        'da_block': da_block,
        'geom_block': geom_block,
        'structure_block': structure_block
    }

def eval_metrics(y_true, y_prob, threshold=0.5):
    """计算评估指标"""
    y_pred = (y_prob >= threshold).astype(int)
    
    # 处理单类别情况
    if len(np.unique(y_true)) <= 1:
        return {
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'f1': 0.0,
            'bal_acc': 0.0
        }
    
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'bal_acc': balanced_accuracy_score(y_true, y_pred)
    }

def cv_run_xgb(X, y, params=None, n_splits=5, seed=42):
    """交叉验证运行XGBoost"""
    if params is None:
        params = dict(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric='logloss', random_state=seed, n_jobs=-1
        )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics = []
    
    for tr_idx, te_idx in skf.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx])
        
        # 预测概率
        prob = model.predict_proba(X[te_idx])[:, 1]
        metrics.append(eval_metrics(y[te_idx], prob))
    
    return pd.DataFrame(metrics).mean().to_dict()
def get_design_safe_features(df):
    """获取设计模型安全特征（移除近标签特征）"""
    import re
    cols = df.columns.tolist()
    
    # 禁用的特征模式
    ban_patterns = [
        r'^s\d+_t\d+_gap$',           # 所有S-T gap
        r'^st_.*gap',                  # 衍生gap
        r'num_inverted_gaps',          # 反转gap计数
        r'primary_inversion',          # 主要反转
        r'.*oscillator$',              # 振子强度
        r'has_.*inversion',            # 反转标志
        r'gap',                        # 任何包含gap的
        r'TADF_strength',              # TADF强度
        r'rTADF_type'                  # rTADF类型
    ]
    
    keep = []
    removed = []
    for c in cols:
        if any(re.search(p, c) for p in ban_patterns):
            removed.append(c) 
            continue
        keep.append(c)
    
    # 排除标签和非数值列
    exclude = {'is_TADF', 'is_rTADF', 'Molecule', 'State', 'smiles', 
               'SMILES', 'conformer', 'inverted_gaps', 'singlet_states', 
               'triplet_states'}
    keep = [c for c in keep if c not in exclude]
    
    print(f"Removed {len(removed)} near-label features for design model")
    if len(removed) < 20:  # 只打印前20个
        print(f"  Examples: {removed[:20]}")
    
    return keep

def safe_cv_xgb_grouped(df, features, target_col, n_splits=5, seeds=(42,), groups_col='Molecule'):
    """
    GroupKFold + XGBoost 的安全CV：
      - 仅保留数值特征
      - 强制数值化/补齐
    返回 dict(metrics...)
    """
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    # --- 关键修复：数值过滤 + 安全数值化 ---
    numeric_feats = _select_numeric_features(df, features)
    X = _to_numeric_matrix(df, numeric_feats)

    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int).values
    if groups_col in df.columns:
        groups = df[groups_col].astype(str).values
    else:
        groups = np.arange(len(df))

    gkf = GroupKFold(n_splits=n_splits)

    aucs, pras, f1s = [], [], []
    for seed in (seeds if isinstance(seeds, (list, tuple)) else [seeds]):
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # 轻量、稳定的参数
            model = XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=int(seed),
                n_jobs=0, eval_metric='logloss'
            )
            model.fit(X_tr, y_tr)

            va_prob = model.predict_proba(X_va)[:, 1]
            # 防止单类折AUC/PR-AUC报错
            if np.unique(y_va).size < 2:
                auc  = 0.0
                pr   = 0.0
                thr  = 0.5
                pred = (va_prob > thr).astype(int)
            else:
                from sklearn.metrics import roc_auc_score, average_precision_score
                from sklearn.metrics import precision_recall_curve
                auc = float(roc_auc_score(y_va, va_prob))
                pr  = float(average_precision_score(y_va, va_prob))
                p, r, thr = precision_recall_curve(y_va, va_prob)
                f1_arr = 2 * p * r / (p + r + 1e-10)
                thr_idx = int(np.argmax(f1_arr[:-1])) if len(f1_arr) > 1 else 0
                thr = float(thr[thr_idx]) if len(thr) else 0.5
                pred = (va_prob > thr).astype(int)

            f1 = float(f1_score(y_va, pred)) if np.unique(y_va).size >= 2 else 0.0
            fold_scores.append((auc, pr, f1))

        aucs.append(np.mean([s[0] for s in fold_scores]))
        pras.append(np.mean([s[1] for s in fold_scores]))
        f1s.append(np.mean([s[2] for s in fold_scores]))

    return dict(
        roc_auc=float(np.mean(aucs)),
        pr_auc=float(np.mean(pras)),
        f1=float(np.mean(f1s)),
        roc_auc_std=float(np.std(aucs)),
        pr_auc_std=float(np.std(pras)),
        f1_std=float(np.std(f1s)),
    )


def random_label_test(df, features, n_iterations=5, target_col='is_rTADF', groups_col='Molecule'):
    """
    将标签打乱，检查模型是否还能学到“性能”——用于泄漏探测。
    """
    import numpy as np

    numeric_feats = _select_numeric_features(df, features)
    X = _to_numeric_matrix(df, numeric_feats)

    if groups_col in df.columns:
        groups = df[groups_col].astype(str).values
    else:
        groups = np.arange(len(df))

    rng = np.random.default_rng(42)
    scores = []
    for i in range(n_iterations):
        y_perm = rng.permutation(pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int).values)
        tmp_df = df.copy()
        tmp_df['_rand_y_'] = y_perm
        m = safe_cv_xgb_grouped(tmp_df, numeric_feats, '_rand_y_', n_splits=5, seeds=(i+1,), groups_col=groups_col)
        scores.append(m['roc_auc'])
        print(f"  Iter {i+1}/{n_iterations}: AUC={m['roc_auc']:.3f}, PR-AUC={m['pr_auc']:.3f}, F1={m['f1']:.3f}")

    print(f"Random-label mean AUC={np.mean(scores):.3f} ± {np.std(scores):.3f}")
    return dict(mean_auc=float(np.mean(scores)), std_auc=float(np.std(scores)))

def run_block_ablation(df, target_col, all_feats, blocks):
    """运行分块消融实验"""
    results = []
    
    # 1) 单独块
    only_gap = blocks['gap_block']
    only_da_geom = sorted(set(blocks['da_block'] + blocks['geom_block']))
    
    # 2) 完整特征和去除某块
    full = all_feats
    drop_gap = [c for c in full if c not in set(blocks['gap_block'])]
    drop_da = [c for c in full if c not in set(blocks['da_block'])]
    drop_geom = [c for c in full if c not in set(blocks['geom_block'])]
    
    exps = {
        'only_gap': only_gap,
        'only_da_geom': only_da_geom,
        'full': full,
        'full_minus_gap': drop_gap,
        'full_minus_da': drop_da,
        'full_minus_geom': drop_geom
    }
    
    for name, cols in exps.items():
        if len(cols) == 0:
            continue
        
        X = df[cols].astype(np.float32).fillna(0).values
        y = df[target_col].astype(int).values
        
        m = cv_run_xgb(X, y)
        m['setting'] = name
        m['n_feats'] = len(cols)
        results.append(m)
    
    return pd.DataFrame(results).sort_values('roc_auc', ascending=False)

# ============ 多种重要性度量一致性 ============
def xgb_importances(model):
    """获取XGBoost的三种重要性"""
    bst = model.get_booster()
    gains = bst.get_score(importance_type='gain')
    splits = bst.get_score(importance_type='weight')
    covers = bst.get_score(importance_type='cover')
    
    def to_df(d, name):
        return pd.DataFrame({'feature': list(d.keys()), name: list(d.values())})
    
    df = to_df(gains, 'gain')
    if splits:
        df = df.merge(to_df(splits, 'split'), on='feature', how='outer')
    if covers:
        df = df.merge(to_df(covers, 'cover'), on='feature', how='outer')
    
    return df.fillna(0)

from sklearn.inspection import permutation_importance


def perm_importance(model, X, y, features, scoring='roc_auc', n_repeats=5):
    """
    计算并返回特征重要性（permutation importance）
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, scoring=scoring
    )
    
    # 确保赋值给正确的变量
    r = result  # 这就是 `permutation_importance` 的返回值

    # 使用 `r` 中的属性，确保在 DataFrame 中存储
    df = pd.DataFrame({
        'feature': features, 
        'perm_importance': r.importances_mean  # 这是PermutationImportance对象的属性
    })
    
    return df


def shap_importance(model, X_train, feature_names, n_sample=500):
    """计算SHAP重要性"""
    # 限制样本数量以加快计算
    n_sample = min(n_sample, len(X_train))
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:n_sample])
    
    # 对于二分类，如果返回列表，取正类的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    imp = np.abs(shap_values).mean(axis=0)
    
    return pd.DataFrame({
        'feature': feature_names,
        'shap_mean_abs': imp
    }).sort_values('shap_mean_abs', ascending=False)

def rank_consistency(df1, col1, df2, col2, topk=20):
    """计算排名一致性"""
    # 确保特征名称匹配
    all_features = list(set(df1['feature']) | set(df2['feature']))
    
    # 创建排名
    rank1 = df1.set_index('feature')[col1].reindex(all_features).fillna(0).rank(ascending=False)
    rank2 = df2.set_index('feature')[col2].reindex(all_features).fillna(0).rank(ascending=False)
    
    # Spearman相关系数
    rho, _ = spearmanr(rank1, rank2)
    
    # Top-K Jaccard相似度
    top1 = set(df1.nlargest(topk, col1)['feature'])
    top2 = set(df2.nlargest(topk, col2)['feature'])
    jaccard = len(top1 & top2) / max(1, len(top1 | top2))
    
    return rho, jaccard

# ============ 阈值敏感性测试 ============
def create_rtadf_label_parametric(df, th_s1tn=0.3, th_s2tn=-0.2):
    """参数化rTADF标签创建"""
    df = df.copy()
    conds = []
    
    # S1-Tn反转
    for i in range(1, 4):
        col = f's1_t{i}_gap'
        if col in df.columns:
            conds.append(df[col].abs() < th_s1tn)
    
    # S2-Tn反转
    for i in range(1, 4):
        col = f's2_t{i}_gap'
        if col in df.columns:
            conds.append(df[col] < th_s2tn)
    
    if conds:
        df['is_rTADF'] = np.any(conds, axis=0).astype(int)
    else:
        df['is_rTADF'] = 0
    
    return df

def sweep_thresholds(prep_df, feats, grid_s1=(0.3, 0.4, 0.5), grid_s2=(-0.1, -0.2, -0.3)):
    """阈值敏感性扫描"""
    records = []
    
    for th_s1 in grid_s1:
        for th_s2 in grid_s2:
            df = create_rtadf_label_parametric(prep_df, th_s1tn=th_s1, th_s2tn=th_s2)
            
            X = df[feats].astype(np.float32).fillna(0).values
            y = df['is_rTADF'].astype(int).values
            
            # 跳过全零标签
            if y.sum() == 0 or y.sum() == len(y):
                continue
            
            metrics = cv_run_xgb(X, y, n_splits=3)  # 减少CV折数加快速度
            pos_rate = y.mean()
            
            metrics.update({
                'th_s1': th_s1,
                'th_s2': th_s2,
                'pos_rate': pos_rate
            })
            
            records.append(metrics)
    
    return pd.DataFrame(records).sort_values('roc_auc', ascending=False)

# ============ 外推稳定性测试 ============
def oof_score_grouped(X, y, groups, seeds=(7, 13, 23), n_splits=5):
    """
    分组Out-of-Fold评分 - 确保同一分子不会同时出现在训练和测试中
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from xgboost import XGBClassifier
    from sklearn.metrics import precision_recall_curve
    
    oof_probs = np.zeros_like(y, dtype=float)
    oof_counts = np.zeros_like(y, dtype=int)
    fold_thresholds = []
    
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_splits)
        
        for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
            # 创建管道确保无泄漏
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', XGBClassifier(
                    n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, 
                    random_state=seed, n_jobs=-1
                ))
            ])
            
            # 训练
            pipe.fit(X[tr_idx], y[tr_idx])
            
            # 预测
            fold_probs = pipe.predict_proba(X[te_idx])[:, 1]
            oof_probs[te_idx] += fold_probs
            oof_counts[te_idx] += 1
            
            # 在验证集上找最优阈值
            if len(np.unique(y[te_idx])) > 1:
                p, r, thresholds = precision_recall_curve(y[te_idx], fold_probs)
                f1_scores = 2 * p * r / (p + r + 1e-12)
                best_idx = np.nanargmax(f1_scores)
                best_threshold = thresholds[max(best_idx-1, 0)]
                fold_thresholds.append(best_threshold)
    
    # 平均预测
    oof_probs = oof_probs / (oof_counts + 1e-12)
    
    # 使用平均最优阈值
    if fold_thresholds:
        optimal_threshold = np.mean(fold_thresholds)
    else:
        optimal_threshold = 0.5
    
    # 评估（使用优化的阈值）
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
    y_pred = (oof_probs >= optimal_threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y, oof_probs) if len(np.unique(y)) > 1 else np.nan,
        'pr_auc': average_precision_score(y, oof_probs) if len(np.unique(y)) > 1 else np.nan,
        'f1': f1_score(y, y_pred, zero_division=0),
        'bal_acc': balanced_accuracy_score(y, y_pred),
        'optimal_threshold': optimal_threshold
    }
    
    return metrics, oof_probs

# 修改原有的oof_score为调用新函数
def oof_score(X, y, seeds=(7, 13, 23), n_splits=5):
    """兼容旧接口，但使用随机分组（不推荐）"""
    # 创建伪分组
    groups = np.arange(len(y))
    return oof_score_grouped(X, y, groups, seeds, n_splits)

def multi_seed_stability(X, y, seeds=(42, 7, 13, 23, 31)):
    """多随机种子稳定性测试"""
    results = []
    
    for seed in seeds:
        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1
        )
        
        # 简单的训练-测试分割
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_te)[:, 1]
        
        metrics = eval_metrics(y_te, y_pred)
        metrics['seed'] = seed
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # 计算均值和标准差
    summary = {
        'roc_auc_mean': df['roc_auc'].mean(),
        'roc_auc_std': df['roc_auc'].std(),
        'f1_mean': df['f1'].mean(),
        'f1_std': df['f1'].std(),
        'pr_auc_mean': df['pr_auc'].mean(),
        'pr_auc_std': df['pr_auc'].std()
    }
    
    return df, summary

def calibrate_probabilities(model, X_train, y_train, X_test, method='isotonic'):
    """概率校准"""
    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    
    # 原始概率
    if hasattr(model, 'predict_proba'):
        prob_uncalibrated = model.predict_proba(X_test)[:, 1]
    else:
        prob_uncalibrated = model.decision_function(X_test)
    
    # 校准后概率
    prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
    
    return prob_uncalibrated, prob_calibrated, calibrated

def plot_calibration_curve(y_true, prob_uncalibrated, prob_calibrated, task=''):
    """绘制校准曲线"""
    from sklearn.calibration import calibration_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 校准前
    fraction_pos, mean_pred = calibration_curve(y_true, prob_uncalibrated, n_bins=10)
    ax1.plot(mean_pred, fraction_pos, 's-', label='Uncalibrated')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Calibration Before - {task}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 校准后
    fraction_pos, mean_pred = calibration_curve(y_true, prob_calibrated, n_bins=10)
    ax2.plot(mean_pred, fraction_pos, 's-', label='Calibrated', color='green')
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Fraction of Positives')
    ax2.set_title(f'Calibration After - {task}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 计算ECE
    ece_before = expected_calibration_error(y_true, prob_uncalibrated)
    ece_after = expected_calibration_error(y_true, prob_calibrated)
    
    plt.suptitle(f'ECE Before: {ece_before:.4f}, ECE After: {ece_after:.4f}')
    plt.tight_layout()
    plt.savefig(f'outputs/calibration_curve_{task}.png', dpi=300)
    plt.close()
    
    return ece_before, ece_after

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """计算期望校准误差(ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def bootstrap_confidence_intervals(X_test, y_test, model, n_bootstrap=1000, alpha=0.05):
    """Bootstrap置信区间"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    n_samples = len(X_test)
    roc_scores = []
    pr_scores = []
    
    for _ in range(n_bootstrap):
        # 重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test[indices]
        
        # 跳过只有一个类的情况
        if len(np.unique(y_boot)) < 2:
            continue
        
        # 预测
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_boot)[:, 1]
        else:
            y_pred = model.decision_function(X_boot)
        
        # 计算指标
        roc_scores.append(roc_auc_score(y_boot, y_pred))
        pr_scores.append(average_precision_score(y_boot, y_pred))
    
    # 计算置信区间
    lower = alpha / 2
    upper = 1 - alpha / 2
    
    ci_results = {
        'roc_auc_mean': np.mean(roc_scores),
        'roc_auc_std': np.std(roc_scores),
        'roc_auc_ci': (np.percentile(roc_scores, lower * 100), 
                       np.percentile(roc_scores, upper * 100)),
        'pr_auc_mean': np.mean(pr_scores),
        'pr_auc_std': np.std(pr_scores),
        'pr_auc_ci': (np.percentile(pr_scores, lower * 100), 
                     np.percentile(pr_scores, upper * 100))
    }
    
    return ci_results