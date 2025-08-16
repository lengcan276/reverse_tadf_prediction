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

def safe_cv_xgb_grouped(df, features, target, group_col='Molecule', n_splits=5, seed=42):
    """使用GroupKFold的安全交叉验证"""
    from sklearn.model_selection import GroupKFold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    X = df[features].astype(np.float32).fillna(0).values
    y = df[target].astype(int).values
    groups = df[group_col].values if group_col in df.columns else None
    
    if groups is None:
        print("Warning: No group column found, using standard CV")
        return cv_run_xgb(X, y, seed=seed)
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, 
            random_state=seed, n_jobs=-1
        ))
    ])
    
    gkf = GroupKFold(n_splits=n_splits)
    metrics = []
    
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1]
        
        m = eval_metrics(y[te_idx], prob)
        m['fold'] = fold
        metrics.append(m)
    
    df_metrics = pd.DataFrame(metrics)
    mean_metrics = df_metrics.drop('fold', axis=1).mean().to_dict()
    std_metrics = df_metrics.drop('fold', axis=1).std().to_dict()
    
    # 添加标准差信息
    for key in std_metrics:
        mean_metrics[f'{key}_std'] = std_metrics[key]
    
    return mean_metrics

def random_label_test(df, features, n_iterations=5):
    """随机标签测试检测泄漏"""
    results = []
    
    for i in range(n_iterations):
        df_random = df.copy()
        # 随机打乱标签
        df_random['is_TADF'] = np.random.permutation(df['is_TADF'].values)
        df_random['is_rTADF'] = np.random.permutation(df['is_rTADF'].values)
        
        # 测试TADF
        metrics_tadf = safe_cv_xgb_grouped(
            df_random, features, 'is_TADF', n_splits=3, seed=42+i
        )
        metrics_tadf['task'] = 'TADF'
        metrics_tadf['iteration'] = i
        results.append(metrics_tadf)
        
        # 测试rTADF
        metrics_rtadf = safe_cv_xgb_grouped(
            df_random, features, 'is_rTADF', n_splits=3, seed=42+i
        )
        metrics_rtadf['task'] = 'rTADF'
        metrics_rtadf['iteration'] = i
        results.append(metrics_rtadf)
    
    df_results = pd.DataFrame(results)
    
    # 汇总
    summary = {
        'TADF_auc_mean': df_results[df_results['task']=='TADF']['roc_auc'].mean(),
        'TADF_auc_std': df_results[df_results['task']=='TADF']['roc_auc'].std(),
        'rTADF_auc_mean': df_results[df_results['task']=='rTADF']['roc_auc'].mean(),
        'rTADF_auc_std': df_results[df_results['task']=='rTADF']['roc_auc'].std(),
    }
    
    print("\nRandom Label Test Results:")
    print(f"  TADF AUC: {summary['TADF_auc_mean']:.3f} ± {summary['TADF_auc_std']:.3f}")
    print(f"  rTADF AUC: {summary['rTADF_auc_mean']:.3f} ± {summary['rTADF_auc_std']:.3f}")
    
    if summary['TADF_auc_mean'] > 0.6 or summary['rTADF_auc_mean'] > 0.6:
        print("  ⚠️ WARNING: Random label AUC > 0.6, possible data leakage!")
    else:
        print("  ✓ Random label test passed")
    
    return summary
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

def perm_importance(model, X_val, y_val, feature_names, scoring='roc_auc', n_repeats=10):
    """计算置换重要性"""
    r = permutation_importance(
        model, X_val, y_val, 
        n_repeats=n_repeats, 
        random_state=0, 
        scoring=scoring, 
        n_jobs=-1
    )
    
    df = pd.DataFrame({
        'feature': feature_names, 
        'perm_importance': r.importances_mean
    })
    return df.sort_values('perm_importance', ascending=False)

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