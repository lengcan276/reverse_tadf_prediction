# src/ensemble.py
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.models import MultiTaskTADFNet, XGBoostTADFPredictor
from src.robustness_test import get_design_safe_features


# --------------------------- #
# utils
# --------------------------- #
def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_design_feature_list(df: pd.DataFrame, available_cols=None):
    """
    生成“设计特征集”（去近标签）并与当前可用列求交集。
    """
    design_feats = get_design_safe_features(df)
    if available_cols is not None:
        design_feats = [f for f in design_feats if f in list(available_cols)]
    return design_feats


# --------------------------- #
# OOF 预测（DL + XGB）
# --------------------------- #
def _safe_pr_auc(y_true, y_prob):
    """若验证集中只有单类，返回0，避免报错。"""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def generate_oof_predictions(
    df: pd.DataFrame,
    feature_cols,
    n_splits: int = 5,
    dl_epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
):
    """
    用 GroupKFold 在“设计特征集”上生成 OOF 预测：
      - 轻量 DL（MultiTaskTADFNet，带类权重 + 早停）
      - XGB（XGBoostTADFPredictor）
    返回：
      oof_dict = {
        'dl_tadf', 'dl_rtadf', 'xgb_tadf', 'xgb_rtadf',
        'y_tadf', 'y_rtadf'
      }
    """
    set_global_seed(seed)

    # === 关键修复：只保留数值列，并把无法解析为数值的内容安全处理 ===
    df_feat_raw = df[feature_cols].copy()
    numeric_cols = df_feat_raw.select_dtypes(include=[np.number]).columns.tolist()
    dropped = [c for c in feature_cols if c not in numeric_cols]
    if dropped:
        print(f"[ensemble.generate_oof_predictions] Dropping {len(dropped)} non-numeric columns:")
        print(f"  Examples: {dropped[:10]}")

    df_feat = df_feat_raw[numeric_cols].copy()
    # 兜底：如果仍有杂质，强制转数值；无法解析的置 NaN -> 0
    df_feat = df_feat.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_full = df_feat.to_numpy(dtype=np.float32)

    y_tadf_full = df['is_TADF'].astype(int).values
    y_rtadf_full = df['is_rTADF'].astype(int).values
    groups = df['Molecule'].values if 'Molecule' in df.columns else np.arange(len(df))

    oof_dl_tadf = np.zeros(len(X_full), dtype=float)
    oof_dl_rtadf = np.zeros(len(X_full), dtype=float)
    oof_xgb_tadf = np.zeros(len(X_full), dtype=float)
    oof_xgb_rtadf = np.zeros(len(X_full), dtype=float)

    gkf = GroupKFold(n_splits=n_splits)
    print("Generating OOF predictions (design features)...")
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_full, y_rtadf_full, groups)):
        print(f"  Fold {fold_idx+1}/{n_splits}...")

        X_tr, X_va = X_full[tr_idx], X_full[va_idx]
        y_tadf_tr, y_rtadf_tr = y_tadf_full[tr_idx], y_rtadf_full[tr_idx]
        y_tadf_va, y_rtadf_va = y_tadf_full[va_idx], y_rtadf_full[va_idx]

        # ----- 训练轻量 DL（类权重 + 早停，默认 CPU）-----
        dev = torch.device(device)
        dl = MultiTaskTADFNet(input_dim=X_tr.shape[1]).to(dev)

        # 类权重
        n_t_pos = y_tadf_tr.sum(); n_t_neg = len(y_tadf_tr) - n_t_pos
        n_r_pos = y_rtadf_tr.sum(); n_r_neg = len(y_rtadf_tr) - n_r_pos
        t_pos_w = float(n_t_neg / (n_t_pos + 1e-6))
        r_pos_w = float(n_r_neg / (n_r_pos + 1e-6))

        optimizer = torch.optim.Adam(dl.parameters(), lr=1e-3)
        loss_t = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([t_pos_w], device=dev))
        loss_r = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r_pos_w], device=dev))

        ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tadf_tr).float(),
            torch.from_numpy(y_rtadf_tr).float(),
        )
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        X_va_t = torch.from_numpy(X_va).float().to(dev)

        # 早停设定
        patience = min(10, max(3, dl_epochs // 3))
        best_score = -np.inf
        patience_cnt = 0
        best_state = None

        dl.train()
        for ep in range(dl_epochs):
            for xb, yb_t, yb_r in loader:
                xb = xb.to(dev); yb_t = yb_t.to(dev); yb_r = yb_r.to(dev)
                optimizer.zero_grad()
                logit_t, logit_r = dl(xb)
                lt = loss_t(logit_t.squeeze(), yb_t)
                lr = loss_r(logit_r.squeeze(), yb_r)
                (lt + lr).backward()
                optimizer.step()

            # 验证 PR-AUC（两任务平均）
            dl.eval()
            with torch.no_grad():
                logit_t_va, logit_r_va = dl(X_va_t)
                pt_va = torch.sigmoid(logit_t_va).cpu().numpy().ravel()
                pr_va = torch.sigmoid(logit_r_va).cpu().numpy().ravel()
            dl.train()

            pr_t = _safe_pr_auc(y_tadf_va, pt_va)
            pr_r = _safe_pr_auc(y_rtadf_va, pr_va)
            mean_pr = (pr_t + pr_r) / 2.0

            if mean_pr > best_score:
                best_score = mean_pr
                best_state = {k: v.cpu().clone() for k, v in dl.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        # 恢复最优
        if best_state is not None:
            dl.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        dl.eval()
        with torch.no_grad():
            logit_t, logit_r = dl(torch.from_numpy(X_va).float().to(dev))
            oof_dl_tadf[va_idx] = torch.sigmoid(logit_t).cpu().numpy().ravel()
            oof_dl_rtadf[va_idx] = torch.sigmoid(logit_r).cpu().numpy().ravel()

        # ----- 训练 XGB -----
        xgb = XGBoostTADFPredictor()
        xgb.train(X_tr, y_tadf_tr, y_rtadf_tr)
        pred_t, pred_r = xgb.predict(X_va)
        oof_xgb_tadf[va_idx] = pred_t
        oof_xgb_rtadf[va_idx] = pred_r

    oof_dict = dict(
        dl_tadf=oof_dl_tadf,
        dl_rtadf=oof_dl_rtadf,
        xgb_tadf=oof_xgb_tadf,
        xgb_rtadf=oof_xgb_rtadf,
        y_tadf=y_tadf_full,
        y_rtadf=y_rtadf_full,
    )
    return oof_dict



# --------------------------- #
# 学 stacking 权重（LogReg）
# --------------------------- #
def fit_stacking_models(oof_dict, seed: int = 42):
    """
    用 OOF 预测学习两路（TADF / rTADF）的 LR stacking 模型。
    返回：
      stacker_tadf, stacker_rtadf, weights_info
    """
    X_tadf = np.column_stack([oof_dict['dl_tadf'], oof_dict['xgb_tadf']])
    X_rtadf = np.column_stack([oof_dict['dl_rtadf'], oof_dict['xgb_rtadf']])
    y_tadf = oof_dict['y_tadf']
    y_rtadf = oof_dict['y_rtadf']

    st_t = LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000)
    st_r = LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000)
    st_t.fit(X_tadf, y_tadf)
    st_r.fit(X_rtadf, y_rtadf)

    print(f"TADF stacking weights: DL={st_t.coef_[0][0]:.3f}, XGB={st_t.coef_[0][1]:.3f}")
    print(f"rTADF stacking weights: DL={st_r.coef_[0][0]:.3f}, XGB={st_r.coef_[0][1]:.3f}")

    weights = dict(
        tadf=dict(dl=float(st_t.coef_[0][0]), xgb=float(st_t.coef_[0][1])),
        rtadf=dict(dl=float(st_r.coef_[0][0]), xgb=float(st_r.coef_[0][1])),
    )
    return st_t, st_r, weights


# --------------------------- #
# 学阈值（F1 / prevalence / Fbeta）
# --------------------------- #
def _thr_f1(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-10)
    if len(thr) == 0:
        return 0.5
    i = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(thr[i])


def _thr_prevalence(y_prob, target_prev):
    s = np.sort(y_prob)[::-1]
    n_pos = int(len(y_prob) * float(target_prev))
    if n_pos < 0:
        n_pos = 0
    if n_pos >= len(s):
        return 0.5
    return float(s[n_pos])


def _thr_fbeta(y_true, y_prob, beta=2):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-10)
    if len(thr) == 0:
        return 0.5
    i = int(np.argmax(f[:-1])) if len(f) > 1 else 0
    return float(thr[i])


def derive_optimal_thresholds(y_true_t, prob_t, y_true_r, prob_r, beta_r: float = 2.0):
    """
    为 TADF / rTADF 计算多种策略阈值；常用：TADF=F1最优，rTADF=Fbeta(2)。
    """
    thr_t = {
        'f1': _thr_f1(y_true_t, prob_t),
        'prevalence': _thr_prevalence(prob_t, np.mean(y_true_t)),
        'fbeta2': _thr_fbeta(y_true_t, prob_t, beta=2),
        'fixed': 0.5,
    }
    thr_r = {
        'f1': _thr_f1(y_true_r, prob_r),
        'prevalence': _thr_prevalence(prob_r, np.mean(y_true_r)),
        'fbeta2': _thr_fbeta(y_true_r, prob_r, beta=beta_r),
        'fixed': 0.5,
    }
    return thr_t, thr_r


# --------------------------- #
# 测试集上应用 stacking
# --------------------------- #
def apply_stacking_to_test(
    dl_model,
    xgb_predictor: XGBoostTADFPredictor,
    stacker_tadf: LogisticRegression,
    stacker_rtadf: LogisticRegression,
    X_test: np.ndarray,
    device: str = "cpu",
):
    """
    用训练好的 DL/XGB + stacker 在测试集上出融合概率。
    """
    dev = torch.device(device)
    dl_model.to(dev).eval()

    with torch.no_grad():
        logits_t, logits_r = dl_model(torch.tensor(X_test, dtype=torch.float32, device=dev))
        dl_pt = torch.sigmoid(logits_t).cpu().numpy().ravel()
        dl_pr = torch.sigmoid(logits_r).cpu().numpy().ravel()

    xgb_pt, xgb_pr = xgb_predictor.predict(X_test)

    test_stack_t = np.column_stack([dl_pt, xgb_pt])
    test_stack_r = np.column_stack([dl_pr, xgb_pr])

    ens_pt = stacker_tadf.predict_proba(test_stack_t)[:, 1]
    ens_pr = stacker_rtadf.predict_proba(test_stack_r)[:, 1]
    return ens_pt, ens_pr


# --------------------------- #
# 一键式：设计特征 OOF-stacking 流水线
# --------------------------- #
def run_design_oof_stacking_pipeline(
    preprocessor,
    feature_cols=None,
    n_splits: int = 5,
    dl_epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "outputs",
):
    """
    端到端：
      1) 选择特征集合（若显式给出 feature_cols 则强制使用；否则自动生成设计安全特征）
      2) 产生 OOF 预测（DL 含类权重 + 早停）
      3) 学 stacking 权重（LR）
      4) 用 OOF 融合概率学阈值
    """
    os.makedirs(save_dir, exist_ok=True)
    set_global_seed(seed)

    # 选择特征
    if feature_cols is not None and len(feature_cols) > 0:
        # 先与当前 df 做交集
        design_features = [c for c in feature_cols if c in preprocessor.df.columns]
        # 再过滤为纯数值列（双保险）
        num_cols = preprocessor.df[design_features].select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < len(design_features):
            dropped = [c for c in design_features if c not in num_cols]
            print(f"[Design OOF-Stacking] Dropping {len(dropped)} non-numeric user-specified features:")
            print(f"  Examples: {dropped[:10]}")
        design_features = num_cols
        print(f"[Design OOF-Stacking] Using user-specified {len(design_features)} numeric features.")
    else:
        design_features = build_design_feature_list(preprocessor.df, preprocessor.feature_cols)
        # 同样只留数值列
        num_cols = preprocessor.df[design_features].select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < len(design_features):
            dropped = [c for c in design_features if c not in num_cols]
            print(f"[Design OOF-Stacking] Auto feature list contained {len(dropped)} non-numeric columns, dropping:")
            print(f"  Examples: {dropped[:10]}")
        design_features = num_cols
        print(f"[Design OOF-Stacking] Using {len(design_features)} design features (auto).")

    # 产生 OOF 预测
    oof = generate_oof_predictions(
        preprocessor.df,
        feature_cols=design_features,
        n_splits=n_splits,
        dl_epochs=dl_epochs,
        seed=seed,
        device=device,
    )

    # 用 OOF 概率先学 LR-stacking
    st_t, st_r, weights = fit_stacking_models(oof, seed=seed)

    # 计算 OOF 的融合概率
    oof_ens_t = st_t.predict_proba(
        np.column_stack([oof['dl_tadf'], oof['xgb_tadf']])
    )[:, 1]
    oof_ens_r = st_r.predict_proba(
        np.column_stack([oof['dl_rtadf'], oof['xgb_rtadf']])
    )[:, 1]

    thr_t, thr_r = derive_optimal_thresholds(
        oof['y_tadf'], oof_ens_t, oof['y_rtadf'], oof_ens_r, beta_r=2.0
    )

    # 可选：保存
    import joblib, json
    joblib.dump(st_t, os.path.join(save_dir, 'stacker_tadf_design.pkl'))
    joblib.dump(st_r, os.path.join(save_dir, 'stacker_rtadf_design.pkl'))
    with open(os.path.join(save_dir, 'stacker_weights_design.json'), 'w') as f:
        json.dump(weights, f, indent=2)
    with open(os.path.join(save_dir, 'stacker_thresholds_design.json'), 'w') as f:
        json.dump({'tadf': thr_t, 'rtadf': thr_r}, f, indent=2)

    return dict(
        design_features=design_features,
        oof_dict=oof,
        stacker_tadf=st_t,
        stacker_rtadf=st_r,
        thresholds_tadf=thr_t,
        thresholds_rtadf=thr_r,
        weights=weights,
    )
