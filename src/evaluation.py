# src/evaluation.py
from sklearn.metrics import (
    classification_report, roc_curve, auc, 
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef
)
import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device='cpu'):
    """独立的模型评估函数"""
    model.eval()
    model.to(device)
    
    predictions = {'tadf': [], 'rtadf': []}
    labels = {'tadf': [], 'rtadf': []}
    
    with torch.no_grad():
        for X_batch, y_tadf, y_rtadf in test_loader:
            X_batch = X_batch.to(device)
            tadf_logits, rtadf_logits = model(X_batch)
            
            # 转换logits为概率
            tadf_probs = torch.sigmoid(tadf_logits)
            rtadf_probs = torch.sigmoid(rtadf_logits)
            
            predictions['tadf'].extend(tadf_probs.cpu().numpy())
            predictions['rtadf'].extend(rtadf_probs.cpu().numpy())
            labels['tadf'].extend(y_tadf.numpy())
            labels['rtadf'].extend(y_rtadf.numpy())
    
    results = {}
    for task in ['tadf', 'rtadf']:
        results[task] = calculate_metrics(
            np.array(labels[task]), 
            np.array(predictions[task]), 
            task_name=task.upper()
        )
    
    return results

def calculate_metrics(y_true, y_pred_proba, task_name='TADF', threshold=0.5):
    """计算详细的评估指标"""
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # 基础指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        f'{task_name}_accuracy': accuracy_score(y_true, y_pred),
        f'{task_name}_precision': precision,
        f'{task_name}_recall': recall,
        f'{task_name}_f1': f1,
        f'{task_name}_auc': roc_auc,
        f'{task_name}_mcc': matthews_corrcoef(y_true, y_pred),
        f'{task_name}_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        f'{task_name}_sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'optimal_threshold': thresholds[np.argmax(tpr - fpr)]
    }
    
    return metrics

def plot_roc_curves(results, save_path='outputs/figures/roc_curves.png'):
    """绘制ROC曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, task in enumerate(['tadf', 'rtadf']):
        fpr = results[task]['roc_curve']['fpr']
        tpr = results[task]['roc_curve']['tpr']
        auc_score = results[task][f'{task.upper()}_auc']
        
        axes[idx].plot(fpr, tpr, 'b-', lw=2, 
                      label=f'AUC = {auc_score:.3f}')
        axes[idx].plot([0, 1], [0, 1], 'r--', lw=1)
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'{task.upper()} ROC Curve')
        axes[idx].legend(loc='lower right')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def print_evaluation_report(results):
    """打印评估报告"""
    for task, metrics in results.items():
        print(f"\n{'='*50}")
        print(f"{task.upper()} Evaluation Results")
        print('='*50)
        
        for key, value in metrics.items():
            if not isinstance(value, dict) and not isinstance(value, list):
                print(f"{key}: {value:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm['tn']}  FP: {cm['fp']}")
        print(f"  FN: {cm['fn']}  TP: {cm['tp']}")