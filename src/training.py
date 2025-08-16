# src/training.py - 替换整个文件
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_recall_curve, balanced_accuracy_score

class ModelTrainer:
    """模型训练器 - 改进版，支持类别加权和动态阈值"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 tadf_pos_weight=1.0, rtadf_pos_weight=1.0):
        self.model = model.to(device)
        self.device = device
        
        # 使用加权损失处理类别不平衡
        self.tadf_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(tadf_pos_weight, device=device)
        )
        self.rtadf_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(rtadf_pos_weight, device=device)
        )
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5  # 改为监控PR-AUC（越大越好）
        )
        
    def find_optimal_threshold(self, y_true, y_prob):
        """找到F1最大的阈值"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = np.nanargmax(f1_scores)
        best_threshold = thresholds[max(best_idx-1, 0)]
        best_f1 = f1_scores[best_idx]
        return best_threshold, best_f1
        
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_tadf_batch, y_rtadf_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_tadf_batch = y_tadf_batch.to(self.device)
            y_rtadf_batch = y_rtadf_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播（注意：现在输出是logits，不是概率）
            tadf_logits, rtadf_logits = self.model(X_batch)
            
            # 计算加权损失
            loss_tadf = self.tadf_criterion(tadf_logits.squeeze(), y_tadf_batch)
            loss_rtadf = self.rtadf_criterion(rtadf_logits.squeeze(), y_rtadf_batch)
            loss = 0.5 * loss_tadf + 0.5 * loss_rtadf
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, use_optimal_threshold=True):
        """模型评估 - 支持动态阈值"""
        self.model.eval()
        
        all_tadf_probs = []
        all_rtadf_probs = []
        all_tadf_labels = []
        all_rtadf_labels = []
        
        with torch.no_grad():
            for X_batch, y_tadf_batch, y_rtadf_batch in dataloader:
                X_batch = X_batch.to(self.device)
                
                tadf_logits, rtadf_logits = self.model(X_batch)
                
                # 转换为概率
                tadf_probs = torch.sigmoid(tadf_logits)
                rtadf_probs = torch.sigmoid(rtadf_logits)
                
                all_tadf_probs.extend(tadf_probs.cpu().numpy())
                all_rtadf_probs.extend(rtadf_probs.cpu().numpy())
                all_tadf_labels.extend(y_tadf_batch.numpy())
                all_rtadf_labels.extend(y_rtadf_batch.numpy())
        
        # 计算指标
        metrics = self.calculate_metrics(
            all_tadf_probs, all_rtadf_probs,
            all_tadf_labels, all_rtadf_labels,
            use_optimal_threshold
        )
        
        return metrics
    
    def calculate_metrics(self, tadf_probs, rtadf_probs, tadf_labels, rtadf_labels, 
                         use_optimal_threshold=True):
        """计算评估指标 - 包括PR-AUC和最优阈值"""
        tadf_probs = np.array(tadf_probs).flatten()
        rtadf_probs = np.array(rtadf_probs).flatten()
        tadf_labels = np.array(tadf_labels)
        rtadf_labels = np.array(rtadf_labels)
        
        metrics = {}
        
        # TADF指标
        if len(np.unique(tadf_labels)) > 1:
            metrics['tadf_auc'] = roc_auc_score(tadf_labels, tadf_probs)
            metrics['tadf_pr_auc'] = average_precision_score(tadf_labels, tadf_probs)
            
            if use_optimal_threshold:
                tadf_threshold, tadf_f1 = self.find_optimal_threshold(tadf_labels, tadf_probs)
                tadf_preds = (tadf_probs >= tadf_threshold).astype(int)
                metrics['tadf_optimal_threshold'] = tadf_threshold
            else:
                tadf_preds = (tadf_probs >= 0.5).astype(int)
                tadf_f1 = f1_score(tadf_labels, tadf_preds)
            
            metrics['tadf_f1'] = tadf_f1
            metrics['tadf_accuracy'] = accuracy_score(tadf_labels, tadf_preds)
            metrics['tadf_balanced_accuracy'] = balanced_accuracy_score(tadf_labels, tadf_preds)
        else:
            metrics['tadf_auc'] = 0.5
            metrics['tadf_pr_auc'] = 0.0
            metrics['tadf_f1'] = 0.0
            metrics['tadf_accuracy'] = 0.0
        
        # rTADF指标
        if len(np.unique(rtadf_labels)) > 1:
            metrics['rtadf_auc'] = roc_auc_score(rtadf_labels, rtadf_probs)
            metrics['rtadf_pr_auc'] = average_precision_score(rtadf_labels, rtadf_probs)
            
            if use_optimal_threshold:
                rtadf_threshold, rtadf_f1 = self.find_optimal_threshold(rtadf_labels, rtadf_probs)
                rtadf_preds = (rtadf_probs >= rtadf_threshold).astype(int)
                metrics['rtadf_optimal_threshold'] = rtadf_threshold
            else:
                rtadf_preds = (rtadf_probs >= 0.5).astype(int)
                rtadf_f1 = f1_score(rtadf_labels, rtadf_preds)
            
            metrics['rtadf_f1'] = rtadf_f1
            metrics['rtadf_accuracy'] = accuracy_score(rtadf_labels, rtadf_preds)
            metrics['rtadf_balanced_accuracy'] = balanced_accuracy_score(rtadf_labels, rtadf_preds)
            
            # 添加基线PR-AUC（阳性率）
            metrics['rtadf_baseline_pr_auc'] = rtadf_labels.mean()
        else:
            metrics['rtadf_auc'] = 0.5
            metrics['rtadf_pr_auc'] = 0.0
            metrics['rtadf_f1'] = 0.0
            metrics['rtadf_accuracy'] = 0.0
        
        return metrics