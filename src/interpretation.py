import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

class ModelInterpreter:
    """模型解释器"""
    
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
    
    def shap_analysis(self, X_test, task='tadf'):
        """SHAP分析"""
        # 确保特征名称与数据维度匹配
        n_features = X_test.shape[1]
        
        if self.feature_names is None or len(self.feature_names) != n_features:
            print(f"Warning: Feature names mismatch. Expected {n_features}, got {len(self.feature_names) if self.feature_names else 0}")
            # 创建默认特征名称
            feature_names_to_use = [f'Feature_{i}' for i in range(n_features)]
        else:
            feature_names_to_use = self.feature_names
        
        # 创建SHAP解释器
        try:
            if hasattr(self.model, 'predict_proba'):
                # XGBoost模型
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test)
            else:
                # 神经网络模型
                explainer = shap.DeepExplainer(self.model, self.X_train[:min(100, len(self.X_train))])
                shap_values = explainer.shap_values(X_test)
            
            # 可视化
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, 
                            feature_names=feature_names_to_use, 
                            show=False)
            plt.title(f'SHAP Feature Importance - {task.upper()}')
            plt.tight_layout()
            plt.savefig(f'shap_{task}.png', dpi=300)
            plt.close()  # 改为close避免显示
            print(f"SHAP plot saved as shap_{task}.png")
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            shap_values = None
        
        return shap_values
    
    def plot_feature_importance(self, importance_scores, top_n=20, task='tadf'):
        """绘制特征重要性"""
        # 确保不超过实际特征数量
        n_features = len(importance_scores)
        top_n = min(top_n, n_features)
        
        # 排序并选择top特征
        indices = np.argsort(importance_scores)[-top_n:]
        
        # 确保feature_names长度正确
        if self.feature_names is None or len(self.feature_names) != n_features:
            feature_names_to_use = [f'Feature_{i}' for i in range(n_features)]
        else:
            feature_names_to_use = self.feature_names
        
        top_features = [feature_names_to_use[i] for i in indices]
        top_scores = importance_scores[indices]
        
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top Feature Importances - {task.upper()}') 
        plt.tight_layout()
        # 使用不同的文件名
        filename = f'feature_importance_{task}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Feature importance plot saved as {filename}")
        
    
    def plot_confusion_matrices(self, y_true, y_pred, labels=['Non-TADF', 'TADF'], task='tadf'):
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签名称
            task: 任务名称
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {task.upper()}')
        plt.tight_layout()
        
        filename = f'confusion_matrix_{task}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Confusion matrix saved as {filename}")