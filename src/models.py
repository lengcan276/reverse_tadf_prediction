# src/models.py
"""
Model definitions for TADF/rTADF prediction
"""

import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np



class MultiTaskTADFNet(nn.Module):
    """多任务TADF预测网络 - 改进版，输出logits"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(MultiTaskTADFNet, self).__init__()
        
        # 共享特征提取层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 任务特定预测头 - 注意：不包含Sigmoid
        self.tadf_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, 1)  # 输出logits，不是概率
        )
        
        self.rtadf_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, 1)  # 输出logits
        )
    
    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_layers(x)
        
        # 任务特定预测 - 输出logits
        tadf_logits = self.tadf_head(shared_features)
        rtadf_logits = self.rtadf_head(shared_features)
        
        return tadf_logits, rtadf_logits
        
    def get_config(self):
        """获取模型配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout
        }


class XGBoostTADFPredictor(BaseEstimator, ClassifierMixin):
    """XGBoost TADF/rTADF预测器"""
    
    def __init__(self, 
                 n_estimators=200,
                 max_depth=5,
                 learning_rate=0.1,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 random_state=42,
                 use_grid_search=True):
        """
        初始化XGBoost预测器
        
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 子采样比例
            colsample_bytree: 特征采样比例
            random_state: 随机种子
            use_grid_search: 是否使用网格搜索
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_grid_search = use_grid_search
        
        self.tadf_model = None
        self.rtadf_model = None
        self.feature_names = None
    
    def _create_base_model(self):
        """创建基础XGBoost模型"""
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            #use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=2
        )
    
    def _get_param_grid(self):
        """获取网格搜索参数"""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    def fit(self, X, y_tadf, y_rtadf, feature_names=None):
        """训练模型"""
        self.feature_names = feature_names
        
        # 检查类别分布
        unique_tadf = np.unique(y_tadf)
        unique_rtadf = np.unique(y_rtadf)
        
        print(f"TADF classes: {unique_tadf}, counts: {np.bincount(y_tadf.astype(int))}")
        print(f"rTADF classes: {unique_rtadf}, counts: {np.bincount(y_rtadf.astype(int))}")
        
        if self.use_grid_search:
            param_grid = self._get_param_grid()
            
            # TADF模型 - 处理单类别情况
            if len(unique_tadf) > 1:
                print("Training TADF model with grid search...")
                self.tadf_model = GridSearchCV(
                    self._create_base_model(),
                    param_grid,
                    cv=3,  # 减少CV折数，因为数据太少
                    scoring='f1',  # 改用F1而不是ROC AUC
                    n_jobs=2,  # 限制并行数
                    verbose=0  # 减少输出
                )
                self.tadf_model.fit(X, y_tadf)
                print(f"Best TADF params: {self.tadf_model.best_params_}")
            else:
                print("Warning: Only one class in TADF labels, using dummy classifier")
                from sklearn.dummy import DummyClassifier
                self.tadf_model = DummyClassifier(strategy='constant', constant=int(unique_tadf[0]))
                self.tadf_model.fit(X, y_tadf)
            
            # rTADF模型
            if len(unique_rtadf) > 1:
                print("Training rTADF model with grid search...")
                self.rtadf_model = GridSearchCV(
                    self._create_base_model(),
                    param_grid,
                    cv=3,
                    scoring='f1',
                    n_jobs=2,
                    verbose=0
                )
                self.rtadf_model.fit(X, y_rtadf)
                print(f"Best rTADF params: {self.rtadf_model.best_params_}")
            else:
                print("Warning: Only one class in rTADF labels, using dummy classifier")
                from sklearn.dummy import DummyClassifier
                self.rtadf_model = DummyClassifier(strategy='constant', constant=int(unique_rtadf[0]))
                self.rtadf_model.fit(X, y_rtadf)
        else:
            # 直接训练时也要处理单类别情况
            if len(unique_tadf) > 1:
                self.tadf_model = self._create_base_model()
                self.tadf_model.fit(X, y_tadf)
            else:
                from sklearn.dummy import DummyClassifier
                self.tadf_model = DummyClassifier(strategy='constant', constant=int(unique_tadf[0]))
                self.tadf_model.fit(X, y_tadf)
            
            if len(unique_rtadf) > 1:
                self.rtadf_model = self._create_base_model()
                self.rtadf_model.fit(X, y_rtadf)
            else:
                from sklearn.dummy import DummyClassifier
                self.rtadf_model = DummyClassifier(strategy='constant', constant=int(unique_rtadf[0]))
                self.rtadf_model.fit(X, y_rtadf)
        
        return self
        
    def train(self, X_train, y_tadf_train, y_rtadf_train):
        """训练方法（兼容旧代码）"""
        return self.fit(X_train, y_tadf_train, y_rtadf_train)
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            tadf_pred: TADF预测概率
            rtadf_pred: rTADF预测概率
        """
        if self.tadf_model is None or self.rtadf_model is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        # 获取实际的模型（处理GridSearchCV的情况）
        tadf_model = (self.tadf_model.best_estimator_ 
                     if hasattr(self.tadf_model, 'best_estimator_') 
                     else self.tadf_model)
        rtadf_model = (self.rtadf_model.best_estimator_ 
                      if hasattr(self.rtadf_model, 'best_estimator_') 
                      else self.rtadf_model)
        
        tadf_pred = tadf_model.predict_proba(X)[:, 1]
        rtadf_pred = rtadf_model.predict_proba(X)[:, 1]
        
        return tadf_pred, rtadf_pred
    
    def predict_proba(self, X):
        """
        预测概率（sklearn兼容接口）
        
        Returns:
            字典包含两个任务的预测概率
        """
        tadf_pred, rtadf_pred = self.predict(X)
        return {
            'tadf': np.column_stack([1-tadf_pred, tadf_pred]),
            'rtadf': np.column_stack([1-rtadf_pred, rtadf_pred])
        }
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            tadf_importance: TADF模型的特征重要性
            rtadf_importance: rTADF模型的特征重要性
        """
        if self.tadf_model is None or self.rtadf_model is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        # 获取实际的模型
        tadf_model = (self.tadf_model.best_estimator_ 
                     if hasattr(self.tadf_model, 'best_estimator_') 
                     else self.tadf_model)
        rtadf_model = (self.rtadf_model.best_estimator_ 
                      if hasattr(self.rtadf_model, 'best_estimator_') 
                      else self.rtadf_model)
        
        tadf_importance = tadf_model.feature_importances_
        rtadf_importance = rtadf_model.feature_importances_
        
        # 如果有特征名称，返回带名称的重要性
        if self.feature_names is not None:
            tadf_importance_dict = dict(zip(self.feature_names, tadf_importance))
            rtadf_importance_dict = dict(zip(self.feature_names, rtadf_importance))
            return tadf_importance_dict, rtadf_importance_dict
        
        return tadf_importance, rtadf_importance
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'use_grid_search': self.use_grid_search
        }
    
    def set_params(self, **params):
        """设置参数（sklearn兼容）"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def score(self, X, y_tadf, y_rtadf):
        """
        计算模型得分
        
        Returns:
            平均AUC分数
        """
        from sklearn.metrics import roc_auc_score
        
        tadf_pred, rtadf_pred = self.predict(X)
        
        tadf_auc = roc_auc_score(y_tadf, tadf_pred)
        rtadf_auc = roc_auc_score(y_rtadf, rtadf_pred)
        
        return (tadf_auc + rtadf_auc) / 2


class EnsemblePredictor:
    """集成预测器，结合深度学习和XGBoost"""
    
    def __init__(self, dl_model=None, xgb_model=None, weights=None):
        """
        初始化集成预测器
        
        Args:
            dl_model: 深度学习模型
            xgb_model: XGBoost模型
            weights: 模型权重 [dl_weight, xgb_weight]
        """
        self.dl_model = dl_model
        self.xgb_model = xgb_model
        self.weights = weights if weights else [0.5, 0.5]
    
    def predict(self, X, device='cpu'):
        """
        集成预测
        
        Args:
            X: 特征矩阵
            device: 计算设备
        """
        predictions = {'tadf': [], 'rtadf': []}
        
        # 深度学习预测
        if self.dl_model is not None:
            self.dl_model.eval()
            self.dl_model.to(device)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                dl_tadf, dl_rtadf = self.dl_model(X_tensor)
                
                predictions['tadf'].append(dl_tadf.cpu().numpy())
                predictions['rtadf'].append(dl_rtadf.cpu().numpy())
        
        # XGBoost预测
        if self.xgb_model is not None:
            xgb_tadf, xgb_rtadf = self.xgb_model.predict(X)
            
            predictions['tadf'].append(xgb_tadf.reshape(-1, 1))
            predictions['rtadf'].append(xgb_rtadf.reshape(-1, 1))
        
        # 加权平均
        tadf_pred = np.average(
            np.hstack(predictions['tadf']), 
            axis=1, 
            weights=self.weights[:len(predictions['tadf'])]
        )
        rtadf_pred = np.average(
            np.hstack(predictions['rtadf']), 
            axis=1, 
            weights=self.weights[:len(predictions['rtadf'])]
        )
        
        return tadf_pred, rtadf_pred