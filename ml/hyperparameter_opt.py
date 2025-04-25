import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import mlflow
import joblib
import warnings
from typing import Dict, Any, Tuple

warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    def __init__(self, experiment_name="hyperparam_opt"):
        self.experiment_name = experiment_name
        self.best_params = {}
        self.best_score = 0
        mlflow.set_experiment(experiment_name)

    def custom_scorer(self, y_true, y_pred) -> float:
        """Scalping için özel metrik (precision ağırlıklı)"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true != 1))
        fn = np.sum((y_pred != 1) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        # Scalping için precision daha önemli
        return 0.7 * precision + 0.3 * recall

    def get_search_spaces(self, model_type: str = 'xgb') -> Dict[str, Any]:
        """Model türüne göre hiperparametre aralıkları"""
        common_spaces = {
            'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 12),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'min_child_weight': Integer(1, 10),
            'gamma': Real(0, 0.5)
        }
        
        if model_type == 'xgb':
            return {
                **common_spaces,
                'n_estimators': Integer(50, 500),
                'reg_alpha': Real(0, 1),
                'reg_lambda': Real(0, 1)
            }
        elif model_type == 'lgbm':
            return {
                **common_spaces,
                'num_leaves': Integer(20, 100),
                'bagging_freq': Integer(1, 10)
            }
        else:  # catboost
            return {
                **common_spaces,
                'iterations': Integer(50, 500),
                'depth': Integer(4, 10),
                'l2_leaf_reg': Real(1, 10)
            }

    def optimize(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'xgb', 
                n_iter: int = 50, cv_splits: int = 3) -> Tuple[Dict, float]:
        """Bayesian optimizasyon ile hiperparametre arama"""
        
        # Model seçimi
        if model_type == 'xgb':
            model = XGBClassifier(objective='binary:logistic', n_jobs=-1)
        elif model_type == 'lgbm':
            model = LGBMClassifier(objective='binary', n_jobs=-1)
        else:
            model = CatBoostClassifier(silent=True, task_type='GPU')
        
        # TimeSeries CV
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Optimizasyon
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=self.get_search_spaces(model_type),
            n_iter=n_iter,
            cv=tscv,
            scoring=make_scorer(self.custom_scorer),
            verbose=1,
            n_jobs=-1
        )
        
        with mlflow.start_run():
            opt.fit(X, y)
            
            # MLflow logging
            mlflow.log_params(opt.best_params_)
            mlflow.log_metric("best_score", opt.best_score_)
            mlflow.sklearn.log_model(opt.best_estimator_, "model")
            
            self.best_params = opt.best_params_
            self.best_score = opt.best_score_
            
            # Modeli kaydet
            joblib.dump(opt.best_estimator_, f'best_{model_type}_model.pkl')
        
        return opt.best_params_, opt.best_score_

    def adaptive_reoptimization(self, X: pd.DataFrame, y: pd.Series, 
                              previous_params: Dict, model_type: str = 'xgb') -> Dict:
        """Mevcut parametreler etrafında lokal optimizasyon"""
        narrowed_spaces = {}
        
        for param, value in previous_params.items():
            if isinstance(value, float):
                narrowed_spaces[param] = Real(
                    max(value * 0.8, 0.001), 
                    min(value * 1.2, 1.0)
            elif isinstance(value, int):
                narrowed_spaces[param] = Integer(
                    max(value - 2, 1), 
                    value + 2)
        
        model = XGBClassifier(**previous_params) if model_type == 'xgb' else \
                LGBMClassifier(**previous_params) if model_type == 'lgbm' else \
                CatBoostClassifier(**previous_params)
        
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=narrowed_spaces,
            n_iter=20,
            cv=TimeSeriesSplit(n_splits=3),
            scoring=make_scorer(self.custom_scorer),
            verbose=1
        )
        
        opt.fit(X, y)
        return opt.best_params_, opt.best_score_