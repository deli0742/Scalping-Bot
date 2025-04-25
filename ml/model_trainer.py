from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV
from ta import add_all_ta_features
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = {}

    def train_with_hyperopt(self, X, y):
        params = {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10)
        }
        
        opt = BayesSearchCV(
            GradientBoostingClassifier(),
            params,
            n_iter=50,
            cv=3,
            scoring='f1_weighted'
        )
        
        opt.fit(X, y)
        self.model = opt.best_estimator_
        self.best_params = opt.best_params_
        
        return opt.best_score_

    def adaptive_retraining(self, new_data):
        # Online learning için incremental training
        self.model.partial_fit(new_data)