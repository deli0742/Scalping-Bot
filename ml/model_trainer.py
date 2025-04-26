from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV
from ta import add_all_ta_features
import pandas as pd
import joblib  # pip install joblib

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
            scoring='f1_weighted',
            n_jobs=-1
        )

        opt.fit(X, y)
        self.model = opt.best_estimator_
        self.best_params = opt.best_params_

        return opt.best_score_

    def adaptive_retraining(self, new_data):
        # Online learning için incremental training
        # new_data: (X_new, y_new) tuple veya DataFrame
        X_new, y_new = new_data
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
        else:
            # Eğer partial_fit yoksa, yeniden fit
            self.model.fit(X_new, y_new)

    def save_model(self, path: str):
        """Eğitilmiş modeli ve parametreleri disk’e kaydeder."""
        if self.model is None:
            raise RuntimeError("Model henüz eğitilmedi, önce train_with_hyperopt çalıştırın.")
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)
        print(f"[✓] Model kaydedildi: {path}")


if __name__ == "__main__":
    # Eğitim verisini yükle
    df = pd.read_csv("data/historical_features_labels.csv")

    # Tarih sütununu sayısala dönüştür veya at
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month

    # Özellik ve etiketleri ayır
    X = df.drop(columns=['date', 'label'])
    y = df['label']

    # Model eğitim ve kaydetme
    trainer = ModelTrainer()
    print("[*] Hiperparametre optimizasyonu başlıyor...")
    best_score = trainer.train_with_hyperopt(X, y)
    print(f"[✓] Eğitim tamamlandı. En iyi F1 skoru: {best_score:.4f}")
    print(f"[✓] En iyi parametreler: {trainer.best_params}")

    output_path = "adaptive_model.pkl"
    trainer.save_model(output_path)
