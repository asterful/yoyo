import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

class OptunaOptimizer:
    def __init__(self, model_class, validator_class, train_data: pd.DataFrame, target_col: str = "target_tb_24h", n_splits: int = 5):
        self.model_class = model_class
        self.validator_class = validator_class
        self.train_data = train_data
        self.target_col = target_col
        self.n_splits = n_splits


    def optimize(self, n_trials: int = 30) -> dict:
        """
        Runs Optuna study maximizing Class 2 F1-score (balancing precision & recall) 
        across walk-forward training folds.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            
            model = self.model_class(params=params)
            validator = self.validator_class(model=model, data=self.train_data, n_splits=self.n_splits)
            results = validator.run(target_col=self.target_col)
            
            fold_f1_scores = []
            for res in results:
                y_true = res["y_true"]
                y_pred = res["y_pred"]
                
                prec, rec, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=[2], average=None, zero_division=0
                )
                
                class_2_f1 = f1[0] if len(f1) > 0 else 0.0
                fold_f1_scores.append(class_2_f1)
                
            return np.mean(fold_f1_scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"Starting Optuna optimization ({n_trials} trials) targeting Class 2 F1-Score...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"Optimization complete! Best Class-2 F1-Score: {study.best_value * 100:.2f}%")
        
        return study.best_params


    def evaluate_cv_folds(self, best_params: dict):
        """
        Re-runs the validator using the best parameters, pools all CV fold predictions,
        and prints a detailed performance report for the training period.
        """
        model = self.model_class(params=best_params)
        validator = self.validator_class(model=model, data=self.train_data, n_splits=self.n_splits)
        results = validator.run(target_col=self.target_col)
        
        # Concatenate all fold true and predicted labels together
        all_y_true = []
        all_y_pred = []
        
        for res in results:
            all_y_true.append(res["y_true"])
            all_y_pred.append(res["y_pred"])
            
        y_true_pooled = pd.concat(all_y_true)
        y_pred_pooled = pd.concat(all_y_pred)
        
        self.print_detailed_report(y_true_pooled, y_pred_pooled, title="TRAIN PERIOD (WALK-FORWARD CV) REPORT")


    @staticmethod
    def print_detailed_report(y_true, y_pred, title="MODEL PERFORMANCE REPORT"):
        """
        Generates a clean, human-readable table of metrics across all classes.
        """
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report_dict).transpose()
        
        df_report = df_report.rename(columns={
            'precision': 'Precision',
            'recall': 'Recall',
            'f1-score': 'F1-Score',
            'support': 'Support'
        })
        
        print("\n" + "=" * 65)
        print(f" {title} ")
        print("=" * 65)
        
        display_df = df_report.copy()
        for col in ['Precision', 'Recall', 'F1-Score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")
        display_df['Support'] = display_df['Support'].astype(int)
        
        print(display_df.to_string())
        print("=" * 65 + "\n")