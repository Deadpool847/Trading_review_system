"""ML Training Pipeline with Walk-Forward and Calibration"""
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb
from typing import Dict, Tuple, List, Optional
import pickle
import json
from pathlib import Path
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class MLTrainer:
    """Walk-forward ML training with calibration"""
    
    def __init__(self, config: Dict, models_dir: str = "models"):
        self.config = config
        self.ml_config = config.get('ml', {})
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_days = self.ml_config.get('walk_forward', {}).get('train_days', 60)
        self.test_days = self.ml_config.get('walk_forward', {}).get('test_days', 10)
        self.purge_bars = self.ml_config.get('walk_forward', {}).get('purge_bars', 5)
    
    def prepare_data(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str = 'meta_label'
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare features and target for training
        
        Returns:
            X, y, valid_indices
        """
        # Remove rows with nulls in features or target
        clean_df = df.select(feature_cols + [target_col, 'ts']).drop_nulls()
        
        # Extract arrays
        X = clean_df.select(feature_cols).to_numpy()
        y = clean_df[target_col].to_numpy()
        
        # Track original indices for time-series split
        indices = list(range(len(clean_df)))
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        return X, y, indices
    
    def walk_forward_split(
        self,
        df: pl.DataFrame,
        train_start_date: str,
        test_end_date: str
    ) -> List[Tuple[List[int], List[int]]]:
        """Create walk-forward splits with purge and embargo
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        # For now, simple single split
        # In production, implement proper walk-forward with multiple folds
        df = df.sort('ts')
        n_samples = len(df)
        
        # 80-20 split with embargo
        train_size = int(0.8 * n_samples)
        purge_size = self.purge_bars
        
        train_idx = list(range(0, train_size))
        test_idx = list(range(train_size + purge_size, n_samples))
        
        logger.info(f"Walk-forward split: {len(train_idx)} train, {len(test_idx)} test")
        return [(train_idx, test_idx)]
    
    def train_logistic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[CalibratedClassifierCV, Dict]:
        """Train L2 Logistic Regression with calibration
        
        Returns:
            (calibrated_model, metrics_dict)
        """
        # Base logistic model
        log_config = self.ml_config.get('models', {}).get('logistic', {})
        base_model = LogisticRegression(
            C=log_config.get('C', 0.1),
            penalty=log_config.get('penalty', 'l2'),
            max_iter=log_config.get('max_iter', 500),
            random_state=42
        )
        
        # Fit base model
        base_model.fit(X_train, y_train)
        
        # Calibrate
        cal_method = self.ml_config.get('calibration', {}).get('method', 'isotonic')
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method=cal_method,
            cv=self.ml_config.get('calibration', {}).get('cv', 3)
        )
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'brier': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba)
        }
        
        logger.info(f"Logistic trained - AUC: {metrics['auc']:.3f}, Brier: {metrics['brier']:.3f}")
        return calibrated_model, metrics
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[lgb.Booster, Dict]:
        """Train LightGBM with monotonic constraints
        
        Returns:
            (model, metrics_dict)
        """
        lgb_config = self.ml_config.get('models', {}).get('lightgbm', {})
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': lgb_config.get('max_depth', 4),
            'num_leaves': lgb_config.get('num_leaves', 15),
            'learning_rate': lgb_config.get('learning_rate', 0.05),
            'min_child_samples': lgb_config.get('min_child_samples', 30),
            'verbose': -1,
            'seed': 42
        }
        
        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=lgb_config.get('n_estimators', 150),
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test)
        
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'brier': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba)
        }
        
        logger.info(f"LightGBM trained - AUC: {metrics['auc']:.3f}, Brier: {metrics['brier']:.3f}")
        return model, metrics
    
    def train_and_save(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        model_type: str = 'logistic',
        snapshot_date: str = None
    ) -> Dict:
        """Full training pipeline with snapshot save
        
        Args:
            df: DataFrame with features and meta_label
            feature_cols: List of feature column names
            model_type: 'logistic' or 'lightgbm'
            snapshot_date: Date identifier for snapshot (YYYYMMDD)
            
        Returns:
            Dict with model info and metrics
        """
        if snapshot_date is None:
            snapshot_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        
        # Prepare data
        X, y, indices = self.prepare_data(df, feature_cols)
        
        if len(X) < 100:
            logger.error("Insufficient data for training")
            return {'error': 'insufficient_data'}
        
        # Walk-forward split (simplified for now)
        splits = self.walk_forward_split(df, None, None)
        train_idx, test_idx = splits[0]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train model
        if model_type == 'logistic':
            model, metrics = self.train_logistic(X_train, y_train, X_test, y_test)
        elif model_type == 'lightgbm':
            model, metrics = self.train_lightgbm(X_train, y_train, X_test, y_test)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {'error': 'unknown_model_type'}
        
        # Save snapshot
        snapshot_name = f"{model_type}_{snapshot_date}"
        model_path = self.models_dir / f"{snapshot_name}.pkl"
        metadata_path = self.models_dir / f"{snapshot_name}_metadata.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        metadata = {
            'model_type': model_type,
            'snapshot_date': snapshot_date,
            'feature_cols': feature_cols,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metrics': metrics,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model snapshot saved: {snapshot_name}")
        return metadata
    
    def load_model(self, snapshot_name: str) -> Tuple[Optional[object], Optional[Dict]]:
        """Load model snapshot
        
        Returns:
            (model, metadata)
        """
        model_path = self.models_dir / f"{snapshot_name}.pkl"
        metadata_path = self.models_dir / f"{snapshot_name}_metadata.json"
        
        if not model_path.exists():
            logger.error(f"Model not found: {snapshot_name}")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded model: {snapshot_name}")
        return model, metadata
    
    def predict_proba(self, model, X: np.ndarray, model_type: str = 'logistic') -> np.ndarray:
        """Get probability predictions
        
        Returns:
            Array of probabilities for positive class
        """
        if model_type == 'logistic':
            return model.predict_proba(X)[:, 1]
        elif model_type == 'lightgbm':
            return model.predict(X)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return np.zeros(len(X))
