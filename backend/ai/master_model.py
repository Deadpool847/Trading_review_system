"""Master AI Model - Ensemble of Advanced ML/DL Models"""
import numpy as np
import polars as pl
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime

# ML/DL imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = logging.getLogger(__name__)

class MasterModel:
    """Advanced ensemble model for stock prediction"""
    
    def __init__(self, symbol: str, models_dir: str = "models"):
        self.symbol = symbol
        self.models_dir = Path(models_dir) / symbol
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensemble components
        self.models = {}
        self.scaler = StandardScaler()
        self.metadata = {}
        
        # Hyperparameters
        self.lookback = 60  # Use last 60 bars
        self.forecast_horizon = 30  # Predict next 30 bars
    
    def train(self, df: pl.DataFrame, feature_cols: list) -> Dict:
        """Train ensemble of models
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            
        Returns:
            Training metrics
        """
        logger.info(f"🎓 Training models for {self.symbol}...")
        
        # Prepare sequences
        X, y = self._prepare_sequences(df, feature_cols)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples (need 100+)")
        
        # Train-test split (80-20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        metrics = {}
        
        # 1. LightGBM (fast, accurate)
        logger.info("  Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        lgb_pred = lgb_model.predict(X_test_scaled)
        metrics['lgb_rmse'] = np.sqrt(np.mean((lgb_pred - y_test) ** 2))
        self.models['lgb'] = lgb_model
        
        # 2. Gradient Boosting
        logger.info("  Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        metrics['gb_rmse'] = np.sqrt(np.mean((gb_pred - y_test) ** 2))
        self.models['gb'] = gb_model
        
        # 3. Random Forest
        logger.info("  Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        metrics['rf_rmse'] = np.sqrt(np.mean((rf_pred - y_test) ** 2))
        self.models['rf'] = rf_model
        
        # 4. Ridge Regression (baseline)
        logger.info("  Training Ridge Regression...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_scaled, y_train)
        ridge_pred = ridge_model.predict(X_test_scaled)
        metrics['ridge_rmse'] = np.sqrt(np.mean((ridge_pred - y_test) ** 2))
        self.models['ridge'] = ridge_model
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (
            0.4 * lgb_pred +
            0.3 * gb_pred +
            0.2 * rf_pred +
            0.1 * ridge_pred
        )
        metrics['ensemble_rmse'] = np.sqrt(np.mean((ensemble_pred - y_test) ** 2))
        
        # Metadata
        self.metadata = {
            'symbol': self.symbol,
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'metrics': metrics,
            'feature_cols': feature_cols
        }
        
        # Save
        self.save()
        
        logger.info(f"✅ Training complete! Ensemble RMSE: {metrics['ensemble_rmse']:.4f}")
        return metrics
    
    def predict(self, df: pl.DataFrame, feature_cols: list, n_steps: int = 30) -> np.ndarray:
        """Predict future prices
        
        Args:
            df: Recent data
            feature_cols: Feature columns
            n_steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        # Get last sequence
        X_last = self._get_last_sequence(df, feature_cols)
        X_last_scaled = self.scaler.transform(X_last)
        
        predictions = []
        
        for _ in range(n_steps):
            # Ensemble prediction
            pred = (
                0.4 * self.models['lgb'].predict(X_last_scaled)[0] +
                0.3 * self.models['gb'].predict(X_last_scaled)[0] +
                0.2 * self.models['rf'].predict(X_last_scaled)[0] +
                0.1 * self.models['ridge'].predict(X_last_scaled)[0]
            )
            
            predictions.append(pred)
            
            # Update sequence (simplified - in production, update full feature set)
            # For now, just shift and append
            X_last_scaled = np.roll(X_last_scaled, -1, axis=1)
        
        return np.array(predictions)
    
    def _prepare_sequences(self, df: pl.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        # Get feature matrix and replace NaN/inf
        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        targets = df['close'].to_numpy()
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        X, y = [], []
        
        for i in range(self.lookback, len(features) - self.forecast_horizon):
            # Input: last lookback bars
            X.append(features[i-self.lookback:i].flatten())
            
            # Target: next bar's close
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Final NaN check
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y
    
    def _get_last_sequence(self, df: pl.DataFrame, feature_cols: list) -> np.ndarray:
        """Get last sequence for prediction"""
        features = df.select(feature_cols).tail(self.lookback).to_numpy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.flatten().reshape(1, -1)
    
    def save(self):
        """Save models and metadata"""
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, self.models_dir / f"{name}.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        
        # Save metadata
        with open(self.models_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"💾 Models saved to {self.models_dir}")
    
    def load(self) -> bool:
        """Load saved models"""
        try:
            # Load models
            for name in ['lgb', 'gb', 'rf', 'ridge']:
                self.models[name] = joblib.load(self.models_dir / f"{name}.pkl")
            
            # Load scaler
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            
            # Load metadata
            with open(self.models_dir / "metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"✅ Models loaded for {self.symbol}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load models: {e}")
            return False
