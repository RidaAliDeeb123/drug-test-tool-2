import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple, Optional, List
import joblib
import logging

logger = logging.getLogger(__name__)

class DrugResponsePredictor:
    """
    Machine learning model for predicting gender-specific drug responses.
    Uses ensemble of models to predict:
    - Drug effectiveness
    - Adverse event risk
    - Gender-specific response patterns
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the drug response predictor.
        
        Args:
            model_dir: Directory to save/load trained models
        """
        self.model_dir = model_dir
        self.models = {
            'effectiveness': None,
            'adverse_event': None,
            'gender_response': None
        }
        self.scaler = None
        self.label_encoders = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess the input data for model training/prediction.
        
        Args:
            df: Input DataFrame containing patient and drug information
            
        Returns:
            Tuple of (processed DataFrame, preprocessing metadata)
        """
        # Handle missing values
        df = df.fillna({
            'age': df['age'].mean(),
            'dosage': df['dosage'].mean()
        })
        
        # Convert categorical variables
        for col in ['gender', 'drug']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_cols = ['age', 'dosage']
        if any(col in df.columns for col in numerical_cols):
            self.scaler = StandardScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df, {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
    
    def train_models(self, df: pd.DataFrame, target_col: str = 'response'):
        """
        Train machine learning models for drug response prediction.
        
        Args:
            df: Training DataFrame
            target_col: Name of the target column
        """
        # Split data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train effectiveness model
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc')
        rf_grid.fit(X_train, y_train)
        self.models['effectiveness'] = rf_grid.best_estimator_
        
        # Train adverse event model
        svc_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        svc = SVC(probability=True, random_state=42)
        svc_grid = GridSearchCV(svc, svc_params, cv=3, scoring='roc_auc')
        svc_grid.fit(X_train, y_train)
        self.models['adverse_event'] = svc_grid.best_estimator_
        
        # Train gender-specific response model
        gender_rf = RandomForestClassifier(random_state=42)
        gender_rf.fit(X_train, y_train)
        self.models['gender_response'] = gender_rf
        
        # Evaluate models
        for model_name, model in self.models.items():
            if model is not None:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                logger.info(f"\n{model_name} model evaluation:")
                logger.info(classification_report(y_test, y_pred))
                logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
                logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
    def predict(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Make predictions using trained models.
        
        Args:
            df: Input DataFrame for prediction
            
        Returns:
            Dictionary of predictions for each model
        """
        if not all(self.models.values()):
            raise ValueError("Models not trained yet. Please train models first.")
            
        # Preprocess data
        df_processed, _ = self.preprocess_data(df)
        
        predictions = {}
        for model_name, model in self.models.items():
            if model is not None:
                pred = model.predict(df_processed)
                prob = model.predict_proba(df_processed)[:, 1]
                predictions[model_name] = {
                    'prediction': pred,
                    'probability': prob,
                    'feature_importance': self._get_feature_importance(model, df_processed)
                }
        
        return predictions
    
    def _get_feature_importance(self, model: Any, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model: Trained model
            df: Input DataFrame
            
        Returns:
            Dictionary of feature importances
        """
        if hasattr(model, 'feature_importances_'):
            return dict(zip(df.columns, model.feature_importances_))
        return {}
    
    def save_models(self):
        """
        Save trained models to disk.
        """
        for model_name, model in self.models.items():
            if model is not None:
                model_path = f"{self.model_dir}/{model_name}_model.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save preprocessing objects
        if self.scaler:
            joblib.dump(self.scaler, f"{self.model_dir}/scaler.joblib")
        for col, le in self.label_encoders.items():
            joblib.dump(le, f"{self.model_dir}/{col}_encoder.joblib")
    
    def load_models(self):
        """
        Load trained models from disk.
        """
        for model_name in self.models.keys():
            model_path = f"{self.model_dir}/{model_name}_model.joblib"
            try:
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load preprocessing objects
        try:
            self.scaler = joblib.load(f"{self.model_dir}/scaler.joblib")
            for col in ['gender', 'drug']:
                le_path = f"{self.model_dir}/{col}_encoder.joblib"
                if os.path.exists(le_path):
                    self.label_encoders[col] = joblib.load(le_path)
        except FileNotFoundError:
            logger.warning("Preprocessing objects not found")

# Create instance for global use
response_predictor = DrugResponsePredictor()
