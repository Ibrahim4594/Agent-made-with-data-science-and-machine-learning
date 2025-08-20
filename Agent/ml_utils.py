import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import json
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MLUtils:
    """Utility class for machine learning operations"""
    
    def __init__(self):
        """Initialize ML utilities"""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def detect_problem_type(self, target_column: pd.Series) -> str:
        """Detect if the problem is classification or regression"""
        if target_column.dtype in ['object', 'category'] or len(target_column.unique()) < 10:
            return 'classification'
        else:
            return 'regression'
    
    def get_available_models(self, problem_type: str) -> Dict[str, Any]:
        """Get available models for the problem type"""
        if problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42),
                'SVM': SVC(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'SVR': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'KNN': KNeighborsRegressor()
            }
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for machine learning"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X = self._encode_categorical_variables(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale numerical features
        X_train, X_test = self._scale_features(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                   problem_type: str, hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Train a machine learning model"""
        try:
            # Get model
            models = self.get_available_models(problem_type)
            if model_name not in models:
                raise ValueError(f"Model {model_name} not available for {problem_type}")
            
            model = models[model_name]
            
            # Hyperparameter tuning if requested
            if hyperparameter_tuning:
                model = self._tune_hyperparameters(model, model_name, problem_type, X_train, y_train)
            else:
                # Train model
                model.fit(X_train, y_train)
            
            # Store model
            self.models[model_name] = model
            
            return {
                "model_name": model_name,
                "problem_type": problem_type,
                "status": "trained",
                "model": model
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, 
                      problem_type: str) -> Dict[str, Any]:
        """Evaluate a trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Train it first.")
            
            model = self.models[model_name]
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if problem_type == 'classification':
                metrics = self._calculate_classification_metrics(y_test, y_pred)
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(model, X_test.columns)
            
            return {
                "model_name": model_name,
                "problem_type": problem_type,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "predictions": y_pred.tolist(),
                "actual": y_test.tolist()
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                           problem_type: str, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on a model"""
        try:
            models = self.get_available_models(problem_type)
            if model_name not in models:
                raise ValueError(f"Model {model_name} not available for {problem_type}")
            
            model = models[model_name]
            
            # Perform cross-validation
            if problem_type == 'classification':
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            else:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            
            return {
                "model_name": model_name,
                "cv_scores": scores.tolist(),
                "mean_score": scores.mean(),
                "std_score": scores.std(),
                "cv_folds": cv_folds
            }
            
        except Exception as e:
            return {"error": f"Cross-validation failed: {str(e)}"}
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """Save a trained model to disk"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Save model
            joblib.dump(self.models[model_name], f"{filepath}_model.pkl")
            
            # Save scaler if exists
            if model_name in self.scalers:
                joblib.dump(self.scalers[model_name], f"{filepath}_scaler.pkl")
            
            # Save encoders if exist
            if model_name in self.encoders:
                joblib.dump(self.encoders[model_name], f"{filepath}_encoders.pkl")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_name: str, filepath: str) -> bool:
        """Load a trained model from disk"""
        try:
            # Load model
            self.models[model_name] = joblib.load(f"{filepath}_model.pkl")
            
            # Load scaler if exists
            try:
                self.scalers[model_name] = joblib.load(f"{filepath}_scaler.pkl")
            except:
                pass
            
            # Load encoders if exist
            try:
                self.encoders[model_name] = joblib.load(f"{filepath}_encoders.pkl")
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train or load it first.")
        
        # Preprocess data
        X = self._handle_missing_values(X)
        X = self._encode_categorical_variables(X)
        
        # Scale if scaler exists
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        # Make prediction
        return self.models[model_name].predict(X)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns, use median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
        
        # For categorical columns, use most frequent
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
        
        return data
    
    def _encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col])
            else:
                # Handle new categories
                unique_values = data[col].unique()
                known_values = self.encoders[col].classes_
                new_values = set(unique_values) - set(known_values)
                
                if len(new_values) > 0:
                    # Add new categories
                    all_values = list(known_values) + list(new_values)
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(all_values)
                
                data[col] = self.encoders[col].transform(data[col])
        
        return data
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features"""
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled = scaler.transform(X_test[numerical_cols])
            
            # Update dataframes
            X_train[numerical_cols] = X_train_scaled
            X_test[numerical_cols] = X_test_scaled
            
            # Store scaler
            self.scalers['current'] = scaler
        
        return X_train, X_test
    
    def _tune_hyperparameters(self, model, model_name: str, problem_type: str, 
                            X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform hyperparameter tuning"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=3, scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            # No hyperparameter tuning available, return original model
            model.fit(X_train, y_train)
            return model
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model, feature_names: pd.Index) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return dict(zip(feature_names, importance))
            elif hasattr(model, 'coef_'):
                # For linear models
                importance = np.abs(model.coef_)
                return dict(zip(feature_names, importance))
            else:
                return {}
        except:
            return {}
