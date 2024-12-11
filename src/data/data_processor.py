import logging
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import great_expectations as ge

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load, validate, and preprocess the data."""
        data = self._load_data()
        data = self._validate_data(data)
        return self._preprocess_data(data)
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from specified source."""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        logger.info(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data using Great Expectations."""
        ge_data = ge.from_pandas(data)
        
        # Define and validate expectations
        suite = ge_data.expect_column_values_to_not_be_null(self.config.target_column)
        for feature in self.config.feature_columns:
            suite = suite.expect_column_values_to_be_between(
                feature,
                min_value=self.config.validation.min_value,
                max_value=self.config.validation.max_value
            )
        
        validation_result = suite.validate()
        if not validation_result.success:
            logger.warning("Data validation failed. Check validation results for details.")
            self._handle_validation_failures(validation_result)
        
        return data
    
    def _handle_validation_failures(self, validation_result) -> None:
        """Handle data validation failures."""
        for result in validation_result.results:
            if not result.success:
                logger.warning(f"Validation failed: {result.expectation_config.kwargs}")
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data with advanced techniques."""
        X = data[self.config.feature_columns]
        y = data[self.config.target_column]
        
        # Handle missing values
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Feature engineering
        X = self._engineer_features(X)
        
        # Normalization
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X.values, y.values
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering techniques."""
        if self.config.get('feature_engineering'):
            for feature in self.config.feature_engineering:
                if feature.type == 'polynomial':
                    X = self._add_polynomial_features(X, feature.columns, feature.degree)
                elif feature.type == 'interaction':
                    X = self._add_interaction_features(X, feature.columns)
        
        return X
    
    def _add_polynomial_features(self, X: pd.DataFrame, columns: list, degree: int) -> pd.DataFrame:
        """Add polynomial features for specified columns."""
        for col in columns:
            for d in range(2, degree + 1):
                X[f"{col}_pow_{d}"] = X[col] ** d
        return X
    
    def _add_interaction_features(self, X: pd.DataFrame, column_pairs: list) -> pd.DataFrame:
        """Add interaction features for specified column pairs."""
        for col1, col2 in column_pairs:
            X[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
        return X
    
    def save_preprocessor(self, path: str) -> None:
        """Save preprocessor objects for inference."""
        import joblib
        
        preprocessor_path = Path(path)
        preprocessor_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, preprocessor_path / "scaler.joblib")
        joblib.dump(self.imputer, preprocessor_path / "imputer.joblib")
        
        logger.info(f"Saved preprocessor objects to {preprocessor_path}") 