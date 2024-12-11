from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive set of classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing various classification metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Handling binary and multiclass cases
    if len(np.unique(y_true)) == 2:
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            metrics['average_precision'] = average_precision_score(y_true, y_pred)
        except ValueError:
            # Handle cases where predictions are not probabilistic
            pass
    else:
        # Multiclass metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix statistics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Additional metrics
    metrics['balanced_accuracy'] = np.mean([
        recall_score(y_true, y_pred, average='macro'),
        precision_score(y_true, y_pred, average='macro')
    ])
    
    return metrics

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive set of regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various regression metrics
    """
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    
    # R-squared score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot)
    
    # Additional metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['explained_variance'] = 1 - (np.var(y_true - y_pred) / np.var(y_true))
    
    return metrics 