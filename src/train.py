import os
import logging
from typing import Dict, Any

import hydra
from omegaconf import DictConfig
import mlflow
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.neural_net import DeepNeuralNetwork
from data.data_processor import DataProcessor
from utils.metrics import calculate_metrics
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class MLTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_tracking()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_tracking(self):
        """Initialize MLflow and Weights & Biases tracking."""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        wandb.init(
            project=self.config.wandb.project_name,
            config=self.config,
            sync_tensorboard=True,
        )
        
    def prepare_data(self):
        """Prepare and split data with proper validation."""
        processor = DataProcessor(self.config.data)
        X, y = processor.load_and_preprocess()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.data.test_size, random_state=42
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        return (
            DataLoader(TensorDataset(X_train, y_train), batch_size=self.config.training.batch_size),
            DataLoader(TensorDataset(X_test, y_test), batch_size=self.config.training.batch_size),
        )
    
    def train_model(self):
        """Train the model with proper logging and validation."""
        train_loader, test_loader = self.prepare_data()
        
        model = DeepNeuralNetwork(
            input_size=self.config.model.input_size,
            hidden_sizes=self.config.model.hidden_sizes,
            output_size=self.config.model.output_size,
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.training.learning_rate)
        
        with mlflow.start_run():
            mlflow.log_params(self.config.model)
            
            for epoch in range(self.config.training.epochs):
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_metrics = self.validate_model(model, test_loader)
                
                # Log metrics
                metrics = {
                    "train_loss": train_loss / len(train_loader),
                    **val_metrics
                }
                
                mlflow.log_metrics(metrics, step=epoch)
                wandb.log(metrics)
                
                logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} - {metrics}")
        
        return model
    
    def validate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Validate the model and return metrics."""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return calculate_metrics(np.array(all_labels), np.array(all_preds))

@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    setup_logging()
    logger.info("Starting training pipeline")
    
    trainer = MLTrainer(config)
    model = trainer.train_model()
    
    # Save model artifacts
    torch.save(model.state_dict(), os.path.join(config.model.save_path, "model.pth"))
    mlflow.pytorch.log_model(model, "model")
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main() 