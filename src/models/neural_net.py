import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # Input validation
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        
        # Create layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights using He initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        return self.output_layer(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate feature importance using gradient-based approach."""
        x.requires_grad = True
        self.zero_grad()
        
        output = self.forward(x)
        output.backward(torch.ones_like(output))
        
        importance = x.grad.abs().mean(dim=0)
        return importance / importance.sum() 