import logging
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from prometheus_client import Counter, Histogram, start_http_server

from models.neural_net import DeepNeuralNetwork
from data.data_processor import DataProcessor
from utils.logging_utils import setup_logging, get_logger

# Initialize logging
logger = get_logger(__name__)

# Initialize metrics
PREDICTION_REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests'
)

class PredictionRequest(BaseModel):
    """Prediction request model."""
    features: List[float] = Field(..., description="List of input features")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: float
    confidence: float
    request_id: Optional[str]
    processing_time: float
    timestamp: datetime

class MLService:
    def __init__(self, config: Dict):
        """Initialize the ML service."""
        self.config = config
        self.model = self._load_model()
        self.processor = DataProcessor(config['data'])
        self.api_keys = set(config['security']['api_keys'])
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="ML Production System API",
            description="Production-ready ML API with monitoring and security",
            version="1.0.0"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Start metrics server
        start_http_server(config['monitoring']['prometheus']['port'])
    
    def _load_model(self) -> DeepNeuralNetwork:
        """Load the trained model."""
        try:
            model = DeepNeuralNetwork(
                input_size=self.config['model']['input_size'],
                hidden_sizes=self.config['model']['hidden_sizes'],
                output_size=self.config['model']['output_size']
            )
            model.load_state_dict(torch.load(self.config['model']['save_path']))
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError("Model loading failed")
    
    def _setup_middleware(self):
        """Setup API middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['security']['cors']['allowed_origins'],
            allow_methods=self.config['security']['cors']['allowed_methods'],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        api_key_header = APIKeyHeader(name="X-API-Key")
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow()}
        
        @self.app.post("/api/v1/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            api_key: str = Security(api_key_header)
        ) -> PredictionResponse:
            # Validate API key
            if api_key not in self.api_keys:
                raise HTTPException(status_code=403, detail="Invalid API key")
            
            # Record request
            PREDICTION_REQUEST_COUNT.inc()
            start_time = datetime.utcnow()
            
            try:
                with PREDICTION_LATENCY.time():
                    # Preprocess input
                    features = torch.FloatTensor(request.features).unsqueeze(0)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = self.model(features)
                        probabilities = torch.softmax(output, dim=1)
                        prediction = float(torch.argmax(output, dim=1))
                        confidence = float(torch.max(probabilities))
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    return PredictionResponse(
                        prediction=prediction,
                        confidence=confidence,
                        request_id=request.request_id,
                        processing_time=processing_time,
                        timestamp=datetime.utcnow()
                    )
            
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail="Prediction failed")
        
        @self.app.get("/metrics")
        async def metrics():
            """Endpoint for Prometheus metrics."""
            return {
                "prediction_requests": PREDICTION_REQUEST_COUNT._value.get(),
                "average_latency": PREDICTION_LATENCY.observe()
            }

def create_app(config: Dict) -> FastAPI:
    """Create and configure the FastAPI application."""
    service = MLService(config)
    return service.app 