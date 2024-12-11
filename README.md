# Production-Grade Machine Learning System

A sophisticated machine learning system designed for production environments, incorporating MLOps best practices, monitoring, and scalability.

## Features

- Production-ready deep learning model training pipeline
- Advanced data preprocessing and feature engineering
- Comprehensive experiment tracking with MLflow and Weights & Biases
- Real-time model serving with FastAPI
- Prometheus metrics and monitoring
- Kubernetes deployment configuration
- Robust logging and error handling
- Security features including API key authentication and rate limiting
- Comprehensive testing suite

## Project Structure

```
ml_production_system/
├── src/
│   ├── api/            # FastAPI service
│   ├── data/           # Data processing
│   ├── models/         # ML models
│   └── utils/          # Utilities
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── config/             # Configuration files
├── models/             # Saved models
├── data/               # Data files
├── notebooks/          # Jupyter notebooks
├── deployment/         # K8s configs
└── docs/              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kakachia777/ml_production_system.git
cd ml_production_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example configuration:
```bash
cp config/config.example.yaml config/config.yaml
```

2. Update the configuration with your settings:
- Set your MLflow tracking URI
- Configure your Weights & Biases credentials
- Adjust model hyperparameters
- Set up monitoring endpoints

## Usage

### Training

Train the model with:
```bash
python src/train.py
```

### Serving

Start the API server:
```bash
uvicorn src.api.service:create_app --host 0.0.0.0 --port 8000 --workers 4
```

### Monitoring

Access metrics at:
- API documentation: http://localhost:8000/docs
- Metrics endpoint: http://localhost:8000/metrics
- Health check: http://localhost:8000/health

## Development

### Testing

Run tests:
```bash
pytest tests/
```

### Code Quality

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
flake8 src/ tests/
```

## Deployment

### Local Deployment

1. Start MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Start the API server:
```bash
python src/api/service.py
```

### Kubernetes Deployment

1. Build the Docker image:
```bash
docker build -t ml-production-system:latest .
```

2. Apply Kubernetes configurations:
```bash
kubectl apply -f deployment/k8s/
```

## Monitoring

### Prometheus Metrics

The system exposes the following metrics:
- Prediction request count
- Prediction latency
- Model performance metrics
- System resource usage

### Logging

Logs are written to:
- Console (colored output)
- File (JSON format)
- MLflow (training metrics)
- Weights & Biases (training visualization)

## Security

- API key authentication required for predictions
- Rate limiting implemented
- CORS configuration
- Input validation
- Secure model serving

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact:
- Email: Beka.kakachia777@gmail.com
