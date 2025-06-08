# Oni Deployment Guide

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/ahricat/oni.git
cd oni

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### 2. Configuration
```bash
# Copy example config
cp config/settings.py.example config/settings.py

# Edit configuration
nano config/settings.py
```

### 3. Environment Variables
```bash
# Create .env file
echo "ELEVENLABS_API_KEY=your_key_here" > .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### 4. Run Tests
```bash
./scripts/run_tests.sh
```

### 5. Start Oni
```bash
python oni_core.py
```

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "oni_core.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oni-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oni
  template:
    metadata:
      labels:
        app: oni
    spec:
      containers:
      - name: oni
        image: oni:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEVICE
          value: "cuda"
```

## Performance Optimization

### GPU Setup
```bash
# Install CUDA support
pip install torch[cuda]>=2.0.0

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Optimization
- Use gradient checkpointing for large models
- Implement model sharding for multi-GPU setups
- Configure appropriate batch sizes

### Monitoring
- Use Prometheus for metrics collection
- Implement health checks
- Set up alerting for critical failures