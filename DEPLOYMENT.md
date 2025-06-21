# ONI Deployment Guide

## Quick Start

### 1. System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 8GB RAM
- 20GB storage
- CPU with AVX support

**Recommended Requirements:**
- Python 3.10+
- CUDA-compatible GPU (RTX 3080 or better)
- 32GB+ RAM
- 100GB+ SSD storage
- High-speed internet connection

### 2. Installation

```bash
# Clone repository
git clone https://github.com/ahricat/oni.git
cd oni

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### 3. Configuration

```bash
# Copy example config
cp config/settings.py.example config/settings.py

# Edit configuration
nano config/settings.py
```

**Key Configuration Options:**
```python
# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configurations
DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 896,
    "num_heads": 8,
    "num_layers": 6,
    "vocab_size": 300000,
    "max_length": 4096
}

# Memory configuration
MEMORY_CONFIG = {
    "working_memory_capacity": 5,
    "ltm_capacity": 10_000_000_000_000,
    "episodic_memory_path": DATA_DIR / "episodes",
    "semantic_memory_path": DATA_DIR / "semantic_memory.json",
    "ltm_summary_path": DATA_DIR / "ltm_data.json"
}
```

### 4. Environment Variables

Create a `.env` file in the project root:
```bash
# API Keys
ELEVENLABS_API_KEY=your_elevenlabs_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional: Custom paths
ONI_DATA_DIR=/path/to/data
ONI_MODELS_DIR=/path/to/models
ONI_CACHE_DIR=/path/to/cache

# Optional: Performance settings
ONI_MAX_WORKERS=4
ONI_BATCH_SIZE=32
ONI_MEMORY_LIMIT=16GB
```

### 5. Verify Installation

```bash
# Run tests
./scripts/run_tests.sh

# Check system status
python -c "from oni_core import create_oni; oni = create_oni(); print(oni.get_system_status())"
```

### 6. Start ONI

```bash
# Activate environment
source oni_env/bin/activate

# Start ONI core system
python oni_core.py

# Or start with specific configuration
python oni_core.py --config custom_config.py
```

---

## Production Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install ONI in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p data models logs cache knowledge_base

# Set environment variables
ENV PYTHONPATH=/app
ENV ONI_DATA_DIR=/app/data
ENV ONI_MODELS_DIR=/app/models

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "from oni_core import create_oni; oni = create_oni(); print('healthy')" || exit 1

# Start command
CMD ["python3", "oni_core.py", "--production"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  oni-core:
    build: .
    container_name: oni-core
    restart: unless-stopped
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - ONI_ENV=production
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./cache:/app/cache
    ports:
      - "8000:8000"
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  oni-memory:
    image: redis:7-alpine
    container_name: oni-memory-cache
    restart: unless-stopped
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  oni-db:
    image: postgres:15-alpine
    container_name: oni-database
    restart: unless-stopped
    environment:
      POSTGRES_DB: oni
      POSTGRES_USER: oni_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

**Deployment YAML:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oni-deployment
  namespace: oni-system
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
        - containerPort: 8001
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: ONI_ENV
          value: "production"
        - name: ELEVENLABS_API_KEY
          valueFrom:
            secretKeyRef:
              name: oni-secrets
              key: elevenlabs-api-key
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: oni-data
          mountPath: /app/data
        - name: oni-models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: oni-data
        persistentVolumeClaim:
          claimName: oni-data-pvc
      - name: oni-models
        persistentVolumeClaim:
          claimName: oni-models-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
---
apiVersion: v1
kind: Service
metadata:
  name: oni-service
  namespace: oni-system
spec:
  selector:
    app: oni
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: websocket
    port: 8001
    targetPort: 8001
  type: LoadBalancer
```

---

## Performance Optimization

### GPU Setup

```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Memory Optimization

**For Large Models:**
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Implement model sharding for multi-GPU
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

**Memory Management:**
```python
# Configure memory settings in config/settings.py
MEMORY_CONFIG = {
    "max_memory_usage": "80%",  # Use 80% of available GPU memory
    "enable_memory_mapping": True,
    "cache_size_limit": "8GB",
    "enable_gradient_checkpointing": True,
    "mixed_precision": True
}
```

### Multi-GPU Configuration

```python
# Distributed training setup
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
```

---

## Monitoring and Logging

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'oni'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 5s
```

### Grafana Dashboard

Key metrics to monitor:
- GPU utilization and memory usage
- Model inference latency
- Memory system performance
- Emotional state metrics
- Compassion framework decisions
- Error rates and system health

### Logging Configuration

```python
# Enhanced logging in config/settings.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": LOGS_DIR / "oni.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": LOGS_DIR / "oni_errors.log",
            "maxBytes": 10485760,
            "backupCount": 3
        }
    },
    "loggers": {
        "oni": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
```

---

## Security and Safety

### Security Best Practices

1. **API Key Management:**
   ```bash
   # Use environment variables or secret management
   export ELEVENLABS_API_KEY=$(cat /run/secrets/elevenlabs_key)
   export OPENAI_API_KEY=$(cat /run/secrets/openai_key)
   ```

2. **Network Security:**
   ```yaml
   # Firewall rules
   - Allow incoming on port 8000 from trusted IPs only
   - Block direct access to internal ports (8001, 8002)
   - Use TLS/SSL for all external communications
   ```

3. **Data Protection:**
   ```python
   # Encrypt sensitive data at rest
   ENCRYPTION_CONFIG = {
       "enable_encryption": True,
       "key_rotation_days": 30,
       "encrypt_memory_dumps": True,
       "encrypt_model_weights": True
   }
   ```

### Safety Mechanisms

1. **Compassion Framework Monitoring:**
   ```python
   # Monitor ethical decision-making
   SAFETY_CONFIG = {
       "enable_compassion_monitoring": True,
       "ethical_threshold": 0.7,
       "auto_shutdown_on_violation": True,
       "human_oversight_required": True
   }
   ```

2. **Resource Limits:**
   ```python
   # Prevent resource exhaustion
   RESOURCE_LIMITS = {
       "max_memory_per_request": "4GB",
       "max_processing_time": 300,  # seconds
       "max_concurrent_requests": 10,
       "rate_limit_per_user": "100/hour"
   }
   ```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   ```bash
   # Reduce batch size or enable gradient checkpointing
   export ONI_BATCH_SIZE=16
   export ONI_ENABLE_CHECKPOINTING=true
   ```

2. **Model Loading Errors:**
   ```bash
   # Clear cache and re-download models
   rm -rf cache/models/*
   python -c "from oni_core import download_models; download_models()"
   ```

3. **Memory System Issues:**
   ```bash
   # Reset memory systems
   python -c "from modules.oni_memory import Memory; Memory.reset_all()"
   ```

### Debug Mode

```bash
# Start ONI in debug mode
python oni_core.py --debug --log-level DEBUG

# Enable memory profiling
python -m memory_profiler oni_core.py

# Profile performance
python -m cProfile -o oni_profile.prof oni_core.py
```

### Health Checks

```python
# System health check script
def health_check():
    checks = {
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": get_memory_usage(),
        "model_loaded": check_model_status(),
        "compassion_system": check_compassion_system(),
        "emotional_system": check_emotional_system()
    }
    return all(checks.values()), checks

# Run health check
healthy, status = health_check()
print(f"System healthy: {healthy}")
print(f"Status: {status}")
```

---

## Scaling and Load Balancing

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: oni-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: oni-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing

```nginx
# Nginx configuration
upstream oni_backend {
    least_conn;
    server oni-1:8000 max_fails=3 fail_timeout=30s;
    server oni-2:8000 max_fails=3 fail_timeout=30s;
    server oni-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name oni.example.com;

    location / {
        proxy_pass http://oni_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /ws {
        proxy_pass http://oni_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

This deployment guide provides comprehensive instructions for deploying ONI in various environments, from development to production-scale deployments. The configuration options and monitoring setup ensure reliable, secure, and performant operation of the ONI system.