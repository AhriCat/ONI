# ONI RLHF Training System

A comprehensive TypeScript-based Reinforcement Learning with Human Feedback (RLHF) system for training ONI models with blockchain integration and proof-of-contribute mechanisms.

## 🚀 Features

### Core RLHF Capabilities
- **Multi-Model Support**: Train any ONI model type (NLP, Vision, Audio, Multimodal, Full-ONI)
- **Advanced PPO Implementation**: Proximal Policy Optimization with clipping and entropy regularization
- **Reward Model Training**: Hybrid AI/Human reward models with continuous learning
- **Compassion Framework Integration**: Ethical AI training with A, C, S metrics
- **Real-time Feedback**: WebSocket-based live feedback collection and processing

### Blockchain Integration
- **Proof-of-Contribute**: Verifiable contribution tracking on blockchain
- **ONI Token Rewards**: Automatic token distribution for contributions
- **Decentralized Training**: Global participation in model improvement
- **Contribution Verification**: Cryptographic proof of training contributions
- **Smart Contract Integration**: Automated reward distribution and verification

### Model Weight Management
- **HuggingFace Compatibility**: Export models in HuggingFace format
- **Multiple Formats**: Support for PyTorch, TensorFlow, ONNX, and HuggingFace
- **Weight Compression**: Gzip, Brotli compression options
- **Quantization**: FP16, INT8, INT4 precision options
- **Automatic Conversion**: Seamless format conversion with metadata preservation

### Advanced Features
- **Emotional Intelligence**: Integrated emotional processing in training
- **Meta-Cognition**: Self-awareness and confidence estimation
- **Memory Integration**: Episodic and semantic memory during training
- **Multi-Modal Training**: Simultaneous text, vision, and audio training
- **Distributed Training**: Multi-GPU and multi-node support

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ahricat/oni.git
cd oni/rlhf

# Install dependencies
npm install

# Build the project
npm run build

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
PORT=3001
NODE_ENV=production

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/oni_rlhf
REDIS_URL=redis://localhost:6379

# Blockchain Configuration
BLOCKCHAIN_NETWORK_ID=oni-mainnet
BLOCKCHAIN_RPC_URL=https://rpc.oni-network.com
CONTRACT_ADDRESS=0x1234567890123456789012345678901234567890
BLOCKCHAIN_PRIVATE_KEY=your_private_key_here

# API Keys
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key

# Storage Configuration
MODELS_PATH=./models
DOWNLOADS_PATH=./downloads
CACHE_PATH=./cache

# Security
JWT_SECRET=your_jwt_secret_here
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.com
```

### Model Configuration

```typescript
const modelConfig: ModelConfig = {
  id: "oni-nlp-v1",
  name: "ONI NLP Model v1",
  type: "nlp",
  architecture: "transformer",
  parameters: 896000000,
  inputShape: [4096],
  outputShape: [300000],
  hyperparameters: {
    vocabSize: 300000,
    hiddenDim: 896,
    numHeads: 8,
    numLayers: 6,
    maxLength: 4096
  }
};
```

## 🚀 Usage

### Starting the System

```bash
# Development mode
npm run dev

# Production mode
npm start
```

### API Endpoints

#### Training Management
```bash
# Start training session
POST /api/training/start
{
  "modelId": "oni-nlp-v1",
  "configuration": {
    "batchSize": 32,
    "learningRate": 0.0001,
    "epochs": 10,
    "rewardModel": {
      "type": "hybrid",
      "humanFeedbackWeight": 0.7,
      "compassionWeight": 0.3
    }
  }
}

# Get training status
GET /api/training/{sessionId}

# Pause training
POST /api/training/{sessionId}/pause

# Resume training
POST /api/training/{sessionId}/resume
```

#### Human Feedback
```bash
# Submit feedback
POST /api/feedback
{
  "sessionId": "session-123",
  "inputText": "Hello, how are you?",
  "modelResponse": "I'm doing well, thank you!",
  "humanRating": 8,
  "compassionRating": 0.9,
  "categories": [
    {"name": "helpfulness", "score": 8, "weight": 1.0},
    {"name": "safety", "score": 9, "weight": 1.2}
  ]
}
```

#### Model Weights
```bash
# Download model weights
POST /api/weights/models/{modelId}/download
{
  "format": "huggingface",
  "compression": "gzip",
  "precision": "fp16",
  "includeOptimizer": false
}

# Check download status
GET /api/weights/downloads/{downloadId}/status
```

#### Blockchain Integration
```bash
# Get user contributions
GET /api/blockchain/contributions/{userId}

# Get user rewards
GET /api/blockchain/rewards/{userId}

# Verify contribution
POST /api/blockchain/verify-contribution
{
  "proof": {
    "contributorId": "user-123",
    "contributionType": "training",
    "computeHours": 5.2,
    "qualityScore": 0.85
  }
}
```

### WebSocket Events

```typescript
// Connect to training session
socket.emit('join-session', sessionId);

// Listen for training progress
socket.on('training-progress', (progress) => {
  console.log('Training progress:', progress);
});

// Submit real-time feedback
socket.emit('submit-feedback', feedbackData);

// Listen for feedback processing
socket.on('feedback-processed', (feedback) => {
  console.log('Feedback processed:', feedback);
});
```

## 🔗 Blockchain Integration

### Smart Contract Functions

The system integrates with ONI blockchain smart contracts:

```solidity
// Record training contribution
function recordContribution(
    string memory sessionId,
    string memory modelId,
    string memory contributorId,
    uint8 contributionType,
    uint256 computeHours,
    uint256 qualityScore
) external;

// Record human feedback
function recordFeedback(
    string memory sessionId,
    string memory feedbackId,
    string memory userId,
    uint256 rating,
    uint256 compassionScore
) external;

// Get contributor rewards
function getContributorRewards(
    string memory contributorId
) external view returns (uint256);
```

### Proof-of-Contribute

The system automatically generates and verifies proof-of-contribute for:

- **Training Contributions**: Compute hours, model improvements, quality scores
- **Feedback Contributions**: Human feedback quality and quantity
- **Validation Contributions**: Model testing and validation work
- **Data Contributions**: Training data provision and curation

## 📊 Monitoring and Metrics

### System Metrics
- Active training sessions
- Queue status and processing times
- GPU utilization and memory usage
- Blockchain synchronization status
- Database performance metrics

### Training Metrics
- Loss curves and convergence rates
- Reward scores and compassion metrics
- Human feedback ratings and trends
- Model performance benchmarks
- Ethical compliance scores

### Blockchain Metrics
- Contribution verification rates
- Token distribution statistics
- Network participation metrics
- Smart contract gas usage

## 🛡️ Security Features

### Authentication & Authorization
- JWT-based authentication
- Role-based access control
- API rate limiting
- Request validation and sanitization

### Data Protection
- Encrypted data at rest
- Secure API communications
- Privacy-preserving feedback collection
- GDPR compliance features

### Blockchain Security
- Cryptographic proof verification
- Smart contract audit trails
- Secure key management
- Multi-signature support

## 🔧 Development

### Project Structure
```
rlhf/
├── src/
│   ├── core/           # Core RLHF components
│   ├── api/            # REST API endpoints
│   ├── blockchain/     # Blockchain integration
│   ├── database/       # Database management
│   ├── queue/          # Job queue management
│   ├── monitoring/     # System monitoring
│   ├── middleware/     # Express middleware
│   ├── utils/          # Utility functions
│   └── types/          # TypeScript type definitions
├── tests/              # Test suites
├── docs/               # Documentation
└── scripts/            # Build and deployment scripts
```

### Testing

```bash
# Run all tests
npm test

# Run specific test suite
npm test -- --grep "RLHFTrainer"

# Run with coverage
npm run test:coverage
```

### Building

```bash
# Build for production
npm run build

# Build with watch mode
npm run build:watch

# Type checking
npm run type-check
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY dist/ ./dist/
COPY public/ ./public/

EXPOSE 3001
CMD ["node", "dist/index.js"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oni-rlhf-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oni-rlhf
  template:
    metadata:
      labels:
        app: oni-rlhf
    spec:
      containers:
      - name: oni-rlhf
        image: oni/rlhf-system:latest
        ports:
        - containerPort: 3001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: oni-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 📈 Performance Optimization

### Training Optimization
- Gradient accumulation for large batch sizes
- Mixed precision training (FP16)
- Model parallelism for large models
- Efficient data loading and preprocessing

### System Optimization
- Redis caching for frequent queries
- Database connection pooling
- Async processing with job queues
- Load balancing across multiple instances

### Blockchain Optimization
- Batch transaction processing
- Gas optimization strategies
- Layer 2 scaling solutions
- Efficient proof generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript best practices
- Write comprehensive tests
- Document all public APIs
- Ensure blockchain integration works
- Test HuggingFace compatibility

## 📄 License

This project is licensed under the Pantheum License - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full API Documentation](./docs/api.md)
- **Issues**: [GitHub Issues](https://github.com/ahricat/oni/issues)
- **Discord**: [ONI Community](https://discord.gg/oni-agi)
- **Email**: support@oni-agi.com

## 🙏 Acknowledgments

- HuggingFace team for transformer implementations
- OpenAI for RLHF research and methodologies
- Ethereum community for blockchain infrastructure
- TensorFlow.js team for in-browser ML capabilities

---

**Built with ❤️ for the ONI AGI ecosystem**