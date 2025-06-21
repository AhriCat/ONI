import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import { RLHFTrainer } from './core/RLHFTrainer';
import { ModelWeightsAPI } from './api/ModelWeightsAPI';
import { BlockchainIntegration } from './blockchain/BlockchainIntegration';
import { TrainingSessionManager } from './core/TrainingSessionManager';
import { FeedbackCollector } from './core/FeedbackCollector';
import { Logger } from './utils/Logger';
import { DatabaseManager } from './database/DatabaseManager';
import { QueueManager } from './queue/QueueManager';
import { MetricsCollector } from './core/MetricsCollector';
import { SystemMonitor } from './monitoring/SystemMonitor';
import { AuthMiddleware } from './middleware/AuthMiddleware';
import { RateLimitMiddleware } from './middleware/RateLimitMiddleware';
import { TrainingSession, ModelConfig, User, SystemStatus } from './types';

class ONIRLHFSystem {
  private app: express.Application;
  private server: any;
  private io: SocketIOServer;
  private logger: Logger;
  private blockchain: BlockchainIntegration;
  private sessionManager: TrainingSessionManager;
  private feedbackCollector: FeedbackCollector;
  private weightsAPI: ModelWeightsAPI;
  private database: DatabaseManager;
  private queueManager: QueueManager;
  private metricsCollector: MetricsCollector;
  private systemMonitor: SystemMonitor;
  private activeSessions: Map<string, RLHFTrainer> = new Map();

  constructor() {
    this.app = express();
    this.server = createServer(this.app);
    this.io = new SocketIOServer(this.server, {
      cors: {
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
        methods: ['GET', 'POST']
      }
    });
    
    this.logger = new Logger('ONIRLHFSystem');
    this.initializeSystem();
  }

  private async initializeSystem(): Promise<void> {
    try {
      this.logger.info('Initializing ONI RLHF System...');

      // Initialize core components
      this.blockchain = new BlockchainIntegration();
      this.database = new DatabaseManager();
      this.queueManager = new QueueManager();
      this.metricsCollector = new MetricsCollector();
      this.systemMonitor = new SystemMonitor();
      
      // Initialize managers
      this.sessionManager = new TrainingSessionManager(this.database, this.blockchain);
      this.feedbackCollector = new FeedbackCollector(this.database, this.blockchain);
      this.weightsAPI = new ModelWeightsAPI();

      // Setup middleware
      this.setupMiddleware();
      
      // Setup routes
      this.setupRoutes();
      
      // Setup WebSocket handlers
      this.setupWebSocketHandlers();
      
      // Start system monitoring
      this.systemMonitor.start();
      
      this.logger.info('ONI RLHF System initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize ONI RLHF System:', error);
      throw error;
    }
  }

  private setupMiddleware(): void {
    this.app.use(helmet());
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
      credentials: true
    }));
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));
    
    // Authentication middleware
    this.app.use('/api', AuthMiddleware);
    
    // Rate limiting
    this.app.use('/api', RateLimitMiddleware);
    
    // Request logging
    this.app.use((req, res, next) => {
      this.logger.info(`${req.method} ${req.path}`, {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        userId: (req as any).user?.id
      });
      next();
    });
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/health', this.healthCheck.bind(this));
    
    // System status
    this.app.get('/api/status', this.getSystemStatus.bind(this));
    
    // Training session management
    this.app.post('/api/training/start', this.startTrainingSession.bind(this));
    this.app.post('/api/training/:sessionId/pause', this.pauseTrainingSession.bind(this));
    this.app.post('/api/training/:sessionId/resume', this.resumeTrainingSession.bind(this));
    this.app.post('/api/training/:sessionId/stop', this.stopTrainingSession.bind(this));
    this.app.get('/api/training/:sessionId', this.getTrainingSession.bind(this));
    this.app.get('/api/training', this.listTrainingSessions.bind(this));
    
    // Human feedback
    this.app.post('/api/feedback', this.submitFeedback.bind(this));
    this.app.get('/api/feedback/:sessionId', this.getFeedback.bind(this));
    
    // Model weights API
    this.app.use('/api/weights', this.weightsAPI.getApp());
    
    // Blockchain integration
    this.app.get('/api/blockchain/contributions/:userId', this.getUserContributions.bind(this));
    this.app.get('/api/blockchain/rewards/:userId', this.getUserRewards.bind(this));
    this.app.post('/api/blockchain/verify-contribution', this.verifyContribution.bind(this));
    
    // Metrics and monitoring
    this.app.get('/api/metrics/:sessionId', this.getSessionMetrics.bind(this));
    this.app.get('/api/metrics', this.getSystemMetrics.bind(this));
    
    // Model management
    this.app.post('/api/models', this.registerModel.bind(this));
    this.app.get('/api/models', this.listModels.bind(this));
    this.app.get('/api/models/:modelId', this.getModel.bind(this));
    
    // Error handling
    this.app.use(this.errorHandler.bind(this));
  }

  private setupWebSocketHandlers(): void {
    this.io.on('connection', (socket) => {
      this.logger.info(`Client connected: ${socket.id}`);
      
      // Join training session room
      socket.on('join-session', (sessionId: string) => {
        socket.join(`session-${sessionId}`);
        this.logger.info(`Client ${socket.id} joined session ${sessionId}`);
      });
      
      // Leave training session room
      socket.on('leave-session', (sessionId: string) => {
        socket.leave(`session-${sessionId}`);
        this.logger.info(`Client ${socket.id} left session ${sessionId}`);
      });
      
      // Real-time feedback submission
      socket.on('submit-feedback', async (feedbackData) => {
        try {
          const feedback = await this.feedbackCollector.processFeedback(feedbackData);
          
          // Broadcast to session participants
          this.io.to(`session-${feedback.sessionId}`).emit('feedback-received', feedback);
          
          // Update training if session is active
          const trainer = this.activeSessions.get(feedback.sessionId);
          if (trainer) {
            await trainer.addHumanFeedback(feedback);
          }
        } catch (error) {
          socket.emit('feedback-error', { error: error.message });
        }
      });
      
      // Request real-time metrics
      socket.on('subscribe-metrics', (sessionId: string) => {
        socket.join(`metrics-${sessionId}`);
      });
      
      socket.on('unsubscribe-metrics', (sessionId: string) => {
        socket.leave(`metrics-${sessionId}`);
      });
      
      socket.on('disconnect', () => {
        this.logger.info(`Client disconnected: ${socket.id}`);
      });
    });
  }

  private async healthCheck(req: express.Request, res: express.Response): Promise<void> {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      version: process.env.npm_package_version || '1.0.0',
      components: {
        database: await this.database.isHealthy(),
        blockchain: await this.blockchain.isHealthy(),
        queue: await this.queueManager.isHealthy()
      }
    };
    
    const isHealthy = Object.values(health.components).every(status => status);
    res.status(isHealthy ? 200 : 503).json(health);
  }

  private async getSystemStatus(req: express.Request, res: express.Response): Promise<void> {
    try {
      const status: SystemStatus = {
        status: 'healthy',
        activeTrainingSessions: this.activeSessions.size,
        totalModels: await this.database.getModelCount(),
        queuedJobs: await this.queueManager.getQueueSize(),
        systemLoad: await this.systemMonitor.getSystemLoad(),
        memoryUsage: await this.systemMonitor.getMemoryUsage(),
        gpuUtilization: await this.systemMonitor.getGPUUtilization(),
        blockchainSync: await this.blockchain.isSynced(),
        lastUpdate: new Date()
      };
      
      res.json({
        success: true,
        data: status,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error getting system status:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get system status',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async startTrainingSession(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { modelId, configuration } = req.body;
      const user = (req as any).user as User;
      
      // Validate request
      if (!modelId || !configuration) {
        return res.status(400).json({
          success: false,
          error: 'Model ID and configuration are required',
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }
      
      // Get model configuration
      const modelConfig = await this.database.getModel(modelId);
      if (!modelConfig) {
        return res.status(404).json({
          success: false,
          error: 'Model not found',
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }
      
      // Create training session
      const session = await this.sessionManager.createSession({
        modelId,
        userId: user.id,
        configuration,
        status: 'pending'
      });
      
      // Initialize trainer
      const trainer = new RLHFTrainer(session, modelConfig, this.blockchain);
      this.activeSessions.set(session.id, trainer);
      
      // Setup event handlers
      this.setupTrainerEventHandlers(trainer, session.id);
      
      // Add to training queue
      await this.queueManager.addTrainingJob(session.id, async () => {
        await trainer.startTraining();
      });
      
      res.json({
        success: true,
        data: { sessionId: session.id, status: session.status },
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error starting training session:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to start training session',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private setupTrainerEventHandlers(trainer: RLHFTrainer, sessionId: string): void {
    trainer.on('trainingStarted', (session) => {
      this.io.to(`session-${sessionId}`).emit('training-started', session);
    });
    
    trainer.on('trainingProgress', (progress) => {
      this.io.to(`session-${sessionId}`).emit('training-progress', progress);
      this.io.to(`metrics-${sessionId}`).emit('metrics-update', progress.metrics);
    });
    
    trainer.on('trainingCompleted', (session) => {
      this.io.to(`session-${sessionId}`).emit('training-completed', session);
      this.activeSessions.delete(sessionId);
    });
    
    trainer.on('trainingFailed', (data) => {
      this.io.to(`session-${sessionId}`).emit('training-failed', data);
      this.activeSessions.delete(sessionId);
    });
    
    trainer.on('checkpointSaved', (checkpoint) => {
      this.io.to(`session-${sessionId}`).emit('checkpoint-saved', checkpoint);
    });
    
    trainer.on('feedbackAdded', (feedback) => {
      this.io.to(`session-${sessionId}`).emit('feedback-processed', feedback);
    });
  }

  private async pauseTrainingSession(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { sessionId } = req.params;
      const trainer = this.activeSessions.get(sessionId);
      
      if (!trainer) {
        return res.status(404).json({
          success: false,
          error: 'Training session not found or not active',
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }
      
      await trainer.pauseTraining();
      
      res.json({
        success: true,
        data: { sessionId, status: 'paused' },
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error pausing training session:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to pause training session',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async submitFeedback(req: express.Request, res: express.Response): Promise<void> {
    try {
      const feedbackData = req.body;
      const user = (req as any).user as User;
      
      feedbackData.userId = user.id;
      feedbackData.timestamp = new Date();
      
      const feedback = await this.feedbackCollector.processFeedback(feedbackData);
      
      // Add to active training session if exists
      const trainer = this.activeSessions.get(feedback.sessionId);
      if (trainer) {
        await trainer.addHumanFeedback(feedback);
      }
      
      res.json({
        success: true,
        data: feedback,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error submitting feedback:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to submit feedback',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async getUserContributions(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { userId } = req.params;
      const contributions = await this.blockchain.getSessionContributions(userId);
      
      res.json({
        success: true,
        data: contributions,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error getting user contributions:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get user contributions',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async getUserRewards(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { userId } = req.params;
      const rewards = await this.blockchain.getContributorRewards(userId);
      
      res.json({
        success: true,
        data: { userId, totalRewards: rewards },
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error getting user rewards:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get user rewards',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async verifyContribution(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { proof } = req.body;
      const isValid = await this.blockchain.verifyProofOfContribute(proof);
      
      res.json({
        success: true,
        data: { valid: isValid },
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error verifying contribution:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to verify contribution',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async getSessionMetrics(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { sessionId } = req.params;
      const trainer = this.activeSessions.get(sessionId);
      
      if (!trainer) {
        return res.status(404).json({
          success: false,
          error: 'Training session not found or not active',
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }
      
      const metrics = trainer.getMetrics();
      
      res.json({
        success: true,
        data: metrics,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error getting session metrics:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get session metrics',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async registerModel(req: express.Request, res: express.Response): Promise<void> {
    try {
      const modelConfig: ModelConfig = req.body;
      const user = (req as any).user as User;
      
      // Validate model configuration
      if (!modelConfig.name || !modelConfig.type || !modelConfig.architecture) {
        return res.status(400).json({
          success: false,
          error: 'Model name, type, and architecture are required',
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }
      
      const model = await this.database.createModel(modelConfig, user.id);
      
      res.json({
        success: true,
        data: model,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error registering model:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to register model',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async listModels(req: express.Request, res: express.Response): Promise<void> {
    try {
      const models = await this.database.listModels();
      
      res.json({
        success: true,
        data: models,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error listing models:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to list models',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private errorHandler(error: any, req: express.Request, res: express.Response, next: express.NextFunction): void {
    this.logger.error('Unhandled error:', error);
    
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      timestamp: new Date(),
      requestId: req.headers['x-request-id']
    });
  }

  public async start(port: number = 3001): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(port, () => {
        this.logger.info(`ONI RLHF System started on port ${port}`);
        resolve();
      });
    });
  }

  public async stop(): Promise<void> {
    this.logger.info('Shutting down ONI RLHF System...');
    
    // Stop all active training sessions
    for (const [sessionId, trainer] of this.activeSessions) {
      await trainer.pauseTraining();
      await trainer.dispose();
    }
    this.activeSessions.clear();
    
    // Stop system components
    await this.systemMonitor.stop();
    await this.queueManager.stop();
    await this.database.disconnect();
    this.blockchain.dispose();
    
    // Close server
    this.server.close();
    
    this.logger.info('ONI RLHF System shut down complete');
  }
}

// Start the system
const system = new ONIRLHFSystem();

const port = parseInt(process.env.PORT || '3001');
system.start(port).catch((error) => {
  console.error('Failed to start ONI RLHF System:', error);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('Received SIGINT, shutting down gracefully...');
  await system.stop();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('Received SIGTERM, shutting down gracefully...');
  await system.stop();
  process.exit(0);
});

export default system;