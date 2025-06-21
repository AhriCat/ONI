import * as tf from '@tensorflow/tfjs-node-gpu';
import { EventEmitter } from 'events';
import { TrainingSession, TrainingConfig, TrainingMetrics, HumanFeedback, ModelConfig } from '../types';
import { RewardModel } from './RewardModel';
import { PPOOptimizer } from './PPOOptimizer';
import { DataLoader } from './DataLoader';
import { MetricsCollector } from './MetricsCollector';
import { BlockchainIntegration } from '../blockchain/BlockchainIntegration';
import { CompassionFramework } from './CompassionFramework';
import { Logger } from '../utils/Logger';

export class RLHFTrainer extends EventEmitter {
  private session: TrainingSession;
  private model: tf.LayersModel;
  private rewardModel: RewardModel;
  private ppoOptimizer: PPOOptimizer;
  private dataLoader: DataLoader;
  private metricsCollector: MetricsCollector;
  private blockchain: BlockchainIntegration;
  private compassion: CompassionFramework;
  private logger: Logger;
  private isTraining: boolean = false;
  private currentStep: number = 0;

  constructor(
    session: TrainingSession,
    modelConfig: ModelConfig,
    blockchain: BlockchainIntegration
  ) {
    super();
    this.session = session;
    this.blockchain = blockchain;
    this.logger = new Logger(`RLHFTrainer-${session.id}`);
    
    this.initializeComponents(modelConfig);
  }

  private async initializeComponents(modelConfig: ModelConfig): Promise<void> {
    try {
      // Initialize model
      this.model = await this.loadModel(modelConfig);
      
      // Initialize reward model
      this.rewardModel = new RewardModel(this.session.configuration.rewardModel);
      
      // Initialize PPO optimizer
      this.ppoOptimizer = new PPOOptimizer(this.session.configuration.ppoConfig);
      
      // Initialize data loader
      this.dataLoader = new DataLoader(this.session.configuration.datasetConfig);
      
      // Initialize metrics collector
      this.metricsCollector = new MetricsCollector();
      
      // Initialize compassion framework
      this.compassion = new CompassionFramework(
        this.session.configuration.rewardModel.ethicalConstraints
      );
      
      this.logger.info('RLHF Trainer initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize RLHF Trainer:', error);
      throw error;
    }
  }

  private async loadModel(config: ModelConfig): Promise<tf.LayersModel> {
    if (config.checkpointPath) {
      return await tf.loadLayersModel(`file://${config.checkpointPath}`);
    }
    
    // Create new model based on configuration
    return this.createModel(config);
  }

  private createModel(config: ModelConfig): tf.LayersModel {
    const model = tf.sequential();
    
    switch (config.type) {
      case 'nlp':
        return this.createNLPModel(config);
      case 'vision':
        return this.createVisionModel(config);
      case 'audio':
        return this.createAudioModel(config);
      case 'multimodal':
        return this.createMultimodalModel(config);
      case 'full-oni':
        return this.createFullONIModel(config);
      default:
        throw new Error(`Unsupported model type: ${config.type}`);
    }
  }

  private createNLPModel(config: ModelConfig): tf.LayersModel {
    const model = tf.sequential({
      layers: [
        tf.layers.embedding({
          inputDim: config.hyperparameters.vocabSize || 300000,
          outputDim: config.hyperparameters.hiddenDim || 896,
          inputLength: config.hyperparameters.maxLength || 4096
        }),
        tf.layers.lstm({
          units: config.hyperparameters.hiddenDim || 896,
          returnSequences: true,
          dropout: 0.1,
          recurrentDropout: 0.1
        }),
        tf.layers.attention(),
        tf.layers.globalAveragePooling1d(),
        tf.layers.dense({
          units: config.hyperparameters.hiddenDim || 896,
          activation: 'relu'
        }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: config.outputShape[0],
          activation: 'softmax'
        })
      ]
    });
    
    return model;
  }

  private createVisionModel(config: ModelConfig): tf.LayersModel {
    const model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          filters: 32,
          kernelSize: 3,
          activation: 'relu',
          inputShape: config.inputShape as [number, number, number]
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }),
        tf.layers.globalAveragePooling2d(),
        tf.layers.dense({ units: 512, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: config.outputShape[0],
          activation: 'softmax'
        })
      ]
    });
    
    return model;
  }

  private createAudioModel(config: ModelConfig): tf.LayersModel {
    const model = tf.sequential({
      layers: [
        tf.layers.conv1d({
          filters: 64,
          kernelSize: 3,
          activation: 'relu',
          inputShape: config.inputShape as [number, number]
        }),
        tf.layers.maxPooling1d({ poolSize: 2 }),
        tf.layers.conv1d({ filters: 128, kernelSize: 3, activation: 'relu' }),
        tf.layers.maxPooling1d({ poolSize: 2 }),
        tf.layers.lstm({ units: 256, returnSequences: false }),
        tf.layers.dense({ units: 512, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: config.outputShape[0],
          activation: 'softmax'
        })
      ]
    });
    
    return model;
  }

  private createMultimodalModel(config: ModelConfig): tf.LayersModel {
    // Create separate branches for different modalities
    const textInput = tf.input({ shape: [config.hyperparameters.maxLength] });
    const imageInput = tf.input({ shape: [224, 224, 3] });
    const audioInput = tf.input({ shape: [1024, 128] });
    
    // Text branch
    const textEmbedding = tf.layers.embedding({
      inputDim: config.hyperparameters.vocabSize || 300000,
      outputDim: 512
    }).apply(textInput) as tf.SymbolicTensor;
    
    const textLSTM = tf.layers.lstm({ units: 512 }).apply(textEmbedding) as tf.SymbolicTensor;
    
    // Image branch
    const imageConv = tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu'
    }).apply(imageInput) as tf.SymbolicTensor;
    
    const imagePool = tf.layers.globalAveragePooling2d().apply(imageConv) as tf.SymbolicTensor;
    
    // Audio branch
    const audioConv = tf.layers.conv1d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu'
    }).apply(audioInput) as tf.SymbolicTensor;
    
    const audioLSTM = tf.layers.lstm({ units: 512 }).apply(audioConv) as tf.SymbolicTensor;
    
    // Fusion layer
    const concatenated = tf.layers.concatenate().apply([textLSTM, imagePool, audioLSTM]) as tf.SymbolicTensor;
    const fusion = tf.layers.dense({ units: 1024, activation: 'relu' }).apply(concatenated) as tf.SymbolicTensor;
    const output = tf.layers.dense({
      units: config.outputShape[0],
      activation: 'softmax'
    }).apply(fusion) as tf.SymbolicTensor;
    
    return tf.model({ inputs: [textInput, imageInput, audioInput], outputs: output });
  }

  private createFullONIModel(config: ModelConfig): tf.LayersModel {
    // This would create the full ONI architecture
    // For now, we'll create a simplified version
    const input = tf.input({ shape: config.inputShape });
    
    // Emotional processing layer
    const emotional = tf.layers.dense({ units: 128, activation: 'tanh', name: 'emotional_layer' });
    
    // Memory attention layer
    const memory = tf.layers.attention({ name: 'memory_attention' });
    
    // Compassion evaluation layer
    const compassion = tf.layers.dense({ units: 64, activation: 'sigmoid', name: 'compassion_layer' });
    
    // Meta-cognition layer
    const metacognition = tf.layers.dense({ units: 32, activation: 'relu', name: 'metacognition_layer' });
    
    // Main processing pipeline
    let x = tf.layers.dense({ units: 896, activation: 'relu' }).apply(input) as tf.SymbolicTensor;
    x = emotional.apply(x) as tf.SymbolicTensor;
    x = tf.layers.dropout({ rate: 0.1 }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.dense({ units: 896, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    
    const output = tf.layers.dense({
      units: config.outputShape[0],
      activation: 'softmax'
    }).apply(x) as tf.SymbolicTensor;
    
    return tf.model({ inputs: input, outputs: output });
  }

  public async startTraining(): Promise<void> {
    if (this.isTraining) {
      throw new Error('Training is already in progress');
    }

    this.isTraining = true;
    this.session.status = 'active';
    this.session.startTime = new Date();
    
    this.logger.info('Starting RLHF training session');
    this.emit('trainingStarted', this.session);

    try {
      await this.blockchain.recordTrainingStart(this.session);
      
      for (let epoch = 0; epoch < this.session.configuration.epochs; epoch++) {
        await this.trainEpoch(epoch);
        
        if (!this.isTraining) break; // Check for early stopping
        
        // Validate and save checkpoint
        await this.validateAndSave(epoch);
      }
      
      await this.completeTraining();
    } catch (error) {
      await this.handleTrainingError(error);
    }
  }

  private async trainEpoch(epoch: number): Promise<void> {
    this.logger.info(`Starting epoch ${epoch + 1}/${this.session.configuration.epochs}`);
    
    const batches = await this.dataLoader.getBatches(this.session.configuration.batchSize);
    
    for (const batch of batches) {
      if (!this.isTraining) break;
      
      // Forward pass
      const predictions = this.model.predict(batch.inputs) as tf.Tensor;
      
      // Get rewards from reward model
      const rewards = await this.rewardModel.computeRewards(
        batch.inputs,
        predictions,
        batch.humanFeedback
      );
      
      // Apply compassion framework
      const compassionScores = await this.compassion.evaluateCompassion(
        batch.inputs,
        predictions,
        batch.context
      );
      
      // Combine rewards with compassion scores
      const adjustedRewards = this.combineRewards(rewards, compassionScores);
      
      // PPO update
      const loss = await this.ppoOptimizer.update(
        this.model,
        batch.inputs,
        predictions,
        adjustedRewards,
        batch.advantages
      );
      
      // Collect metrics
      const metrics = await this.metricsCollector.collect({
        loss: await loss.data(),
        rewards: await adjustedRewards.data(),
        compassionScores: await compassionScores.data(),
        predictions: await predictions.data()
      });
      
      // Update session metrics
      this.session.metrics = metrics;
      this.session.currentStep = ++this.currentStep;
      
      // Emit progress update
      this.emit('trainingProgress', {
        session: this.session,
        epoch,
        step: this.currentStep,
        metrics
      });
      
      // Record contribution to blockchain
      await this.blockchain.recordContribution({
        sessionId: this.session.id,
        step: this.currentStep,
        metrics,
        timestamp: new Date()
      });
      
      // Cleanup tensors
      predictions.dispose();
      rewards.dispose();
      compassionScores.dispose();
      adjustedRewards.dispose();
      loss.dispose();
    }
  }

  private combineRewards(
    rewards: tf.Tensor,
    compassionScores: tf.Tensor
  ): tf.Tensor {
    const compassionWeight = this.session.configuration.rewardModel.compassionWeight;
    return tf.add(
      tf.mul(rewards, 1 - compassionWeight),
      tf.mul(compassionScores, compassionWeight)
    );
  }

  private async validateAndSave(epoch: number): Promise<void> {
    // Run validation
    const validationMetrics = await this.runValidation();
    
    // Save checkpoint
    const checkpointPath = `./checkpoints/${this.session.id}/epoch_${epoch}`;
    await this.model.save(`file://${checkpointPath}`);
    
    // Update blockchain with checkpoint
    await this.blockchain.recordCheckpoint({
      sessionId: this.session.id,
      epoch,
      checkpointPath,
      metrics: validationMetrics,
      timestamp: new Date()
    });
    
    this.emit('checkpointSaved', {
      session: this.session,
      epoch,
      checkpointPath,
      metrics: validationMetrics
    });
  }

  private async runValidation(): Promise<TrainingMetrics> {
    const validationData = await this.dataLoader.getValidationData();
    const predictions = this.model.predict(validationData.inputs) as tf.Tensor;
    
    return await this.metricsCollector.collect({
      predictions: await predictions.data(),
      targets: await validationData.targets.data(),
      isValidation: true
    });
  }

  private async completeTraining(): Promise<void> {
    this.isTraining = false;
    this.session.status = 'completed';
    this.session.endTime = new Date();
    
    // Final blockchain record
    await this.blockchain.recordTrainingCompletion(this.session);
    
    this.logger.info('Training completed successfully');
    this.emit('trainingCompleted', this.session);
  }

  private async handleTrainingError(error: any): Promise<void> {
    this.isTraining = false;
    this.session.status = 'failed';
    this.session.endTime = new Date();
    
    this.logger.error('Training failed:', error);
    this.emit('trainingFailed', { session: this.session, error });
    
    await this.blockchain.recordTrainingFailure(this.session, error.message);
  }

  public async pauseTraining(): Promise<void> {
    this.isTraining = false;
    this.session.status = 'paused';
    
    this.logger.info('Training paused');
    this.emit('trainingPaused', this.session);
  }

  public async resumeTraining(): Promise<void> {
    if (this.session.status !== 'paused') {
      throw new Error('Can only resume paused training sessions');
    }
    
    this.isTraining = true;
    this.session.status = 'active';
    
    this.logger.info('Training resumed');
    this.emit('trainingResumed', this.session);
  }

  public async addHumanFeedback(feedback: HumanFeedback): Promise<void> {
    // Validate feedback
    await this.validateFeedback(feedback);
    
    // Add to reward model
    await this.rewardModel.addFeedback(feedback);
    
    // Record on blockchain
    await this.blockchain.recordFeedback(feedback);
    
    this.emit('feedbackAdded', feedback);
  }

  private async validateFeedback(feedback: HumanFeedback): Promise<void> {
    // Implement feedback validation logic
    if (feedback.humanRating < 1 || feedback.humanRating > 10) {
      throw new Error('Human rating must be between 1 and 10');
    }
    
    if (feedback.compassionRating < 0 || feedback.compassionRating > 1) {
      throw new Error('Compassion rating must be between 0 and 1');
    }
  }

  public getSession(): TrainingSession {
    return { ...this.session };
  }

  public getMetrics(): TrainingMetrics {
    return { ...this.session.metrics };
  }

  public async dispose(): Promise<void> {
    this.isTraining = false;
    this.model?.dispose();
    this.removeAllListeners();
  }
}