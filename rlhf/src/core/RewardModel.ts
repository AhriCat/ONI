import * as tf from '@tensorflow/tfjs-node-gpu';
import { RewardModelConfig, HumanFeedback, TrainingMetrics } from '../types';
import { Logger } from '../utils/Logger';

export class RewardModel {
  private config: RewardModelConfig;
  private model: tf.LayersModel | null = null;
  private feedbackBuffer: HumanFeedback[] = [];
  private logger: Logger;

  constructor(config: RewardModelConfig) {
    this.config = config;
    this.logger = new Logger('RewardModel');
    this.initializeModel();
  }

  private async initializeModel(): Promise<void> {
    if (this.config.modelPath) {
      try {
        this.model = await tf.loadLayersModel(`file://${this.config.modelPath}`);
        this.logger.info('Loaded existing reward model');
      } catch (error) {
        this.logger.warn('Failed to load existing model, creating new one');
        this.createNewModel();
      }
    } else {
      this.createNewModel();
    }
  }

  private createNewModel(): void {
    // Create a simple reward model architecture
    this.model = tf.sequential({
      layers: [
        tf.layers.dense({
          units: 512,
          activation: 'relu',
          inputShape: [1024] // Adjust based on input features
        }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: 256,
          activation: 'relu'
        }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: 128,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid' // Output reward score between 0 and 1
        })
      ]
    });

    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    this.logger.info('Created new reward model');
  }

  public async computeRewards(
    inputs: tf.Tensor,
    predictions: tf.Tensor,
    humanFeedback?: HumanFeedback[]
  ): Promise<tf.Tensor> {
    if (!this.model) {
      throw new Error('Reward model not initialized');
    }

    // Extract features from inputs and predictions
    const features = this.extractFeatures(inputs, predictions);
    
    // Get AI-based rewards
    const aiRewards = this.model.predict(features) as tf.Tensor;
    
    // Combine with human feedback if available
    if (humanFeedback && humanFeedback.length > 0) {
      const humanRewards = this.processHumanFeedback(humanFeedback);
      return this.combineRewards(aiRewards, humanRewards);
    }
    
    return aiRewards;
  }

  private extractFeatures(inputs: tf.Tensor, predictions: tf.Tensor): tf.Tensor {
    // Extract relevant features for reward computation
    // This is a simplified version - in practice, this would be more sophisticated
    
    const inputFeatures = tf.mean(inputs, axis=-1, keepDims=true);
    const predictionFeatures = tf.mean(predictions, axis=-1, keepDims=true);
    const confidence = tf.max(predictions, axis=-1, keepDims=true);
    const entropy = this.computeEntropy(predictions);
    
    return tf.concat([inputFeatures, predictionFeatures, confidence, entropy], axis=-1);
  }

  private computeEntropy(predictions: tf.Tensor): tf.Tensor {
    // Compute entropy of predictions as a measure of uncertainty
    const logProbs = tf.log(tf.add(predictions, 1e-8));
    const entropy = tf.neg(tf.sum(tf.mul(predictions, logProbs), axis=-1, keepDims=true));
    return entropy;
  }

  private processHumanFeedback(feedback: HumanFeedback[]): tf.Tensor {
    // Convert human feedback to tensor format
    const ratings = feedback.map(f => f.humanRating / 10); // Normalize to 0-1
    const compassionRatings = feedback.map(f => f.compassionRating);
    
    // Combine ratings with compassion scores
    const combinedRatings = ratings.map((rating, i) => 
      rating * this.config.humanFeedbackWeight + 
      compassionRatings[i] * this.config.compassionWeight
    );
    
    return tf.tensor2d([combinedRatings]);
  }

  private combineRewards(aiRewards: tf.Tensor, humanRewards: tf.Tensor): tf.Tensor {
    const aiWeight = this.config.aiRewardWeight;
    const humanWeight = this.config.humanFeedbackWeight;
    
    return tf.add(
      tf.mul(aiRewards, aiWeight),
      tf.mul(humanRewards, humanWeight)
    );
  }

  public async addFeedback(feedback: HumanFeedback): Promise<void> {
    this.feedbackBuffer.push(feedback);
    
    // Retrain model if we have enough feedback
    if (this.feedbackBuffer.length >= 100) {
      await this.retrainModel();
      this.feedbackBuffer = []; // Clear buffer after retraining
    }
  }

  private async retrainModel(): Promise<void> {
    if (!this.model || this.feedbackBuffer.length === 0) {
      return;
    }

    this.logger.info(`Retraining reward model with ${this.feedbackBuffer.length} feedback samples`);

    // Prepare training data from feedback
    const { inputs, targets } = this.prepareFeedbackData();
    
    // Train the model
    await this.model.fit(inputs, targets, {
      epochs: 10,
      batchSize: 32,
      validationSplit: 0.2,
      verbose: 0
    });

    this.logger.info('Reward model retrained successfully');
  }

  private prepareFeedbackData(): { inputs: tf.Tensor; targets: tf.Tensor } {
    // This is a simplified version - in practice, you'd need to reconstruct
    // the original inputs and predictions from the feedback
    const feedbackData = this.feedbackBuffer.map(feedback => ({
      features: this.extractFeaturesFromFeedback(feedback),
      rating: feedback.humanRating / 10 // Normalize to 0-1
    }));

    const inputs = tf.stack(feedbackData.map(d => d.features));
    const targets = tf.tensor2d(feedbackData.map(d => [d.rating]));

    return { inputs, targets };
  }

  private extractFeaturesFromFeedback(feedback: HumanFeedback): tf.Tensor {
    // Extract features from feedback for training
    // This is a placeholder - in practice, you'd need more sophisticated feature extraction
    const textLength = feedback.inputText.length / 1000; // Normalize
    const responseLength = feedback.modelResponse.length / 1000; // Normalize
    const compassionScore = feedback.compassionRating;
    const categoryScores = feedback.categories.map(c => c.score);
    
    const features = [textLength, responseLength, compassionScore, ...categoryScores];
    
    // Pad or truncate to fixed size
    while (features.length < 1024) {
      features.push(0);
    }
    
    return tf.tensor1d(features.slice(0, 1024));
  }

  public async saveModel(path: string): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }
    
    await this.model.save(`file://${path}`);
    this.logger.info(`Reward model saved to ${path}`);
  }

  public async loadModel(path: string): Promise<void> {
    this.model = await tf.loadLayersModel(`file://${path}`);
    this.logger.info(`Reward model loaded from ${path}`);
  }

  public dispose(): void {
    this.model?.dispose();
  }
}