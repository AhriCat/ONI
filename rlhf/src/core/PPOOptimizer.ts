import * as tf from '@tensorflow/tfjs-node-gpu';
import { PPOConfig } from '../types';
import { Logger } from '../utils/Logger';

export class PPOOptimizer {
  private config: PPOConfig;
  private logger: Logger;
  private optimizer: tf.Optimizer;

  constructor(config: PPOConfig) {
    this.config = config;
    this.logger = new Logger('PPOOptimizer');
    this.optimizer = tf.train.adam(0.0001);
  }

  public async update(
    model: tf.LayersModel,
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    advantages: tf.Tensor,
    oldLogProbs?: tf.Tensor
  ): Promise<tf.Tensor> {
    let totalLoss = tf.scalar(0);

    // Perform multiple PPO epochs
    for (let epoch = 0; epoch < this.config.ppoEpochs; epoch++) {
      const miniBatches = this.createMiniBatches(
        states,
        actions,
        rewards,
        advantages,
        oldLogProbs
      );

      for (const batch of miniBatches) {
        const loss = await this.computePPOLoss(model, batch);
        
        // Perform gradient update
        const grads = tf.variableGrads(() => loss, model.trainableWeights);
        
        // Clip gradients
        const clippedGrads = this.clipGradients(grads.grads, this.config.maxGradNorm);
        
        // Apply gradients
        this.optimizer.applyGradients(clippedGrads);
        
        totalLoss = tf.add(totalLoss, loss);
        
        // Cleanup
        Object.values(grads.grads).forEach(grad => grad.dispose());
        Object.values(clippedGrads).forEach(grad => grad.dispose());
        loss.dispose();
      }

      // Cleanup mini-batches
      miniBatches.forEach(batch => {
        Object.values(batch).forEach(tensor => {
          if (tensor instanceof tf.Tensor) {
            tensor.dispose();
          }
        });
      });
    }

    return totalLoss;
  }

  private createMiniBatches(
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    advantages: tf.Tensor,
    oldLogProbs?: tf.Tensor
  ): any[] {
    const batchSize = states.shape[0];
    const miniBatchSize = this.config.miniBatchSize;
    const numMiniBatches = Math.ceil(batchSize / miniBatchSize);
    
    const miniBatches = [];
    
    for (let i = 0; i < numMiniBatches; i++) {
      const start = i * miniBatchSize;
      const end = Math.min(start + miniBatchSize, batchSize);
      
      const batch = {
        states: states.slice([start, 0], [end - start, -1]),
        actions: actions.slice([start, 0], [end - start, -1]),
        rewards: rewards.slice([start, 0], [end - start, -1]),
        advantages: advantages.slice([start, 0], [end - start, -1]),
        oldLogProbs: oldLogProbs?.slice([start, 0], [end - start, -1])
      };
      
      miniBatches.push(batch);
    }
    
    return miniBatches;
  }

  private async computePPOLoss(
    model: tf.LayersModel,
    batch: any
  ): Promise<tf.Tensor> {
    return tf.tidy(() => {
      // Forward pass
      const outputs = model.predict(batch.states) as tf.Tensor;
      
      // Compute action probabilities
      const actionProbs = tf.softmax(outputs);
      const logProbs = tf.log(tf.add(actionProbs, 1e-8));
      
      // Get log probabilities for taken actions
      const actionLogProbs = tf.sum(
        tf.mul(logProbs, batch.actions),
        axis=-1,
        keepDims=true
      );
      
      // Compute probability ratio
      const ratio = batch.oldLogProbs 
        ? tf.exp(tf.sub(actionLogProbs, batch.oldLogProbs))
        : tf.ones(actionLogProbs.shape);
      
      // Compute clipped surrogate loss
      const surrogateObj = tf.mul(ratio, batch.advantages);
      const clippedRatio = tf.clipByValue(
        ratio,
        1 - this.config.clipRatio,
        1 + this.config.clipRatio
      );
      const clippedSurrogateObj = tf.mul(clippedRatio, batch.advantages);
      
      const policyLoss = tf.neg(tf.mean(tf.minimum(surrogateObj, clippedSurrogateObj)));
      
      // Compute value loss (simplified - assumes value head exists)
      const valueLoss = tf.mean(tf.square(tf.sub(outputs, batch.rewards)));
      
      // Compute entropy bonus
      const entropy = tf.neg(tf.sum(tf.mul(actionProbs, logProbs), axis=-1));
      const entropyBonus = tf.mul(tf.mean(entropy), this.config.entropyCoefficient);
      
      // Total loss
      const totalLoss = tf.add(
        tf.add(policyLoss, tf.mul(valueLoss, this.config.valueCoefficient)),
        tf.neg(entropyBonus)
      );
      
      return totalLoss;
    });
  }

  private clipGradients(
    gradients: { [varName: string]: tf.Tensor },
    maxNorm: number
  ): { [varName: string]: tf.Tensor } {
    const clippedGrads: { [varName: string]: tf.Tensor } = {};
    
    // Compute global norm
    let globalNorm = tf.scalar(0);
    Object.values(gradients).forEach(grad => {
      globalNorm = tf.add(globalNorm, tf.sum(tf.square(grad)));
    });
    globalNorm = tf.sqrt(globalNorm);
    
    // Clip gradients
    const clipCoeff = tf.minimum(tf.div(maxNorm, tf.add(globalNorm, 1e-6)), 1.0);
    
    Object.keys(gradients).forEach(varName => {
      clippedGrads[varName] = tf.mul(gradients[varName], clipCoeff);
    });
    
    globalNorm.dispose();
    clipCoeff.dispose();
    
    return clippedGrads;
  }

  public dispose(): void {
    this.optimizer.dispose();
  }
}