export interface ModelConfig {
  id: string;
  name: string;
  type: 'nlp' | 'vision' | 'audio' | 'multimodal' | 'full-oni';
  architecture: string;
  parameters: number;
  inputShape: number[];
  outputShape: number[];
  hyperparameters: Record<string, any>;
  checkpointPath?: string;
  huggingfaceRepo?: string;
}

export interface TrainingSession {
  id: string;
  modelId: string;
  userId: string;
  status: 'pending' | 'active' | 'paused' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  totalSteps: number;
  currentStep: number;
  metrics: TrainingMetrics;
  configuration: TrainingConfig;
  blockchain: BlockchainInfo;
}

export interface TrainingConfig {
  batchSize: number;
  learningRate: number;
  epochs: number;
  validationSplit: number;
  rewardModel: RewardModelConfig;
  ppoConfig: PPOConfig;
  datasetConfig: DatasetConfig;
  distributedConfig?: DistributedConfig;
}

export interface RewardModelConfig {
  type: 'human' | 'ai' | 'hybrid';
  modelPath?: string;
  humanFeedbackWeight: number;
  aiRewardWeight: number;
  compassionWeight: number;
  ethicalConstraints: EthicalConstraints;
}

export interface PPOConfig {
  clipRatio: number;
  valueClipRatio: number;
  entropyCoefficient: number;
  valueCoefficient: number;
  maxGradNorm: number;
  ppoEpochs: number;
  miniBatchSize: number;
  targetKL: number;
}

export interface DatasetConfig {
  type: 'conversation' | 'instruction' | 'preference' | 'multimodal';
  sources: string[];
  preprocessing: PreprocessingConfig;
  augmentation?: AugmentationConfig;
}

export interface PreprocessingConfig {
  tokenizer: string;
  maxLength: number;
  padding: boolean;
  truncation: boolean;
  specialTokens: Record<string, string>;
}

export interface AugmentationConfig {
  enabled: boolean;
  techniques: string[];
  probability: number;
}

export interface DistributedConfig {
  enabled: boolean;
  nodes: number;
  gpusPerNode: number;
  communicationBackend: 'nccl' | 'gloo' | 'mpi';
}

export interface TrainingMetrics {
  loss: number;
  rewardScore: number;
  compassionScore: number;
  ethicalScore: number;
  perplexity: number;
  bleuScore?: number;
  rougeScore?: number;
  humanRating?: number;
  convergenceRate: number;
  gradientNorm: number;
  learningRate: number;
  memoryUsage: number;
  throughput: number;
}

export interface HumanFeedback {
  id: string;
  sessionId: string;
  userId: string;
  timestamp: Date;
  inputText: string;
  modelResponse: string;
  humanRating: number; // 1-10 scale
  humanPreference?: 'model_a' | 'model_b' | 'tie';
  categories: FeedbackCategory[];
  comments?: string;
  ethicalConcerns?: string[];
  compassionRating: number;
  verified: boolean;
  blockchainHash?: string;
}

export interface FeedbackCategory {
  name: string;
  score: number;
  weight: number;
}

export interface BlockchainInfo {
  networkId: string;
  contractAddress: string;
  transactionHash?: string;
  blockNumber?: number;
  gasUsed?: number;
  proofOfContribute: ProofOfContribute;
}

export interface ProofOfContribute {
  contributorId: string;
  contributionType: 'training' | 'feedback' | 'validation' | 'compute';
  computeHours: number;
  dataContributed: number;
  feedbackProvided: number;
  qualityScore: number;
  rewardTokens: number;
  timestamp: Date;
  verified: boolean;
  merkleProof?: string[];
}

export interface EthicalConstraints {
  maxHarmScore: number;
  minCompassionScore: number;
  biasThresholds: Record<string, number>;
  prohibitedTopics: string[];
  requiredDisclosures: string[];
}

export interface ModelWeights {
  format: 'pytorch' | 'tensorflow' | 'onnx' | 'huggingface';
  compression: 'none' | 'gzip' | 'brotli' | 'quantized';
  precision: 'fp32' | 'fp16' | 'int8' | 'int4';
  sharding?: ShardingInfo;
  metadata: WeightMetadata;
}

export interface ShardingInfo {
  totalShards: number;
  shardSize: number;
  shardingStrategy: 'layer' | 'parameter' | 'tensor';
}

export interface WeightMetadata {
  modelId: string;
  version: string;
  trainingSteps: number;
  performance: TrainingMetrics;
  compatibility: string[];
  license: string;
  author: string;
  description: string;
  tags: string[];
  huggingfaceConfig?: any;
}

export interface DownloadRequest {
  modelId: string;
  format: string;
  compression: string;
  precision: string;
  includeOptimizer: boolean;
  includeScheduler: boolean;
  userId: string;
  purpose: 'research' | 'commercial' | 'personal';
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'contributor' | 'trainer' | 'validator' | 'admin';
  reputation: number;
  totalContributions: number;
  verificationLevel: 'unverified' | 'email' | 'kyc' | 'expert';
  walletAddress?: string;
  apiKey: string;
  permissions: string[];
  createdAt: Date;
  lastActive: Date;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
  requestId: string;
}

export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: Date;
  sessionId: string;
  userId?: string;
}

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'down';
  activeTrainingSessions: number;
  totalModels: number;
  queuedJobs: number;
  systemLoad: number;
  memoryUsage: number;
  gpuUtilization: number[];
  blockchainSync: boolean;
  lastUpdate: Date;
}