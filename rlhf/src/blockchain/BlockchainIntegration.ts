import Web3 from 'web3';
import { Contract } from 'web3-eth-contract';
import { AbiItem } from 'web3-utils';
import { TrainingSession, ProofOfContribute, HumanFeedback } from '../types';
import { Logger } from '../utils/Logger';
import * as fs from 'fs-extra';
import * as path from 'path';
import * as crypto from 'crypto';

export class BlockchainIntegration {
  private web3: Web3;
  private contract: Contract;
  private logger: Logger;
  private networkId: string;
  private contractAddress: string;
  private contractAbi: AbiItem[];
  private isInitialized: boolean = false;
  private contributionsCache: Map<string, any> = new Map();
  private syncInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.logger = new Logger('BlockchainIntegration');
    this.networkId = process.env.BLOCKCHAIN_NETWORK_ID || 'oni-testnet';
    this.contractAddress = process.env.CONTRACT_ADDRESS || '';
    
    // Load contract ABI
    this.contractAbi = this.loadContractAbi();
    
    // Initialize Web3 and contract
    this.initializeBlockchain();
    
    // Start periodic sync
    this.startPeriodicSync();
  }

  private loadContractAbi(): AbiItem[] {
    try {
      // Try to load from file
      const abiPath = process.env.CONTRACT_ABI_PATH || path.join(__dirname, '../../../chain/oni_contract_abi.json');
      if (fs.existsSync(abiPath)) {
        return JSON.parse(fs.readFileSync(abiPath, 'utf8'));
      }
      
      // Fallback to embedded ABI
      return [
        {
          "inputs": [
            {"name": "_sessionId", "type": "string"},
            {"name": "_modelId", "type": "string"},
            {"name": "_contributorId", "type": "string"},
            {"name": "_contributionType", "type": "uint8"},
            {"name": "_computeHours", "type": "uint256"},
            {"name": "_qualityScore", "type": "uint256"}
          ],
          "name": "recordContribution",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {"name": "_sessionId", "type": "string"},
            {"name": "_feedbackId", "type": "string"},
            {"name": "_userId", "type": "string"},
            {"name": "_rating", "type": "uint256"},
            {"name": "_compassionScore", "type": "uint256"}
          ],
          "name": "recordFeedback",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {"name": "_sessionId", "type": "string"},
            {"name": "_modelId", "type": "string"},
            {"name": "_trainerId", "type": "string"}
          ],
          "name": "startTrainingSession",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {"name": "_sessionId", "type": "string"},
            {"name": "_finalMetrics", "type": "string"}
          ],
          "name": "completeTrainingSession",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {"name": "_contributorId", "type": "string"}
          ],
          "name": "getContributorRewards",
          "outputs": [{"name": "", "type": "uint256"}],
          "stateMutability": "view",
          "type": "function"
        }
      ];
    } catch (error) {
      this.logger.error('Failed to load contract ABI:', error);
      return [];
    }
  }

  private initializeBlockchain(): void {
    try {
      // Initialize Web3
      const rpcUrl = process.env.BLOCKCHAIN_RPC_URL || 'http://localhost:8545';
      this.web3 = new Web3(new Web3.providers.HttpProvider(rpcUrl));
      
      // Initialize contract
      this.contract = new this.web3.eth.Contract(this.contractAbi, this.contractAddress);
      
      // Add account if private key is provided
      const privateKey = process.env.BLOCKCHAIN_PRIVATE_KEY;
      if (privateKey) {
        const account = this.web3.eth.accounts.privateKeyToAccount(privateKey);
        this.web3.eth.accounts.wallet.add(account);
        this.logger.info(`Added account: ${account.address}`);
      }
      
      this.isInitialized = true;
      this.logger.info('Blockchain integration initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize blockchain:', error);
      this.isInitialized = false;
    }
  }

  private startPeriodicSync(): void {
    const syncIntervalMs = parseInt(process.env.BLOCKCHAIN_SYNC_INTERVAL || '300000', 10); // Default: 5 minutes
    
    this.syncInterval = setInterval(() => {
      this.syncContributions().catch(error => {
        this.logger.error('Error during periodic sync:', error);
      });
    }, syncIntervalMs);
    
    this.logger.info(`Started periodic sync with interval: ${syncIntervalMs}ms`);
  }

  private async syncContributions(): Promise<void> {
    if (!this.isInitialized) {
      this.logger.warn('Cannot sync contributions: blockchain not initialized');
      return;
    }
    
    try {
      this.logger.info('Syncing contributions from blockchain...');
      
      // In a real implementation, this would query the blockchain for all contributions
      // For this example, we'll just log the sync attempt
      
      this.logger.info('Contributions synced successfully');
    } catch (error) {
      this.logger.error('Failed to sync contributions:', error);
      throw error;
    }
  }

  public async recordTrainingStart(session: TrainingSession): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - don't actually send transaction
        const txHash = this.simulateTransaction('startTrainingSession', [
          session.id,
          session.modelId,
          session.userId
        ]);
        
        this.logger.info(`[SIMULATION] Training session ${session.id} recorded on blockchain`, {
          transactionHash: txHash
        });
        
        return txHash;
      }
      
      // Real blockchain transaction
      const tx = await this.contract.methods.startTrainingSession(
        session.id,
        session.modelId,
        session.userId
      ).send({
        from: fromAccount,
        gas: 200000
      });

      this.logger.info(`Training session ${session.id} recorded on blockchain`, {
        transactionHash: tx.transactionHash,
        blockNumber: tx.blockNumber
      });

      return tx.transactionHash;
    } catch (error) {
      this.logger.error('Failed to record training start on blockchain:', error);
      throw error;
    }
  }

  public async recordContribution(contribution: {
    sessionId: string;
    step: number;
    metrics: any;
    timestamp: Date;
  }): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      // Calculate contribution metrics
      const computeHours = this.calculateComputeHours(contribution.metrics);
      const qualityScore = this.calculateQualityScore(contribution.metrics);
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - don't actually send transaction
        const txHash = this.simulateTransaction('recordContribution', [
          contribution.sessionId,
          'training_model', // modelId placeholder
          fromAccount, // contributorId
          0, // training contribution type
          this.web3.utils.toWei(computeHours.toString(), 'ether'),
          this.web3.utils.toWei(qualityScore.toString(), 'ether')
        ]);
        
        this.logger.info(`[SIMULATION] Contribution recorded on blockchain`, {
          sessionId: contribution.sessionId,
          step: contribution.step,
          transactionHash: txHash
        });
        
        return txHash;
      }

      // Real blockchain transaction
      const tx = await this.contract.methods.recordContribution(
        contribution.sessionId,
        'training_model', // modelId placeholder
        fromAccount, // contributorId
        0, // training contribution type
        this.web3.utils.toWei(computeHours.toString(), 'ether'),
        this.web3.utils.toWei(qualityScore.toString(), 'ether')
      ).send({
        from: fromAccount,
        gas: 150000
      });

      this.logger.info(`Contribution recorded on blockchain`, {
        sessionId: contribution.sessionId,
        step: contribution.step,
        transactionHash: tx.transactionHash
      });

      return tx.transactionHash;
    } catch (error) {
      this.logger.error('Failed to record contribution on blockchain:', error);
      throw error;
    }
  }

  public async recordFeedback(feedback: HumanFeedback): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - don't actually send transaction
        const txHash = this.simulateTransaction('recordFeedback', [
          feedback.sessionId,
          feedback.id,
          feedback.userId,
          Math.floor(feedback.humanRating * 100), // Scale to avoid decimals
          Math.floor(feedback.compassionRating * 100)
        ]);
        
        this.logger.info(`[SIMULATION] Feedback ${feedback.id} recorded on blockchain`, {
          transactionHash: txHash
        });
        
        return txHash;
      }

      // Real blockchain transaction
      const tx = await this.contract.methods.recordFeedback(
        feedback.sessionId,
        feedback.id,
        feedback.userId,
        Math.floor(feedback.humanRating * 100), // Scale to avoid decimals
        Math.floor(feedback.compassionRating * 100)
      ).send({
        from: fromAccount,
        gas: 120000
      });

      this.logger.info(`Feedback ${feedback.id} recorded on blockchain`, {
        transactionHash: tx.transactionHash,
        blockNumber: tx.blockNumber
      });

      return tx.transactionHash;
    } catch (error) {
      this.logger.error('Failed to record feedback on blockchain:', error);
      throw error;
    }
  }

  public async recordTrainingCompletion(session: TrainingSession): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];
      
      const metricsJson = JSON.stringify(session.metrics);
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - don't actually send transaction
        const txHash = this.simulateTransaction('completeTrainingSession', [
          session.id,
          metricsJson
        ]);
        
        this.logger.info(`[SIMULATION] Training completion recorded on blockchain`, {
          sessionId: session.id,
          transactionHash: txHash
        });
        
        return txHash;
      }

      // Real blockchain transaction
      const tx = await this.contract.methods.completeTrainingSession(
        session.id,
        metricsJson
      ).send({
        from: fromAccount,
        gas: 180000
      });

      this.logger.info(`Training completion recorded on blockchain`, {
        sessionId: session.id,
        transactionHash: tx.transactionHash
      });

      return tx.transactionHash;
    } catch (error) {
      this.logger.error('Failed to record training completion on blockchain:', error);
      throw error;
    }
  }

  public async recordCheckpoint(checkpoint: {
    sessionId: string;
    epoch: number;
    checkpointPath: string;
    metrics: any;
    timestamp: Date;
  }): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      // Create a hash of the checkpoint for verification
      const checkpointHash = this.web3.utils.keccak256(
        JSON.stringify({
          sessionId: checkpoint.sessionId,
          epoch: checkpoint.epoch,
          metrics: checkpoint.metrics,
          timestamp: checkpoint.timestamp.toISOString()
        })
      );
      
      // Calculate quality score from metrics
      const qualityScore = this.calculateQualityScore(checkpoint.metrics);
      
      // Record as a contribution with checkpoint type
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - don't actually send transaction
        const txHash = this.simulateTransaction('recordContribution', [
          checkpoint.sessionId,
          'checkpoint',
          fromAccount,
          2, // checkpoint contribution type
          0, // no compute hours for checkpoint
          this.web3.utils.toWei(qualityScore.toString(), 'ether')
        ]);
        
        this.logger.info(`[SIMULATION] Checkpoint recorded on blockchain`, {
          sessionId: checkpoint.sessionId,
          epoch: checkpoint.epoch,
          hash: checkpointHash,
          transactionHash: txHash
        });
        
        return txHash;
      }

      // Real blockchain transaction
      const tx = await this.contract.methods.recordContribution(
        checkpoint.sessionId,
        'checkpoint',
        fromAccount,
        2, // checkpoint contribution type
        0, // no compute hours for checkpoint
        this.web3.utils.toWei(qualityScore.toString(), 'ether')
      ).send({
        from: fromAccount,
        gas: 150000
      });

      this.logger.info(`Checkpoint recorded on blockchain`, {
        sessionId: checkpoint.sessionId,
        epoch: checkpoint.epoch,
        hash: checkpointHash,
        transactionHash: tx.transactionHash
      });

      return tx.transactionHash;
    } catch (error) {
      this.logger.error('Failed to record checkpoint on blockchain:', error);
      throw error;
    }
  }

  public async recordTrainingFailure(session: TrainingSession, errorMessage: string): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      const failureData = JSON.stringify({
        sessionId: session.id,
        error: errorMessage,
        timestamp: new Date().toISOString(),
        finalMetrics: session.metrics
      });
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - don't actually send transaction
        const txHash = this.simulateTransaction('completeTrainingSession', [
          session.id,
          failureData
        ]);
        
        this.logger.info(`[SIMULATION] Training failure recorded on blockchain`, {
          sessionId: session.id,
          error: errorMessage,
          transactionHash: txHash
        });
        
        return txHash;
      }

      // Real blockchain transaction
      const tx = await this.contract.methods.completeTrainingSession(
        session.id,
        failureData
      ).send({
        from: fromAccount,
        gas: 180000
      });

      this.logger.info(`Training failure recorded on blockchain`, {
        sessionId: session.id,
        error: errorMessage,
        transactionHash: tx.transactionHash
      });

      return tx.transactionHash;
    } catch (error) {
      this.logger.error('Failed to record training failure on blockchain:', error);
      throw error;
    }
  }

  public async getContributorRewards(contributorId: string): Promise<number> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      // Check cache first
      if (this.contributionsCache.has(`rewards_${contributorId}`)) {
        const cachedValue = this.contributionsCache.get(`rewards_${contributorId}`);
        if (cachedValue.timestamp > Date.now() - 300000) { // 5 minute cache
          return cachedValue.value;
        }
      }
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - return simulated value
        const simulatedRewards = Math.floor(Math.random() * 1000);
        
        // Cache the result
        this.contributionsCache.set(`rewards_${contributorId}`, {
          value: simulatedRewards,
          timestamp: Date.now()
        });
        
        return simulatedRewards;
      }
      
      // Real blockchain query
      const rewards = await this.contract.methods.getContributorRewards(contributorId).call();
      const rewardsValue = parseFloat(this.web3.utils.fromWei(rewards, 'ether'));
      
      // Cache the result
      this.contributionsCache.set(`rewards_${contributorId}`, {
        value: rewardsValue,
        timestamp: Date.now()
      });
      
      return rewardsValue;
    } catch (error) {
      this.logger.error('Failed to get contributor rewards:', error);
      return 0;
    }
  }

  public async getSessionContributions(sessionId: string): Promise<ProofOfContribute[]> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      // Check cache first
      if (this.contributionsCache.has(`contributions_${sessionId}`)) {
        const cachedValue = this.contributionsCache.get(`contributions_${sessionId}`);
        if (cachedValue.timestamp > Date.now() - 300000) { // 5 minute cache
          return cachedValue.value;
        }
      }
      
      // Check if we're using a real blockchain or simulation mode
      if (process.env.BLOCKCHAIN_SIMULATION === 'true') {
        // Simulation mode - return simulated contributions
        const simulatedContributions = this.generateSimulatedContributions(sessionId);
        
        // Cache the result
        this.contributionsCache.set(`contributions_${sessionId}`, {
          value: simulatedContributions,
          timestamp: Date.now()
        });
        
        return simulatedContributions;
      }
      
      // Real blockchain query
      // This would query the blockchain for contributions related to the session
      // For now, return an empty array
      return [];
    } catch (error) {
      this.logger.error('Failed to get session contributions:', error);
      return [];
    }
  }

  private generateSimulatedContributions(sessionId: string): ProofOfContribute[] {
    // Generate some simulated contributions for testing
    const contributions: ProofOfContribute[] = [];
    const contributionTypes = ['training', 'feedback', 'validation', 'compute'];
    const numContributions = Math.floor(Math.random() * 10) + 1;
    
    for (let i = 0; i < numContributions; i++) {
      const contributionType = contributionTypes[Math.floor(Math.random() * contributionTypes.length)] as any;
      const computeHours = Math.random() * 10;
      const qualityScore = Math.random();
      
      contributions.push({
        contributorId: `contributor_${Math.floor(Math.random() * 10)}`,
        contributionType,
        computeHours,
        dataContributed: Math.floor(Math.random() * 1000),
        feedbackProvided: Math.floor(Math.random() * 10),
        qualityScore,
        rewardTokens: computeHours * 10 * (1 + qualityScore),
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 86400000)),
        verified: Math.random() > 0.2,
        merkleProof: []
      });
    }
    
    return contributions;
  }

  public async createProofOfContribute(
    contributorId: string,
    sessionId: string,
    contributionData: any
  ): Promise<ProofOfContribute> {
    const computeHours = this.calculateComputeHours(contributionData.metrics);
    const qualityScore = this.calculateQualityScore(contributionData.metrics);
    
    // Calculate reward tokens based on contribution
    const rewardTokens = this.calculateRewardTokens(computeHours, qualityScore);
    
    const proof: ProofOfContribute = {
      contributorId,
      contributionType: contributionData.type || 'training',
      computeHours,
      dataContributed: contributionData.dataSize || 0,
      feedbackProvided: contributionData.feedbackCount || 0,
      qualityScore,
      rewardTokens,
      timestamp: new Date(),
      verified: false,
      merkleProof: []
    };
    
    // Record on blockchain
    try {
      const txHash = await this.recordContribution({
        sessionId,
        step: contributionData.step || 0,
        metrics: contributionData.metrics,
        timestamp: proof.timestamp
      });
      
      proof.verified = true;
      this.logger.info(`Proof of contribute created and verified`, {
        contributorId,
        sessionId,
        txHash
      });
    } catch (error) {
      this.logger.error('Failed to verify proof of contribute on blockchain:', error);
    }
    
    return proof;
  }

  public async verifyProofOfContribute(proof: ProofOfContribute): Promise<boolean> {
    if (!this.isInitialized) {
      throw new Error('Blockchain not initialized');
    }
    
    try {
      // In a real implementation, this would verify the proof on the blockchain
      // For this example, we'll just return true if the proof is marked as verified
      return proof.verified;
    } catch (error) {
      this.logger.error('Failed to verify proof of contribute:', error);
      return false;
    }
  }

  private calculateComputeHours(metrics: any): number {
    // Calculate compute hours based on metrics
    // This is a simplified calculation
    const baseComputeTime = 0.1; // Base time per step
    const complexityMultiplier = metrics.loss ? (1 / Math.max(metrics.loss, 0.1)) : 1;
    return baseComputeTime * complexityMultiplier;
  }

  private calculateQualityScore(metrics: any): number {
    // Calculate quality score based on metrics
    let score = 0.5; // Base score
    
    if (metrics.rewardScore) {
      score += metrics.rewardScore * 0.3;
    }
    
    if (metrics.compassionScore) {
      score += metrics.compassionScore * 0.3;
    }
    
    if (metrics.ethicalScore) {
      score += metrics.ethicalScore * 0.2;
    }
    
    return Math.min(Math.max(score, 0), 1); // Clamp between 0 and 1
  }

  private calculateRewardTokens(computeHours: number, qualityScore: number): number {
    // Base reward calculation
    const baseReward = 10; // Base tokens per hour
    const qualityMultiplier = 1 + qualityScore; // 1x to 2x multiplier
    
    return computeHours * baseReward * qualityMultiplier;
  }

  private simulateTransaction(method: string, params: any[]): string {
    // Generate a fake transaction hash for simulation
    const txData = JSON.stringify({
      method,
      params,
      timestamp: Date.now()
    });
    
    return '0x' + crypto.createHash('sha256').update(txData).digest('hex');
  }

  public async isSynced(): Promise<boolean> {
    // Check if blockchain is synced
    if (!this.isInitialized) {
      return false;
    }
    
    try {
      // In a real implementation, this would check blockchain sync status
      // For this example, we'll just return true
      return true;
    } catch (error) {
      this.logger.error('Failed to check blockchain sync status:', error);
      return false;
    }
  }

  public async isHealthy(): Promise<boolean> {
    // Check if blockchain integration is healthy
    if (!this.isInitialized) {
      return false;
    }
    
    try {
      // Check if we can connect to the blockchain
      await this.web3.eth.getBlockNumber();
      return true;
    } catch (error) {
      this.logger.error('Blockchain health check failed:', error);
      return false;
    }
  }

  public dispose(): void {
    // Clean up resources
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }
    
    // Close web3 connection if possible
    if (this.web3 && this.web3.currentProvider && typeof (this.web3.currentProvider as any).disconnect === 'function') {
      (this.web3.currentProvider as any).disconnect();
    }
    
    this.logger.info('Blockchain integration disposed');
  }
}