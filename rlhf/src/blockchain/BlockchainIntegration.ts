import Web3 from 'web3';
import { Contract } from 'web3-eth-contract';
import { TrainingSession, ProofOfContribute, HumanFeedback } from '../types';
import { Logger } from '../utils/Logger';

export class BlockchainIntegration {
  private web3: Web3;
  private contract: Contract;
  private logger: Logger;
  private networkId: string;
  private contractAddress: string;

  constructor() {
    this.logger = new Logger('BlockchainIntegration');
    this.networkId = process.env.BLOCKCHAIN_NETWORK_ID || 'oni-testnet';
    this.contractAddress = process.env.CONTRACT_ADDRESS || '';
    
    this.initializeWeb3();
    this.initializeContract();
  }

  private initializeWeb3(): void {
    const rpcUrl = process.env.BLOCKCHAIN_RPC_URL || 'http://localhost:8545';
    this.web3 = new Web3(new Web3.providers.HttpProvider(rpcUrl));
    
    // Add account if private key is provided
    const privateKey = process.env.BLOCKCHAIN_PRIVATE_KEY;
    if (privateKey) {
      this.web3.eth.accounts.wallet.add(privateKey);
    }
  }

  private initializeContract(): void {
    const contractABI = this.getContractABI();
    this.contract = new this.web3.eth.Contract(contractABI, this.contractAddress);
  }

  private getContractABI(): any[] {
    // ONI Proof of Contribute Contract ABI
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
      },
      {
        "inputs": [
          {"name": "_sessionId", "type": "string"}
        ],
        "name": "getSessionContributions",
        "outputs": [
          {
            "components": [
              {"name": "contributorId", "type": "string"},
              {"name": "contributionType", "type": "uint8"},
              {"name": "computeHours", "type": "uint256"},
              {"name": "qualityScore", "type": "uint256"},
              {"name": "rewardTokens", "type": "uint256"},
              {"name": "timestamp", "type": "uint256"}
            ],
            "name": "",
            "type": "tuple[]"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      }
    ];
  }

  public async recordTrainingStart(session: TrainingSession): Promise<string> {
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

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
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      // Calculate contribution metrics
      const computeHours = this.calculateComputeHours(contribution.metrics);
      const qualityScore = this.calculateQualityScore(contribution.metrics);

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
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

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
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      const metricsJson = JSON.stringify(session.metrics);

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

      // Record as a contribution with checkpoint type
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      const tx = await this.contract.methods.recordContribution(
        checkpoint.sessionId,
        'checkpoint',
        fromAccount,
        2, // checkpoint contribution type
        0, // no compute hours for checkpoint
        this.web3.utils.toWei(this.calculateQualityScore(checkpoint.metrics).toString(), 'ether')
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
    try {
      const accounts = await this.web3.eth.getAccounts();
      const fromAccount = accounts[0];

      const failureData = JSON.stringify({
        sessionId: session.id,
        error: errorMessage,
        timestamp: new Date().toISOString(),
        finalMetrics: session.metrics
      });

      // Record as a special completion with failure status
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
    try {
      const rewards = await this.contract.methods.getContributorRewards(contributorId).call();
      return parseFloat(this.web3.utils.fromWei(rewards, 'ether'));
    } catch (error) {
      this.logger.error('Failed to get contributor rewards:', error);
      return 0;
    }
  }

  public async getSessionContributions(sessionId: string): Promise<ProofOfContribute[]> {
    try {
      const contributions = await this.contract.methods.getSessionContributions(sessionId).call();
      
      return contributions.map((contrib: any) => ({
        contributorId: contrib.contributorId,
        contributionType: this.mapContributionType(contrib.contributionType),
        computeHours: parseFloat(this.web3.utils.fromWei(contrib.computeHours, 'ether')),
        dataContributed: 0, // Would need additional tracking
        feedbackProvided: 0, // Would need additional tracking
        qualityScore: parseFloat(this.web3.utils.fromWei(contrib.qualityScore, 'ether')),
        rewardTokens: parseFloat(this.web3.utils.fromWei(contrib.rewardTokens, 'ether')),
        timestamp: new Date(contrib.timestamp * 1000),
        verified: true, // On-chain contributions are verified
        merkleProof: [] // Would be generated if needed
      }));
    } catch (error) {
      this.logger.error('Failed to get session contributions:', error);
      return [];
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

  private mapContributionType(typeId: number): string {
    const types = ['training', 'feedback', 'validation', 'compute'];
    return types[typeId] || 'unknown';
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

  private calculateRewardTokens(computeHours: number, qualityScore: number): number {
    // Base reward calculation
    const baseReward = 10; // Base tokens per hour
    const qualityMultiplier = 1 + qualityScore; // 1x to 2x multiplier
    
    return computeHours * baseReward * qualityMultiplier;
  }

  public async verifyProofOfContribute(proof: ProofOfContribute): Promise<boolean> {
    try {
      // Verify the proof exists on blockchain
      const contributions = await this.getSessionContributions(proof.contributorId);
      
      return contributions.some(contrib => 
        contrib.contributorId === proof.contributorId &&
        contrib.contributionType === proof.contributionType &&
        Math.abs(contrib.computeHours - proof.computeHours) < 0.001 &&
        Math.abs(contrib.qualityScore - proof.qualityScore) < 0.001
      );
    } catch (error) {
      this.logger.error('Failed to verify proof of contribute:', error);
      return false;
    }
  }

  public dispose(): void {
    // Cleanup blockchain connections
    if (this.web3.currentProvider && typeof this.web3.currentProvider.disconnect === 'function') {
      this.web3.currentProvider.disconnect();
    }
  }
}