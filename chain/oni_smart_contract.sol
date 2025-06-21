// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/**
 * @title ONI Token
 * @dev ERC20 token for the ONI ecosystem with fixed supply and proof-of-contribute rewards
 */
contract ONIToken is ERC20, Ownable, ReentrancyGuard {
    uint256 public constant TOTAL_SUPPLY = 10_000_000_000 * 10**18; // 10 billion tokens
    uint256 public constant BLOCK_REWARD = 10 * 10**18; // 10 ONI tokens per block
    
    // Contribution types
    enum ContributionType {
        TRAINING,
        FEEDBACK,
        VALIDATION,
        COMPUTE,
        DATA_PROVISION
    }
    
    // Proof of Contribute structure
    struct ProofOfContribute {
        string contributorId;
        ContributionType contributionType;
        uint256 computeHours;
        uint256 dataContributed;
        uint256 feedbackProvided;
        uint256 qualityScore;
        uint256 rewardTokens;
        uint256 timestamp;
        bool verified;
        bytes32 merkleRoot;
    }
    
    // Training session structure
    struct TrainingSession {
        string sessionId;
        string modelId;
        string trainerId;
        uint256 startTime;
        uint256 endTime;
        bool completed;
        string finalMetrics;
        uint256 totalRewards;
    }
    
    // Human feedback structure
    struct HumanFeedback {
        string feedbackId;
        string sessionId;
        string userId;
        uint256 rating;
        uint256 compassionScore;
        uint256 timestamp;
        bool verified;
        uint256 rewardTokens;
    }
    
    // Mappings
    mapping(string => ProofOfContribute[]) public contributorProofs;
    mapping(string => uint256) public contributorRewards;
    mapping(string => TrainingSession) public trainingSessions;
    mapping(string => HumanFeedback[]) public sessionFeedback;
    mapping(address => bool) public authorizedValidators;
    mapping(string => bool) public processedContributions;
    
    // Events
    event ContributionRecorded(
        string indexed contributorId,
        ContributionType contributionType,
        uint256 computeHours,
        uint256 qualityScore,
        uint256 rewardTokens
    );
    
    event FeedbackRecorded(
        string indexed sessionId,
        string indexed userId,
        uint256 rating,
        uint256 compassionScore,
        uint256 rewardTokens
    );
    
    event TrainingSessionStarted(
        string indexed sessionId,
        string indexed modelId,
        string indexed trainerId
    );
    
    event TrainingSessionCompleted(
        string indexed sessionId,
        string finalMetrics,
        uint256 totalRewards
    );
    
    event RewardsDistributed(
        string indexed contributorId,
        uint256 amount
    );
    
    event ValidatorAuthorized(address indexed validator);
    event ValidatorRevoked(address indexed validator);
    
    constructor() ERC20("ONI Token", "ONI") {
        _mint(address(this), TOTAL_SUPPLY);
        authorizedValidators[msg.sender] = true;
    }
    
    modifier onlyValidator() {
        require(authorizedValidators[msg.sender], "Not an authorized validator");
        _;
    }
    
    modifier validContributor(string memory contributorId) {
        require(bytes(contributorId).length > 0, "Invalid contributor ID");
        _;
    }
    
    modifier validSession(string memory sessionId) {
        require(bytes(sessionId).length > 0, "Invalid session ID");
        _;
    }
    
    /**
     * @dev Record a training contribution with proof
     */
    function recordContribution(
        string memory sessionId,
        string memory modelId,
        string memory contributorId,
        ContributionType contributionType,
        uint256 computeHours,
        uint256 qualityScore
    ) external onlyValidator validContributor(contributorId) {
        require(qualityScore <= 100, "Quality score must be <= 100");
        
        // Calculate reward tokens based on contribution
        uint256 rewardTokens = calculateRewardTokens(
            contributionType,
            computeHours,
            qualityScore
        );
        
        // Create proof of contribute
        ProofOfContribute memory proof = ProofOfContribute({
            contributorId: contributorId,
            contributionType: contributionType,
            computeHours: computeHours,
            dataContributed: 0, // Set based on contribution type
            feedbackProvided: 0, // Set based on contribution type
            qualityScore: qualityScore,
            rewardTokens: rewardTokens,
            timestamp: block.timestamp,
            verified: true,
            merkleRoot: bytes32(0) // Will be set by off-chain process
        });
        
        // Store the proof
        contributorProofs[contributorId].push(proof);
        contributorRewards[contributorId] += rewardTokens;
        
        // Transfer tokens to contributor (if they have a wallet address)
        // This would require a mapping from contributorId to wallet address
        // For now, tokens are held in contract until claimed
        
        emit ContributionRecorded(
            contributorId,
            contributionType,
            computeHours,
            qualityScore,
            rewardTokens
        );
    }
    
    /**
     * @dev Record human feedback for a training session
     */
    function recordFeedback(
        string memory sessionId,
        string memory feedbackId,
        string memory userId,
        uint256 rating,
        uint256 compassionScore
    ) external onlyValidator validSession(sessionId) {
        require(rating >= 1 && rating <= 10, "Rating must be between 1 and 10");
        require(compassionScore <= 100, "Compassion score must be <= 100");
        
        // Calculate feedback reward
        uint256 rewardTokens = calculateFeedbackReward(rating, compassionScore);
        
        // Create feedback record
        HumanFeedback memory feedback = HumanFeedback({
            feedbackId: feedbackId,
            sessionId: sessionId,
            userId: userId,
            rating: rating,
            compassionScore: compassionScore,
            timestamp: block.timestamp,
            verified: true,
            rewardTokens: rewardTokens
        });
        
        // Store feedback
        sessionFeedback[sessionId].push(feedback);
        contributorRewards[userId] += rewardTokens;
        
        emit FeedbackRecorded(
            sessionId,
            userId,
            rating,
            compassionScore,
            rewardTokens
        );
    }
    
    /**
     * @dev Start a new training session
     */
    function startTrainingSession(
        string memory sessionId,
        string memory modelId,
        string memory trainerId
    ) external onlyValidator validSession(sessionId) {
        require(trainingSessions[sessionId].startTime == 0, "Session already exists");
        
        trainingSessions[sessionId] = TrainingSession({
            sessionId: sessionId,
            modelId: modelId,
            trainerId: trainerId,
            startTime: block.timestamp,
            endTime: 0,
            completed: false,
            finalMetrics: "",
            totalRewards: 0
        });
        
        emit TrainingSessionStarted(sessionId, modelId, trainerId);
    }
    
    /**
     * @dev Complete a training session
     */
    function completeTrainingSession(
        string memory sessionId,
        string memory finalMetrics
    ) external onlyValidator validSession(sessionId) {
        TrainingSession storage session = trainingSessions[sessionId];
        require(session.startTime > 0, "Session does not exist");
        require(!session.completed, "Session already completed");
        
        session.endTime = block.timestamp;
        session.completed = true;
        session.finalMetrics = finalMetrics;
        
        // Calculate session rewards based on duration and quality
        uint256 sessionDuration = session.endTime - session.startTime;
        uint256 sessionRewards = calculateSessionRewards(sessionDuration, finalMetrics);
        session.totalRewards = sessionRewards;
        
        // Distribute rewards to trainer
        contributorRewards[session.trainerId] += sessionRewards;
        
        emit TrainingSessionCompleted(sessionId, finalMetrics, sessionRewards);
    }
    
    /**
     * @dev Claim accumulated rewards
     */
    function claimRewards(string memory contributorId, address walletAddress) 
        external 
        nonReentrant 
        validContributor(contributorId) 
    {
        uint256 rewards = contributorRewards[contributorId];
        require(rewards > 0, "No rewards to claim");
        require(walletAddress != address(0), "Invalid wallet address");
        
        // Reset rewards
        contributorRewards[contributorId] = 0;
        
        // Transfer tokens
        require(transfer(walletAddress, rewards), "Token transfer failed");
        
        emit RewardsDistributed(contributorId, rewards);
    }
    
    /**
     * @dev Get contributor's total rewards
     */
    function getContributorRewards(string memory contributorId) 
        external 
        view 
        returns (uint256) 
    {
        return contributorRewards[contributorId];
    }
    
    /**
     * @dev Get all contributions for a contributor
     */
    function getContributorProofs(string memory contributorId) 
        external 
        view 
        returns (ProofOfContribute[] memory) 
    {
        return contributorProofs[contributorId];
    }
    
    /**
     * @dev Get all feedback for a session
     */
    function getSessionFeedback(string memory sessionId) 
        external 
        view 
        returns (HumanFeedback[] memory) 
    {
        return sessionFeedback[sessionId];
    }
    
    /**
     * @dev Get training session details
     */
    function getTrainingSession(string memory sessionId) 
        external 
        view 
        returns (TrainingSession memory) 
    {
        return trainingSessions[sessionId];
    }
    
    /**
     * @dev Calculate reward tokens based on contribution
     */
    function calculateRewardTokens(
        ContributionType contributionType,
        uint256 computeHours,
        uint256 qualityScore
    ) internal pure returns (uint256) {
        uint256 baseReward = 10 * 10**18; // 10 ONI base reward
        uint256 typeMultiplier = getTypeMultiplier(contributionType);
        uint256 qualityMultiplier = (qualityScore * 10**16) / 100; // Scale quality to 0-1
        uint256 computeMultiplier = computeHours * 10**17; // 0.1 ONI per compute hour
        
        return baseReward + (typeMultiplier * qualityMultiplier) + computeMultiplier;
    }
    
    /**
     * @dev Get multiplier based on contribution type
     */
    function getTypeMultiplier(ContributionType contributionType) 
        internal 
        pure 
        returns (uint256) 
    {
        if (contributionType == ContributionType.TRAINING) return 20 * 10**18;
        if (contributionType == ContributionType.FEEDBACK) return 5 * 10**18;
        if (contributionType == ContributionType.VALIDATION) return 15 * 10**18;
        if (contributionType == ContributionType.COMPUTE) return 10 * 10**18;
        if (contributionType == ContributionType.DATA_PROVISION) return 25 * 10**18;
        return 10 * 10**18; // Default
    }
    
    /**
     * @dev Calculate feedback reward
     */
    function calculateFeedbackReward(uint256 rating, uint256 compassionScore) 
        internal 
        pure 
        returns (uint256) 
    {
        uint256 baseReward = 2 * 10**18; // 2 ONI base reward
        uint256 ratingBonus = (rating * 10**17); // 0.1 ONI per rating point
        uint256 compassionBonus = (compassionScore * 10**16) / 100; // Scale compassion to 0-1 ONI
        
        return baseReward + ratingBonus + compassionBonus;
    }
    
    /**
     * @dev Calculate session completion rewards
     */
    function calculateSessionRewards(uint256 duration, string memory metrics) 
        internal 
        pure 
        returns (uint256) 
    {
        uint256 baseReward = 50 * 10**18; // 50 ONI base reward
        uint256 durationBonus = (duration / 3600) * 10**18; // 1 ONI per hour
        
        // Parse metrics for quality bonus (simplified)
        // In practice, this would parse JSON metrics
        uint256 qualityBonus = 10 * 10**18; // Placeholder
        
        return baseReward + durationBonus + qualityBonus;
    }
    
    /**
     * @dev Verify proof of contribute using Merkle proof
     */
    function verifyProofOfContribute(
        string memory contributorId,
        uint256 proofIndex,
        bytes32[] memory merkleProof,
        bytes32 merkleRoot
    ) external view returns (bool) {
        require(proofIndex < contributorProofs[contributorId].length, "Invalid proof index");
        
        ProofOfContribute memory proof = contributorProofs[contributorId][proofIndex];
        
        // Create leaf hash from proof data
        bytes32 leaf = keccak256(abi.encodePacked(
            proof.contributorId,
            uint256(proof.contributionType),
            proof.computeHours,
            proof.qualityScore,
            proof.rewardTokens,
            proof.timestamp
        ));
        
        return MerkleProof.verify(merkleProof, merkleRoot, leaf);
    }
    
    /**
     * @dev Authorize a new validator
     */
    function authorizeValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = true;
        emit ValidatorAuthorized(validator);
    }
    
    /**
     * @dev Revoke validator authorization
     */
    function revokeValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = false;
        emit ValidatorRevoked(validator);
    }
    
    /**
     * @dev Emergency function to pause contract (if needed)
     */
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = balanceOf(address(this));
        require(transfer(owner(), balance), "Emergency withdrawal failed");
    }
    
    /**
     * @dev Get contract statistics
     */
    function getContractStats() external view returns (
        uint256 totalSupply_,
        uint256 contractBalance,
        uint256 totalDistributed
    ) {
        totalSupply_ = totalSupply();
        contractBalance = balanceOf(address(this));
        totalDistributed = totalSupply_ - contractBalance;
    }
}