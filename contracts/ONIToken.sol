// SPDX-License-Identifier: Pantheum
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
    uint256 public constant MAX_GAS_PER_BLOCK = 10_000_000; // Maximum gas per block
    uint256 public constant BASE_GAS_PRICE = 1 gwei; // Very low base gas price
    
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
    
    // Transaction structure
    struct Transaction {
        address sender;
        address recipient;
        uint256 amount;
        uint256 gasPrice;
        uint256 gasLimit;
        uint256 nonce;
        uint256 timestamp;
        bytes data;
        bytes32 txHash;
    }
    
    // Block structure
    struct Block {
        uint256 index;
        bytes32 previousHash;
        uint256 timestamp;
        bytes32 merkleRoot;
        uint256 nonce;
        uint256 difficulty;
        uint256 gasUsed;
        Transaction[] transactions;
    }
    
    // Mappings
    mapping(string => ProofOfContribute[]) public contributorProofs;
    mapping(string => uint256) public contributorRewards;
    mapping(string => TrainingSession) public trainingSessions;
    mapping(string => HumanFeedback[]) public sessionFeedback;
    mapping(address => bool) public authorizedValidators;
    mapping(string => bool) public processedContributions;
    mapping(bytes32 => bool) public processedTransactions;
    mapping(uint256 => Block) public blocks;
    
    // Chain state
    uint256 public currentBlockIndex;
    bytes32 public currentBlockHash;
    uint256 public currentDifficulty = 2; // Initial difficulty (2 leading zeros)
    uint256 public targetBlockTime = 3 seconds; // Target time for block creation
    uint256 public lastBlockTime;
    uint256 public totalTransactions;
    
    // Gas price state
    uint256 public currentGasPrice = BASE_GAS_PRICE;
    uint256 public minGasPrice = 100 wei; // Absolute minimum gas price
    uint256 public maxGasPrice = 10 gwei; // Maximum gas price
    
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
    
    event BlockMined(
        uint256 indexed blockIndex,
        bytes32 blockHash,
        uint256 difficulty,
        uint256 gasUsed,
        uint256 transactionCount
    );
    
    event TransactionProcessed(
        bytes32 indexed txHash,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 gasUsed,
        uint256 fee
    );
    
    event GasPriceUpdated(
        uint256 oldGasPrice,
        uint256 newGasPrice
    );
    
    event DifficultyAdjusted(
        uint256 oldDifficulty,
        uint256 newDifficulty,
        uint256 blockTime
    );
    
    constructor() ERC20("ONI Token", "ONI") {
        _mint(address(this), TOTAL_SUPPLY);
        authorizedValidators[msg.sender] = true;
        
        // Create genesis block
        Block storage genesisBlock = blocks[0];
        genesisBlock.index = 0;
        genesisBlock.previousHash = bytes32(0);
        genesisBlock.timestamp = block.timestamp;
        genesisBlock.merkleRoot = bytes32(0);
        genesisBlock.nonce = 0;
        genesisBlock.difficulty = currentDifficulty;
        genesisBlock.gasUsed = 0;
        
        // Set current block state
        currentBlockIndex = 0;
        currentBlockHash = keccak256(abi.encodePacked(genesisBlock.index, genesisBlock.previousHash, genesisBlock.timestamp, genesisBlock.merkleRoot, genesisBlock.nonce));
        lastBlockTime = block.timestamp;
        
        emit BlockMined(0, currentBlockHash, currentDifficulty, 0, 0);
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
        
        // Generate unique contribution ID
        string memory contributionId = string(abi.encodePacked(contributorId, "_", sessionId, "_", uint2str(block.timestamp)));
        require(!processedContributions[contributionId], "Contribution already processed");
        
        // Mark as processed
        processedContributions[contributionId] = true;
        
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
        
        // Check if feedback already processed
        require(!processedContributions[feedbackId], "Feedback already processed");
        
        // Mark as processed
        processedContributions[feedbackId] = true;
        
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
     * @dev Process a transaction and add it to the current block
     */
    function processTransaction(
        address sender,
        address recipient,
        uint256 amount,
        uint256 gasPrice,
        uint256 gasLimit,
        uint256 nonce,
        bytes calldata data
    ) external onlyValidator returns (bytes32) {
        // Create transaction
        Transaction memory tx = Transaction({
            sender: sender,
            recipient: recipient,
            amount: amount,
            gasPrice: gasPrice,
            gasLimit: gasLimit,
            nonce: nonce,
            timestamp: block.timestamp,
            data: data,
            txHash: bytes32(0)
        });
        
        // Calculate transaction hash
        tx.txHash = keccak256(abi.encodePacked(
            sender,
            recipient,
            amount,
            gasPrice,
            gasLimit,
            nonce,
            block.timestamp,
            data
        ));
        
        // Check if transaction already processed
        require(!processedTransactions[tx.txHash], "Transaction already processed");
        
        // Mark as processed
        processedTransactions[tx.txHash] = true;
        
        // Calculate gas used (simplified)
        uint256 gasUsed = calculateGasUsed(tx);
        require(gasUsed <= gasLimit, "Gas limit exceeded");
        
        // Calculate fee
        uint256 fee = gasUsed * gasPrice;
        
        // Add transaction to current block
        Block storage currentBlock = blocks[currentBlockIndex + 1];
        if (currentBlock.index == 0) {
            // Initialize new block
            currentBlock.index = currentBlockIndex + 1;
            currentBlock.previousHash = currentBlockHash;
            currentBlock.timestamp = block.timestamp;
            currentBlock.difficulty = currentDifficulty;
        }
        
        // Add transaction to block
        currentBlock.transactions.push(tx);
        currentBlock.gasUsed += gasUsed;
        
        // Update total transactions
        totalTransactions++;
        
        // Check if block is full or it's time to mine
        if (currentBlock.gasUsed >= MAX_GAS_PER_BLOCK || 
            currentBlock.transactions.length >= 1000 ||
            block.timestamp >= lastBlockTime + 10) { // Mine at least every 10 seconds
            
            // Mine the block
            mineBlock();
        }
        
        emit TransactionProcessed(
            tx.txHash,
            sender,
            recipient,
            amount,
            gasUsed,
            fee
        );
        
        return tx.txHash;
    }
    
    /**
     * @dev Mine the current block
     */
    function mineBlock() public onlyValidator {
        Block storage currentBlock = blocks[currentBlockIndex + 1];
        require(currentBlock.index > 0, "No block to mine");
        require(currentBlock.transactions.length > 0, "No transactions to mine");
        
        // Calculate merkle root
        currentBlock.merkleRoot = calculateMerkleRoot(currentBlock.transactions);
        
        // Find proof of work (simplified)
        uint256 nonce = 0;
        bytes32 blockHash;
        bytes32 target = bytes32(uint256(2) ** (256 - currentDifficulty * 4) - 1);
        
        do {
            blockHash = keccak256(abi.encodePacked(
                currentBlock.index,
                currentBlock.previousHash,
                currentBlock.timestamp,
                currentBlock.merkleRoot,
                nonce
            ));
            nonce++;
        } while (uint256(blockHash) > uint256(target) && nonce < 1000000); // Limit iterations
        
        // Set block nonce and update chain state
        currentBlock.nonce = nonce - 1;
        
        // Update chain state
        uint256 newBlockIndex = currentBlockIndex + 1;
        bytes32 newBlockHash = blockHash;
        
        // Calculate block time and adjust difficulty if needed
        uint256 blockTime = block.timestamp - lastBlockTime;
        if (newBlockIndex % 100 == 0) {
            adjustDifficulty(blockTime);
        }
        
        // Update gas price based on block fullness
        adjustGasPrice(currentBlock.gasUsed);
        
        // Update state
        currentBlockIndex = newBlockIndex;
        currentBlockHash = newBlockHash;
        lastBlockTime = block.timestamp;
        
        emit BlockMined(
            currentBlockIndex,
            currentBlockHash,
            currentDifficulty,
            currentBlock.gasUsed,
            currentBlock.transactions.length
        );
    }
    
    /**
     * @dev Adjust difficulty based on block time
     */
    function adjustDifficulty(uint256 blockTime) internal {
        uint256 oldDifficulty = currentDifficulty;
        
        if (blockTime < targetBlockTime / 2) {
            // Blocks are too fast, increase difficulty
            currentDifficulty = currentDifficulty + 1;
        } else if (blockTime > targetBlockTime * 2) {
            // Blocks are too slow, decrease difficulty
            currentDifficulty = currentDifficulty > 1 ? currentDifficulty - 1 : 1;
        }
        
        emit DifficultyAdjusted(oldDifficulty, currentDifficulty, blockTime);
    }
    
    /**
     * @dev Adjust gas price based on block fullness
     */
    function adjustGasPrice(uint256 gasUsed) internal {
        uint256 oldGasPrice = currentGasPrice;
        
        // Calculate block fullness percentage
        uint256 fullness = (gasUsed * 100) / MAX_GAS_PER_BLOCK;
        
        if (fullness > 80) {
            // Block is nearly full, increase gas price
            currentGasPrice = (currentGasPrice * 110) / 100; // +10%
        } else if (fullness < 30) {
            // Block is mostly empty, decrease gas price
            currentGasPrice = (currentGasPrice * 90) / 100; // -10%
        }
        
        // Ensure gas price stays within bounds
        if (currentGasPrice < minGasPrice) {
            currentGasPrice = minGasPrice;
        } else if (currentGasPrice > maxGasPrice) {
            currentGasPrice = maxGasPrice;
        }
        
        if (currentGasPrice != oldGasPrice) {
            emit GasPriceUpdated(oldGasPrice, currentGasPrice);
        }
    }
    
    /**
     * @dev Calculate gas used by a transaction
     */
    function calculateGasUsed(Transaction memory tx) internal pure returns (uint256) {
        uint256 baseGas = 21000; // Base gas for a standard transaction
        
        // Add gas for data
        if (tx.data.length > 0) {
            uint256 zeroBytes = 0;
            uint256 nonZeroBytes = 0;
            
            for (uint256 i = 0; i < tx.data.length; i++) {
                if (tx.data[i] == 0) {
                    zeroBytes++;
                } else {
                    nonZeroBytes++;
                }
            }
            
            baseGas += zeroBytes * 4; // 4 gas for each zero byte
            baseGas += nonZeroBytes * 16; // 16 gas for each non-zero byte
        }
        
        return baseGas;
    }
    
    /**
     * @dev Calculate merkle root of transactions
     */
    function calculateMerkleRoot(Transaction[] memory transactions) internal pure returns (bytes32) {
        if (transactions.length == 0) {
            return bytes32(0);
        }
        
        bytes32[] memory leaves = new bytes32[](transactions.length);
        for (uint256 i = 0; i < transactions.length; i++) {
            leaves[i] = transactions[i].txHash;
        }
        
        return merkleRoot(leaves);
    }
    
    /**
     * @dev Calculate merkle root from leaves
     */
    function merkleRoot(bytes32[] memory leaves) internal pure returns (bytes32) {
        if (leaves.length == 0) {
            return bytes32(0);
        }
        
        if (leaves.length == 1) {
            return leaves[0];
        }
        
        bytes32[] memory nextLevel = new bytes32[]((leaves.length + 1) / 2);
        
        for (uint256 i = 0; i < nextLevel.length; i++) {
            uint256 leftIdx = i * 2;
            uint256 rightIdx = leftIdx + 1;
            
            bytes32 left = leaves[leftIdx];
            bytes32 right = rightIdx < leaves.length ? leaves[rightIdx] : left;
            
            nextLevel[i] = keccak256(abi.encodePacked(left, right));
        }
        
        return merkleRoot(nextLevel);
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
     * @dev Get block details
     */
    function getBlock(uint256 blockIndex) 
        external 
        view 
        returns (
            uint256 index,
            bytes32 previousHash,
            uint256 timestamp,
            bytes32 merkleRoot,
            uint256 nonce,
            uint256 difficulty,
            uint256 gasUsed,
            uint256 transactionCount
        ) 
    {
        Block storage block_ = blocks[blockIndex];
        return (
            block_.index,
            block_.previousHash,
            block_.timestamp,
            block_.merkleRoot,
            block_.nonce,
            block_.difficulty,
            block_.gasUsed,
            block_.transactions.length
        );
    }
    
    /**
     * @dev Get transaction details
     */
    function getTransaction(uint256 blockIndex, uint256 txIndex) 
        external 
        view 
        returns (
            bytes32 txHash,
            address sender,
            address recipient,
            uint256 amount,
            uint256 gasPrice,
            uint256 gasLimit,
            uint256 nonce,
            uint256 timestamp
        ) 
    {
        require(blockIndex <= currentBlockIndex, "Block index out of range");
        Block storage block_ = blocks[blockIndex];
        require(txIndex < block_.transactions.length, "Transaction index out of range");
        
        Transaction storage tx = block_.transactions[txIndex];
        return (
            tx.txHash,
            tx.sender,
            tx.recipient,
            tx.amount,
            tx.gasPrice,
            tx.gasLimit,
            tx.nonce,
            tx.timestamp
        );
    }
    
    /**
     * @dev Get chain statistics
     */
    function getChainStats() 
        external 
        view 
        returns (
            uint256 blockCount,
            uint256 txCount,
            uint256 difficulty,
            uint256 gasPrice,
            uint256 lastBlockTime,
            uint256 targetTime
        ) 
    {
        return (
            currentBlockIndex + 1,
            totalTransactions,
            currentDifficulty,
            currentGasPrice,
            lastBlockTime,
            targetBlockTime
        );
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
    
    /**
     * @dev Convert uint to string
     */
    function uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 length;
        while (j != 0) {
            length++;
            j /= 10;
        }
        bytes memory bstr = new bytes(length);
        uint256 k = length;
        j = _i;
        while (j != 0) {
            bstr[--k] = bytes1(uint8(48 + j % 10));
            j /= 10;
        }
        return string(bstr);
    }
}