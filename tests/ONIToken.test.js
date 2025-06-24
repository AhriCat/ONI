const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ONIToken", function () {
  let ONIToken;
  let oniToken;
  let owner;
  let validator;
  let user1;
  let user2;

  const TOTAL_SUPPLY = ethers.parseEther("10000000000"); // 10 billion tokens
  const BLOCK_REWARD = ethers.parseEther("10"); // 10 tokens per block

  beforeEach(async function () {
    // Get signers
    [owner, validator, user1, user2] = await ethers.getSigners();

    // Deploy the contract
    ONIToken = await ethers.getContractFactory("ONIToken");
    oniToken = await ONIToken.deploy();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await oniToken.owner()).to.equal(owner.address);
    });

    it("Should assign the total supply to the contract", async function () {
      const contractBalance = await oniToken.balanceOf(await oniToken.getAddress());
      expect(contractBalance).to.equal(TOTAL_SUPPLY);
    });

    it("Should authorize the deployer as a validator", async function () {
      expect(await oniToken.authorizedValidators(owner.address)).to.equal(true);
    });

    it("Should create a genesis block", async function () {
      const [index, , , , , , ,] = await oniToken.getBlock(0);
      expect(index).to.equal(0);
    });
  });

  describe("Validator Management", function () {
    it("Should allow owner to authorize a new validator", async function () {
      await oniToken.authorizeValidator(validator.address);
      expect(await oniToken.authorizedValidators(validator.address)).to.equal(true);
    });

    it("Should allow owner to revoke a validator", async function () {
      await oniToken.authorizeValidator(validator.address);
      await oniToken.revokeValidator(validator.address);
      expect(await oniToken.authorizedValidators(validator.address)).to.equal(false);
    });

    it("Should prevent non-owners from authorizing validators", async function () {
      await expect(
        oniToken.connect(user1).authorizeValidator(user2.address)
      ).to.be.revertedWithCustomError(oniToken, "OwnableUnauthorizedAccount");
    });
  });

  describe("Contribution Recording", function () {
    beforeEach(async function () {
      // Authorize validator
      await oniToken.authorizeValidator(validator.address);
    });

    it("Should record a training contribution", async function () {
      await oniToken.connect(validator).recordContribution(
        "session1",
        "model1",
        "contributor1",
        0, // TRAINING
        10, // 10 compute hours
        80 // 80% quality score
      );

      const proofs = await oniToken.getContributorProofs("contributor1");
      expect(proofs.length).to.equal(1);
    });

    it("Should calculate and assign rewards correctly", async function () {
      await oniToken.connect(validator).recordContribution(
        "session1",
        "model1",
        "contributor1",
        0, // TRAINING
        10, // 10 compute hours
        80 // 80% quality score
      );

      const rewards = await oniToken.getContributorRewards("contributor1");
      expect(rewards).to.be.gt(0);
    });

    it("Should prevent duplicate contributions", async function () {
      // Use the same session ID and timestamp for both calls
      const sessionId = "session1";
      const contributorId = "contributor1";
      
      // First contribution should succeed
      await oniToken.connect(validator).recordContribution(
        sessionId,
        "model1",
        contributorId,
        0, // TRAINING
        10, // 10 compute hours
        80 // 80% quality score
      );
      
      // Second identical contribution should fail
      // Note: In a real test, we would need to mock the timestamp to be the same
      await expect(
        oniToken.connect(validator).recordContribution(
          sessionId,
          "model1",
          contributorId,
          0, // TRAINING
          10, // 10 compute hours
          80 // 80% quality score
        )
      ).to.be.revertedWith("Contribution already processed");
    });
  });

  describe("Feedback Recording", function () {
    beforeEach(async function () {
      // Authorize validator
      await oniToken.authorizeValidator(validator.address);
    });

    it("Should record feedback correctly", async function () {
      await oniToken.connect(validator).recordFeedback(
        "session1",
        "feedback1",
        "user1",
        8, // 8/10 rating
        75 // 75% compassion score
      );

      const feedback = await oniToken.getSessionFeedback("session1");
      expect(feedback.length).to.equal(1);
      expect(feedback[0].rating).to.equal(8);
      expect(feedback[0].compassionScore).to.equal(75);
    });

    it("Should validate feedback rating range", async function () {
      await expect(
        oniToken.connect(validator).recordFeedback(
          "session1",
          "feedback1",
          "user1",
          11, // Invalid: > 10
          75
        )
      ).to.be.revertedWith("Rating must be between 1 and 10");

      await expect(
        oniToken.connect(validator).recordFeedback(
          "session1",
          "feedback1",
          "user1",
          0, // Invalid: < 1
          75
        )
      ).to.be.revertedWith("Rating must be between 1 and 10");
    });
  });

  describe("Training Sessions", function () {
    beforeEach(async function () {
      // Authorize validator
      await oniToken.authorizeValidator(validator.address);
    });

    it("Should start a training session", async function () {
      await oniToken.connect(validator).startTrainingSession(
        "session1",
        "model1",
        "trainer1"
      );

      const session = await oniToken.getTrainingSession("session1");
      expect(session.modelId).to.equal("model1");
      expect(session.trainerId).to.equal("trainer1");
      expect(session.completed).to.equal(false);
    });

    it("Should complete a training session", async function () {
      await oniToken.connect(validator).startTrainingSession(
        "session1",
        "model1",
        "trainer1"
      );

      await oniToken.connect(validator).completeTrainingSession(
        "session1",
        '{"accuracy":0.95,"loss":0.05}'
      );

      const session = await oniToken.getTrainingSession("session1");
      expect(session.completed).to.equal(true);
      expect(session.finalMetrics).to.equal('{"accuracy":0.95,"loss":0.05}');
      expect(session.totalRewards).to.be.gt(0);
    });

    it("Should prevent completing a non-existent session", async function () {
      await expect(
        oniToken.connect(validator).completeTrainingSession(
          "nonexistent",
          '{"accuracy":0.95,"loss":0.05}'
        )
      ).to.be.revertedWith("Session does not exist");
    });
  });

  describe("Reward Claiming", function () {
    beforeEach(async function () {
      // Authorize validator
      await oniToken.authorizeValidator(validator.address);

      // Record a contribution to generate rewards
      await oniToken.connect(validator).recordContribution(
        "session1",
        "model1",
        "contributor1",
        0, // TRAINING
        10, // 10 compute hours
        80 // 80% quality score
      );
    });

    it("Should allow claiming rewards", async function () {
      const initialRewards = await oniToken.getContributorRewards("contributor1");
      expect(initialRewards).to.be.gt(0);

      await oniToken.claimRewards("contributor1", user1.address);

      // Check user received tokens
      const userBalance = await oniToken.balanceOf(user1.address);
      expect(userBalance).to.equal(initialRewards);

      // Check rewards were reset
      const finalRewards = await oniToken.getContributorRewards("contributor1");
      expect(finalRewards).to.equal(0);
    });

    it("Should prevent claiming if no rewards", async function () {
      await expect(
        oniToken.claimRewards("no_rewards", user1.address)
      ).to.be.revertedWith("No rewards to claim");
    });
  });

  describe("Transaction Processing", function () {
    beforeEach(async function () {
      // Authorize validator
      await oniToken.authorizeValidator(validator.address);
    });

    it("Should process a transaction", async function () {
      const txHash = await oniToken.connect(validator).processTransaction(
        user1.address,
        user2.address,
        ethers.parseEther("1"),
        ethers.parseUnits("1", "gwei"), // 1 gwei gas price
        100000, // gas limit
        1, // nonce
        "0x" // empty data
      );

      expect(txHash).to.not.be.null;
    });

    it("Should mine a block with transactions", async function () {
      // Process a transaction
      await oniToken.connect(validator).processTransaction(
        user1.address,
        user2.address,
        ethers.parseEther("1"),
        ethers.parseUnits("1", "gwei"),
        100000,
        1,
        "0x"
      );

      // Mine the block
      await oniToken.connect(validator).mineBlock();

      // Check block was created
      const [index, , , , , , , txCount] = await oniToken.getBlock(1);
      expect(index).to.equal(1);
      expect(txCount).to.equal(1);
    });
  });

  describe("Chain Statistics", function () {
    it("Should return correct chain statistics", async function () {
      const stats = await oniToken.getChainStats();
      expect(stats.blockCount).to.equal(1); // Genesis block
      expect(stats.txCount).to.equal(0);
      expect(stats.difficulty).to.be.gt(0);
      expect(stats.gasPrice).to.be.gt(0);
    });
  });

  describe("Contract Stats", function () {
    it("Should return correct contract statistics", async function () {
      const [totalSupply, contractBalance, totalDistributed] = await oniToken.getContractStats();
      expect(totalSupply).to.equal(TOTAL_SUPPLY);
      expect(contractBalance).to.equal(TOTAL_SUPPLY);
      expect(totalDistributed).to.equal(0);
    });
  });
});
