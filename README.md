# ONI Blockchain

A high-performance, low-cost blockchain designed specifically for the ONI ecosystem, featuring Proof of Compute consensus and optimized for AI model training and contribution tracking.

## Features

- **Ultra-Low Gas Fees**: Designed for high throughput with minimal transaction costs
- **Fast Block Times**: 3-second target block time for near-instant confirmations
- **Proof of Compute Consensus**: Rewards AI model training and contributions
- **Dynamic Difficulty Adjustment**: Maintains consistent block times
- **Adaptive Gas Pricing**: Automatically adjusts based on network demand
- **Efficient Memory Pool**: Prioritizes transactions by fee
- **Batch Transaction Processing**: Reduces overhead for multiple operations
- **Model Update Tracking**: Specialized support for AI model versioning
- **Contribution Verification**: Cryptographic proof of AI training contributions
- **Smart Contract Support**: Full EVM compatibility with ONI Token contract

## Getting Started

### Prerequisites

- Python 3.8+ for the Python implementation
- Node.js 16+ for the Ethereum smart contract
- Hardhat for contract deployment

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahricat/oni.git
   cd oni
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies:
   ```bash
   npm install
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Blockchain Node

Start the ONI blockchain node:

```bash
python -m chain.oni_blockchain_api
```

This will start the blockchain API server on the configured host and port (default: http://localhost:5000).

### Deploying the Smart Contract

Deploy the ONI Token smart contract to your chosen network:

```bash
npx hardhat run scripts/deploy.js --network oni
```

## API Endpoints

The ONI blockchain exposes the following RESTful API endpoints:

- **GET /health**: Health check endpoint
- **GET /balance/{address}**: Get balance for an address
- **POST /transactions/new**: Create a new transaction
- **POST /transactions/batch**: Process a batch of transactions
- **POST /mine**: Mine a new block
- **GET /chain**: Get the full blockchain
- **GET /transactions/pending**: Get pending transactions
- **GET /transaction/{tx_hash}**: Get transaction details
- **GET /models/updates**: Get all model updates
- **GET /models/updates/{model_id}**: Get model updates for a specific model
- **GET /models/download/{update_id}**: Download a model update
- **GET /stats**: Get blockchain statistics
- **POST /verify**: Verify a proof of compute
- **POST /nodes/register**: Register new nodes in the network
- **GET /nodes/resolve**: Resolve conflicts between nodes
- **GET /gas-price**: Get the current gas price
- **POST /estimate-gas**: Estimate gas for a transaction
- **POST /proof-of-compute/submit**: Submit proof of compute for a model update
- **GET /node/status**: Get node status

## Smart Contract Functions

The ONI Token smart contract provides the following functions:

- **recordContribution**: Record a training contribution with proof
- **recordFeedback**: Record human feedback for a training session
- **startTrainingSession**: Start a new training session
- **completeTrainingSession**: Complete a training session
- **claimRewards**: Claim accumulated rewards
- **processTransaction**: Process a transaction and add it to the current block
- **mineBlock**: Mine the current block
- **getContributorRewards**: Get contributor's total rewards
- **getContributorProofs**: Get all contributions for a contributor
- **getSessionFeedback**: Get all feedback for a session
- **getTrainingSession**: Get training session details
- **getBlock**: Get block details
- **getTransaction**: Get transaction details
- **getChainStats**: Get chain statistics

## Architecture

The ONI blockchain consists of the following components:

1. **Core Blockchain Logic** (`oni_proof_of_compute.py`): Defines the blockchain data structures, consensus mechanism, and transaction processing.

2. **API Layer** (`oni_blockchain_api.py`): Exposes the blockchain functionality via a RESTful API.

3. **Client Library** (`oni_blockchain_client.py`): Provides a Python client for interacting with the blockchain API.

4. **Integration Layer** (`oni_blockchain_integration.py`): Connects the ONI system with the blockchain, handling local state and synchronization.

5. **Smart Contract** (`ONIToken.sol`): Ethereum-compatible smart contract for the ONI token and on-chain contribution tracking.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Pantheum License - see the [LICENSE](LICENSE) file for details.