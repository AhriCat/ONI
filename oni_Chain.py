import hashlib
import time
import json
from typing import List, Dict, Optional

# Blockchain configuration
BLOCK_REWARD = 10  # Fixed reward for "mining" a block (training AI model)
DIFFICULTY = 4     # Number of leading zeros required in hash (adjust for PoC)
INITIAL_SUPPLY = 10_000_000_000  # Total ONI tokens (fixed supply)

class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, transactions: List[Dict], nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce

    def compute_hash(self) -> str:
        """Create a SHA-256 hash of the block's data."""
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.total_supply = INITIAL_SUPPLY
        self.reserve = self.total_supply  # Tokens reserved for redistribution
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block in the chain."""
        genesis_block = Block(0, "0", time.time(), [])
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    @property
    def last_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]

    def add_block(self, block: Block, proof: str):
        """Add a validated block to the chain."""
        if block.previous_hash != self.last_block.compute_hash():
            raise ValueError("Invalid previous hash")

        if not self.is_valid_proof(block, proof):
            raise ValueError("Invalid proof of work")

        block.hash = proof
        self.chain.append(block)

    def is_valid_proof(self, block: Block, proof: str) -> bool:
        """Validate the proof of work for the block."""
        return proof.startswith("0" * DIFFICULTY) and proof == block.compute_hash()

    def proof_of_compute(self, block: Block) -> str:
        """Perform Proof-of-Compute by solving a hash puzzle."""
        block.nonce = 0
        while True:
            proof = block.compute_hash()
            if proof.startswith("0" * DIFFICULTY):
                return proof
            block.nonce += 1

    def add_transaction(self, sender: str, recipient: str, amount: int, model_update: Optional[Dict] = None):
        """Add a transaction to the pending pool."""
        if amount > self.reserve:
            raise ValueError("Insufficient reserve for transaction")
        
        # Placeholder: Model update logic
        if model_update:
            self.validate_model_update(model_update)
        
        self.pending_transactions.append({
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "model_update": model_update,
        })

    def validate_model_update(self, model_update: Dict) -> bool:
        """Placeholder: Validate AI model update."""
        # Implement zk-SNARK or homomorphic encryption for secure AI training validation
        print(f"Validating model update: {model_update}")
        return True

    def mine_block(self, miner_address: str):
        """Create a new block by mining."""
        # Reward the miner
        self.add_transaction(sender="ONI Reserve", recipient=miner_address, amount=BLOCK_REWARD)

        new_block = Block(
            index=len(self.chain),
            previous_hash=self.last_block.compute_hash(),
            timestamp=time.time(),
            transactions=self.pending_transactions,
        )
        proof = self.proof_of_compute(new_block)
        self.add_block(new_block, proof)

        # Reset pending transactions
        self.pending_transactions = []

        # Adjust reserve
        self.reserve -= BLOCK_REWARD


# Instantiate the blockchain
oni_blockchain = Blockchain()

# Simulate adding transactions
oni_blockchain.add_transaction(sender="User1", recipient="User2", amount=50)
oni_blockchain.add_transaction(sender="User3", recipient="AI Validator", amount=20, model_update={"weights": "encrypted_weights_here"})

# Simulate mining a block
print("Mining block...")
oni_blockchain.mine_block(miner_address="ValidatorNode1")

# Print the blockchain
for block in oni_blockchain.chain:
    print(f"Block {block.index}: {block.__dict__}")
