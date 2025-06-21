import hashlib
import time
import json
from typing import List, Dict, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ONI-PoC")

# Blockchain configuration
BLOCK_REWARD = 10  # Fixed reward for "mining" a block (training AI model)
DIFFICULTY = 4     # Number of leading zeros required in hash (adjust for PoC)
INITIAL_SUPPLY = 10_000_000_000  # Total ONI tokens (fixed supply)
MAX_TRANSACTIONS_PER_BLOCK = 100  # Maximum transactions per block

class Transaction:
    def __init__(self, sender: str, recipient: str, amount: int, model_update: Optional[Dict] = None):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.model_update = model_update
        self.timestamp = time.time()
        self.signature = None
        
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary for hashing."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "model_update": self.model_update,
            "timestamp": self.timestamp,
            "signature": self.signature
        }
        
    def compute_hash(self) -> str:
        """Create a SHA-256 hash of the transaction."""
        transaction_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(transaction_string.encode()).hexdigest()
    
    def sign_transaction(self, private_key: str) -> None:
        """Sign the transaction with a private key."""
        # In a real implementation, this would use proper cryptographic signing
        # For this example, we'll just create a simple signature
        transaction_hash = self.compute_hash()
        self.signature = hashlib.sha256((transaction_hash + private_key).encode()).hexdigest()
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify the transaction signature."""
        # In a real implementation, this would use proper cryptographic verification
        # For this example, we'll just verify our simple signature
        if not self.signature:
            return False
        
        transaction_hash = self.compute_hash()
        expected_signature = hashlib.sha256((transaction_hash + public_key).encode()).hexdigest()
        return self.signature == expected_signature


class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, transactions: List[Transaction], nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.merkle_root = self.calculate_merkle_root()
        self.hash = None  # Will be set when mined
        
    def calculate_merkle_root(self) -> str:
        """Calculate the Merkle root of all transactions."""
        if not self.transactions:
            return hashlib.sha256("empty".encode()).hexdigest()
        
        # Get transaction hashes
        tx_hashes = [tx.compute_hash() for tx in self.transactions]
        
        # If odd number of transactions, duplicate the last one
        if len(tx_hashes) % 2 == 1:
            tx_hashes.append(tx_hashes[-1])
            
        # Build the Merkle tree
        while len(tx_hashes) > 1:
            next_level = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = next_level
            
            # If odd number of hashes, duplicate the last one
            if len(tx_hashes) % 2 == 1 and len(tx_hashes) > 1:
                tx_hashes.append(tx_hashes[-1])
                
        return tx_hashes[0]

    def to_dict(self) -> Dict:
        """Convert block to dictionary for hashing."""
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce
        }

    def compute_hash(self) -> str:
        """Create a SHA-256 hash of the block's data."""
        block_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class ProofOfCompute:
    """Proof of Compute consensus mechanism for AI training."""
    
    def __init__(self, difficulty: int = DIFFICULTY):
        self.difficulty = difficulty
        
    def validate_proof(self, block: Block, block_hash: str) -> bool:
        """Validate the proof of work/compute."""
        # Check if hash starts with required number of zeros
        return block_hash.startswith('0' * self.difficulty) and block_hash == block.compute_hash()
    
    def find_proof(self, block: Block) -> str:
        """Find a valid proof of compute by adjusting the nonce."""
        block.nonce = 0
        computed_hash = block.compute_hash()
        
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
            
        return computed_hash
    
    def validate_model_update(self, model_update: Dict) -> Tuple[bool, float]:
        """
        Validate an AI model update using zero-knowledge proofs.
        
        Returns:
            Tuple[bool, float]: (is_valid, quality_score)
        """
        # In a real implementation, this would use zk-SNARKs or similar to validate
        # the model update without revealing the actual model weights
        
        # For this example, we'll use a simple validation
        if not model_update:
            return False, 0.0
            
        # Check if model update has required fields
        required_fields = ["model_id", "version", "update_type", "metrics"]
        if not all(field in model_update for field in required_fields):
            return False, 0.0
            
        # Validate metrics
        metrics = model_update.get("metrics", {})
        if not metrics:
            return False, 0.0
            
        # Calculate quality score based on metrics
        quality_score = self._calculate_quality_score(metrics)
        
        return True, quality_score
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate a quality score from model metrics."""
        # This would be a more sophisticated calculation in practice
        score = 0.0
        
        # Example metrics that might be included
        if "accuracy" in metrics:
            score += metrics["accuracy"] * 0.4
            
        if "loss" in metrics:
            # Lower loss is better
            loss_score = max(0, 1 - metrics["loss"])
            score += loss_score * 0.3
            
        if "convergence_rate" in metrics:
            score += metrics["convergence_rate"] * 0.2
            
        if "ethical_score" in metrics:
            score += metrics["ethical_score"] * 0.1
            
        return min(1.0, max(0.0, score))


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.total_supply = INITIAL_SUPPLY
        self.reserve = self.total_supply  # Tokens reserved for redistribution
        self.nodes = set()  # Set of participating nodes
        self.proof_of_compute = ProofOfCompute()
        self.create_genesis_block()
        self.accounts: Dict[str, int] = {"ONI Reserve": self.reserve}
        
    def create_genesis_block(self) -> None:
        """Create the first block in the chain."""
        genesis_block = Block(0, "0", time.time(), [])
        genesis_hash = self.proof_of_compute.find_proof(genesis_block)
        genesis_block.hash = genesis_hash
        self.chain.append(genesis_block)
        logger.info(f"Genesis block created with hash: {genesis_hash}")

    @property
    def last_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]

    def register_node(self, node_address: str) -> None:
        """Add a new node to the list of nodes."""
        self.nodes.add(node_address)
        logger.info(f"Registered new node: {node_address}")

    def add_transaction(self, sender: str, recipient: str, amount: int, 
                       model_update: Optional[Dict] = None, private_key: Optional[str] = None) -> bool:
        """
        Add a transaction to the pending pool.
        
        Args:
            sender: Address of the sender
            recipient: Address of the recipient
            amount: Amount to transfer
            model_update: Optional AI model update data
            private_key: Optional private key for signing
            
        Returns:
            bool: True if transaction was added successfully
        """
        # Validate sender has sufficient balance
        if sender != "ONI Reserve" and self.accounts.get(sender, 0) < amount:
            logger.warning(f"Insufficient balance for transaction: {sender} -> {recipient}, {amount} ONI")
            return False
        
        # Create transaction
        transaction = Transaction(sender, recipient, amount, model_update)
        
        # Sign transaction if private key provided
        if private_key:
            transaction.sign_transaction(private_key)
        
        # Validate model update if present
        if model_update:
            is_valid, quality_score = self.proof_of_compute.validate_model_update(model_update)
            if not is_valid:
                logger.warning(f"Invalid model update in transaction: {sender} -> {recipient}")
                return False
            
            # Adjust transaction amount based on quality score for model updates
            if sender == "ONI Reserve":
                # This is a reward transaction, adjust based on quality
                transaction.amount = int(amount * quality_score)
                logger.info(f"Adjusted reward to {transaction.amount} ONI based on quality score {quality_score}")
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        logger.info(f"Added transaction: {sender} -> {recipient}, {amount} ONI")
        
        return True

    def mine_block(self, miner_address: str) -> Optional[Block]:
        """
        Create a new block by mining (performing proof of compute).
        
        Args:
            miner_address: Address to receive the mining reward
            
        Returns:
            Block: The newly mined block, or None if mining failed
        """
        if not self.pending_transactions:
            logger.info("No transactions to mine")
            return None
            
        # Limit transactions per block
        transactions_to_include = self.pending_transactions[:MAX_TRANSACTIONS_PER_BLOCK]
        
        # Add mining reward transaction
        reward_tx = Transaction("ONI Reserve", miner_address, BLOCK_REWARD)
        transactions_to_include.append(reward_tx)
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            previous_hash=self.last_block.hash,
            timestamp=time.time(),
            transactions=transactions_to_include,
        )
        
        # Find proof of compute
        logger.info(f"Mining block {new_block.index}...")
        start_time = time.time()
        proof = self.proof_of_compute.find_proof(new_block)
        mining_time = time.time() - start_time
        
        # Add block to chain
        new_block.hash = proof
        self.chain.append(new_block)
        
        # Update account balances
        for tx in transactions_to_include:
            # Deduct from sender
            if tx.sender in self.accounts:
                self.accounts[tx.sender] = max(0, self.accounts[tx.sender] - tx.amount)
            
            # Add to recipient
            if tx.recipient in self.accounts:
                self.accounts[tx.recipient] += tx.amount
            else:
                self.accounts[tx.recipient] = tx.amount
                
            # Update reserve
            if tx.sender == "ONI Reserve":
                self.reserve -= tx.amount
        
        # Remove processed transactions from pending
        self.pending_transactions = self.pending_transactions[MAX_TRANSACTIONS_PER_BLOCK:]
        
        logger.info(f"Block {new_block.index} mined in {mining_time:.2f} seconds with hash: {proof}")
        logger.info(f"Reward of {BLOCK_REWARD} ONI sent to {miner_address}")
        
        return new_block

    def is_valid_chain(self, chain: List[Block]) -> bool:
        """
        Check if a blockchain is valid.
        
        Args:
            chain: The blockchain to validate
            
        Returns:
            bool: True if the chain is valid
        """
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i-1]
            
            # Check hash integrity
            if current.previous_hash != previous.hash:
                logger.warning(f"Invalid previous hash in block {current.index}")
                return False
                
            # Check proof of compute
            if not self.proof_of_compute.validate_proof(current, current.hash):
                logger.warning(f"Invalid proof of compute in block {current.index}")
                return False
                
            # Validate merkle root
            if current.merkle_root != current.calculate_merkle_root():
                logger.warning(f"Invalid merkle root in block {current.index}")
                return False
                
        return True

    def resolve_conflicts(self, chains: List[List[Block]]) -> bool:
        """
        Consensus algorithm: resolve conflicts by replacing our chain with the longest valid chain.
        
        Args:
            chains: List of chains from other nodes
            
        Returns:
            bool: True if our chain was replaced
        """
        max_length = len(self.chain)
        new_chain = None
        
        # Find the longest valid chain
        for chain in chains:
            if len(chain) > max_length and self.is_valid_chain(chain):
                max_length = len(chain)
                new_chain = chain
                
        # Replace our chain if a longer valid one is found
        if new_chain:
            self.chain = new_chain
            logger.info(f"Chain replaced with longer chain of length {len(new_chain)}")
            return True
            
        logger.info("Current chain is authoritative")
        return False
    
    def get_balance(self, address: str) -> int:
        """Get the balance of an address."""
        return self.accounts.get(address, 0)
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Get a block by its index."""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Get a block by its hash."""
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None
    
    def get_transactions_by_address(self, address: str) -> List[Dict]:
        """Get all transactions involving an address."""
        transactions = []
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address or tx.recipient == address:
                    transactions.append({
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "timestamp": tx.timestamp,
                        "sender": tx.sender,
                        "recipient": tx.recipient,
                        "amount": tx.amount,
                        "has_model_update": tx.model_update is not None
                    })
                    
        return transactions
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the blockchain."""
        return {
            "length": len(self.chain),
            "total_supply": self.total_supply,
            "circulating_supply": self.total_supply - self.reserve,
            "reserve": self.reserve,
            "difficulty": self.proof_of_compute.difficulty,
            "pending_transactions": len(self.pending_transactions),
            "total_accounts": len(self.accounts),
            "last_block_time": self.last_block.timestamp if self.chain else 0
        }


class ONINode:
    """Node in the ONI blockchain network."""
    
    def __init__(self, node_id: str, private_key: str):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = hashlib.sha256(private_key.encode()).hexdigest()  # Simplified
        self.blockchain = Blockchain()
        self.peers = set()
        self.model_cache = {}  # Cache for model updates
        
    def register_peer(self, peer_address: str) -> None:
        """Register a new peer node."""
        self.peers.add(peer_address)
        self.blockchain.register_node(peer_address)
        
    def broadcast_transaction(self, transaction: Transaction) -> None:
        """Broadcast a transaction to all peers."""
        # In a real implementation, this would use networking to send to peers
        logger.info(f"Broadcasting transaction to {len(self.peers)} peers")
        
    def broadcast_block(self, block: Block) -> None:
        """Broadcast a new block to all peers."""
        # In a real implementation, this would use networking to send to peers
        logger.info(f"Broadcasting block {block.index} to {len(self.peers)} peers")
        
    def submit_model_update(self, model_id: str, version: str, update_data: Dict, metrics: Dict) -> bool:
        """
        Submit an AI model update to the blockchain.
        
        Args:
            model_id: Identifier for the model
            version: Version of the model update
            update_data: Model update data (weights, gradients, etc.)
            metrics: Performance metrics for the update
            
        Returns:
            bool: True if the update was accepted
        """
        # Create model update package
        model_update = {
            "model_id": model_id,
            "version": version,
            "update_type": "weights",
            "timestamp": time.time(),
            "metrics": metrics,
            "data_hash": hashlib.sha256(json.dumps(update_data).encode()).hexdigest()
        }
        
        # Cache the actual update data (not stored on blockchain)
        update_id = model_update["data_hash"]
        self.model_cache[update_id] = update_data
        
        # Create and sign transaction
        success = self.blockchain.add_transaction(
            sender="ONI Reserve",
            recipient=self.node_id,
            amount=BLOCK_REWARD,  # Base reward, will be adjusted based on quality
            model_update=model_update,
            private_key=self.private_key
        )
        
        if success:
            logger.info(f"Model update submitted for {model_id} v{version}")
            # In a real system, we would broadcast this transaction
            return True
        else:
            logger.warning(f"Failed to submit model update for {model_id} v{version}")
            return False
    
    def mine_pending_transactions(self) -> Optional[Block]:
        """Mine pending transactions into a new block."""
        new_block = self.blockchain.mine_block(self.node_id)
        
        if new_block:
            # Broadcast the new block to peers
            self.broadcast_block(new_block)
            
        return new_block
    
    def validate_and_add_block(self, block: Block) -> bool:
        """
        Validate a block received from a peer and add it to the chain if valid.
        
        Args:
            block: The block to validate and add
            
        Returns:
            bool: True if the block was added successfully
        """
        # Check if block index is valid
        if block.index != len(self.blockchain.chain):
            logger.warning(f"Invalid block index: {block.index}, expected {len(self.blockchain.chain)}")
            return False
            
        # Check if previous hash matches
        if block.previous_hash != self.blockchain.last_block.hash:
            logger.warning(f"Invalid previous hash in block {block.index}")
            return False
            
        # Validate proof of compute
        if not self.blockchain.proof_of_compute.validate_proof(block, block.hash):
            logger.warning(f"Invalid proof of compute in block {block.index}")
            return False
            
        # Validate transactions
        for tx in block.transactions:
            # Skip mining reward transaction
            if tx.sender == "ONI Reserve" and tx.amount == BLOCK_REWARD:
                continue
                
            # Validate transaction signature
            if tx.signature and not tx.verify_signature(tx.sender):
                logger.warning(f"Invalid transaction signature in block {block.index}")
                return False
                
            # Validate model update if present
            if tx.model_update:
                is_valid, _ = self.blockchain.proof_of_compute.validate_model_update(tx.model_update)
                if not is_valid:
                    logger.warning(f"Invalid model update in block {block.index}")
                    return False
        
        # Add block to chain
        self.blockchain.chain.append(block)
        logger.info(f"Added block {block.index} to chain")
        
        return True
    
    def sync_with_network(self) -> bool:
        """
        Synchronize with the network by requesting chains from peers.
        
        Returns:
            bool: True if synchronization was successful
        """
        # In a real implementation, this would request chains from peers
        # For this example, we'll assume no peers
        logger.info("Synchronizing with network...")
        return True
    
    def get_model_update(self, update_id: str) -> Optional[Dict]:
        """
        Get a model update from the cache.
        
        Args:
            update_id: The ID of the update to retrieve
            
        Returns:
            Dict: The model update data, or None if not found
        """
        return self.model_cache.get(update_id)


# Example usage
if __name__ == "__main__":
    # Create a node
    node = ONINode("node1", "private_key_1")
    
    # Submit a model update
    model_metrics = {
        "accuracy": 0.85,
        "loss": 0.15,
        "convergence_rate": 0.75,
        "ethical_score": 0.9
    }
    
    model_data = {
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "gradients": [0.01, 0.02, 0.03, 0.04, 0.05]
    }
    
    node.submit_model_update(
        model_id="oni-nlp-v1",
        version="1.0.0",
        update_data=model_data,
        metrics=model_metrics
    )
    
    # Mine a block
    new_block = node.mine_pending_transactions()
    
    if new_block:
        print(f"Mined block {new_block.index} with hash: {new_block.hash}")
        print(f"Block contains {len(new_block.transactions)} transactions")
        print(f"Node balance: {node.blockchain.get_balance(node.node_id)} ONI")
    
    # Get blockchain stats
    stats = node.blockchain.get_chain_stats()
    print(f"Blockchain stats: {json.dumps(stats, indent=2)}")