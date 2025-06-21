import hashlib
import time
import json
from typing import List, Dict, Optional, Tuple, Any
import logging
import threading
import queue
import uuid
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Blockchain configuration
BLOCK_REWARD = 10  # Fixed reward for "mining" a block (training AI model)
DIFFICULTY = 2     # Number of leading zeros required in hash (lower for faster blocks)
INITIAL_SUPPLY = 10_000_000_000  # Total ONI tokens (fixed supply)
MAX_TRANSACTIONS_PER_BLOCK = 1000  # Increased for higher throughput
TARGET_BLOCK_TIME = 3  # Target time in seconds for block creation (very fast)
DIFFICULTY_ADJUSTMENT_INTERVAL = 100  # Adjust difficulty every 100 blocks
GAS_LIMIT = 10_000_000  # Gas limit per block
BASE_GAS_PRICE = 0.00001  # Very low base gas price

class Transaction:
    def __init__(self, sender: str, recipient: str, amount: int, gas_price: float = BASE_GAS_PRICE, 
                 gas_limit: int = 100000, model_update: Optional[Dict] = None, data: Optional[str] = None):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.gas_price = gas_price
        self.gas_limit = gas_limit
        self.model_update = model_update
        self.data = data
        self.timestamp = time.time()
        self.nonce = 0  # For transaction ordering/replay protection
        self.signature = None
        self.tx_hash = None
        self.compute_hash()  # Initialize hash
        
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary for hashing."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "gas_price": self.gas_price,
            "gas_limit": self.gas_limit,
            "model_update": self.model_update,
            "data": self.data,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }
        
    def compute_hash(self) -> str:
        """Create a SHA-256 hash of the transaction."""
        transaction_string = json.dumps(self.to_dict(), sort_keys=True)
        self.tx_hash = hashlib.sha256(transaction_string.encode()).hexdigest()
        return self.tx_hash
    
    def sign_transaction(self, private_key: str) -> None:
        """Sign the transaction with a private key."""
        # In a real implementation, this would use proper cryptographic signing
        # For this example, we'll just create a simple signature
        transaction_hash = self.compute_hash()
        self.nonce = int(time.time() * 1000)  # Use timestamp as nonce for simplicity
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
    
    def calculate_gas(self) -> int:
        """Calculate gas used by this transaction."""
        base_gas = 21000  # Base gas for a standard transaction
        
        # Add gas for data
        if self.data:
            # 4 gas for each zero byte, 68 for non-zero
            for byte in self.data.encode():
                if byte == 0:
                    base_gas += 4
                else:
                    base_gas += 68
        
        # Add gas for model update (if present)
        if self.model_update:
            # Simplified: 1000 gas + 100 per key in the model update
            base_gas += 1000 + (len(self.model_update) * 100)
        
        return min(base_gas, self.gas_limit)
    
    def calculate_fee(self) -> float:
        """Calculate the transaction fee."""
        return self.calculate_gas() * self.gas_price


class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, transactions: List[Transaction], 
                 difficulty: int = DIFFICULTY, nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.difficulty = difficulty
        self.nonce = nonce
        self.merkle_root = self.calculate_merkle_root()
        self.hash = None  # Will be set when mined
        self.gas_used = sum(tx.calculate_gas() for tx in transactions)
        self.size = len(json.dumps(self.to_dict()).encode())
        
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
            "difficulty": self.difficulty,
            "nonce": self.nonce,
            "gas_used": self.gas_used,
            "size": self.size
        }

    def compute_hash(self) -> str:
        """Create a SHA-256 hash of the block's data."""
        block_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get a transaction by its hash."""
        for tx in self.transactions:
            if tx.tx_hash == tx_hash:
                return tx
        return None


class ProofOfCompute:
    """Proof of Compute consensus mechanism for AI training."""
    
    def __init__(self, difficulty: int = DIFFICULTY, target_block_time: int = TARGET_BLOCK_TIME):
        self.difficulty = difficulty
        self.target_block_time = target_block_time
        self.last_adjustment_time = time.time()
        self.blocks_since_adjustment = 0
        
    def validate_proof(self, block: Block, block_hash: str) -> bool:
        """Validate the proof of work/compute."""
        # Check if hash starts with required number of zeros
        return block_hash.startswith('0' * block.difficulty) and block_hash == block.compute_hash()
    
    def find_proof(self, block: Block) -> str:
        """Find a valid proof of compute by adjusting the nonce."""
        block.nonce = 0
        computed_hash = block.compute_hash()
        
        start_time = time.time()
        
        while not computed_hash.startswith('0' * block.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
            
            # Optional: Add a timeout to prevent infinite loops
            if time.time() - start_time > 60:  # 1 minute timeout
                logger.warning("Mining timeout reached, adjusting difficulty")
                if block.difficulty > 1:
                    block.difficulty -= 1
                start_time = time.time()
            
        mining_time = time.time() - start_time
        logger.info(f"Block mined in {mining_time:.2f} seconds with difficulty {block.difficulty}")
        
        return computed_hash
    
    def adjust_difficulty(self, blocks: List[Block]) -> int:
        """Adjust difficulty based on recent block times to maintain target block time."""
        if len(blocks) < DIFFICULTY_ADJUSTMENT_INTERVAL:
            return self.difficulty
        
        # Calculate average block time for the last interval
        recent_blocks = blocks[-DIFFICULTY_ADJUSTMENT_INTERVAL:]
        if len(recent_blocks) < 2:
            return self.difficulty
            
        first_block_time = recent_blocks[0].timestamp
        last_block_time = recent_blocks[-1].timestamp
        avg_block_time = (last_block_time - first_block_time) / (len(recent_blocks) - 1)
        
        # Adjust difficulty to target block time
        if avg_block_time < self.target_block_time * 0.5:
            # Blocks are too fast, increase difficulty
            new_difficulty = self.difficulty + 1
        elif avg_block_time > self.target_block_time * 2:
            # Blocks are too slow, decrease difficulty
            new_difficulty = max(1, self.difficulty - 1)
        else:
            # Block time is acceptable
            new_difficulty = self.difficulty
            
        logger.info(f"Adjusted difficulty from {self.difficulty} to {new_difficulty} (avg block time: {avg_block_time:.2f}s)")
        return new_difficulty
    
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


class MemPool:
    """Memory pool for pending transactions with advanced features."""
    
    def __init__(self, max_size: int = 10000):
        self.transactions = OrderedDict()  # tx_hash -> Transaction
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the mempool."""
        with self.lock:
            # Check if mempool is full
            if len(self.transactions) >= self.max_size:
                # Remove lowest fee transaction if new one has higher fee
                if self._should_replace_lowest_fee(transaction):
                    self._remove_lowest_fee_transaction()
                else:
                    return False
            
            # Add transaction to mempool
            self.transactions[transaction.tx_hash] = transaction
            return True
    
    def get_transactions(self, max_count: int = MAX_TRANSACTIONS_PER_BLOCK, 
                         max_gas: int = GAS_LIMIT) -> List[Transaction]:
        """Get transactions for a new block, prioritizing by fee."""
        with self.lock:
            # Sort transactions by fee (highest first)
            sorted_txs = sorted(
                self.transactions.values(),
                key=lambda tx: tx.calculate_fee(),
                reverse=True
            )
            
            selected_txs = []
            total_gas = 0
            
            for tx in sorted_txs:
                tx_gas = tx.calculate_gas()
                if total_gas + tx_gas <= max_gas and len(selected_txs) < max_count:
                    selected_txs.append(tx)
                    total_gas += tx_gas
                    
                    # Remove from mempool
                    del self.transactions[tx.tx_hash]
                
                if len(selected_txs) >= max_count or total_gas >= max_gas:
                    break
                    
            return selected_txs
    
    def remove_transaction(self, tx_hash: str) -> bool:
        """Remove a transaction from the mempool."""
        with self.lock:
            if tx_hash in self.transactions:
                del self.transactions[tx_hash]
                return True
            return False
    
    def contains_transaction(self, tx_hash: str) -> bool:
        """Check if a transaction is in the mempool."""
        with self.lock:
            return tx_hash in self.transactions
    
    def clear(self) -> None:
        """Clear all transactions from the mempool."""
        with self.lock:
            self.transactions.clear()
    
    def size(self) -> int:
        """Get the number of transactions in the mempool."""
        with self.lock:
            return len(self.transactions)
    
    def _should_replace_lowest_fee(self, transaction: Transaction) -> bool:
        """Check if the new transaction has a higher fee than the lowest in the pool."""
        with self.lock:
            if not self.transactions:
                return True
                
            lowest_fee_tx = min(self.transactions.values(), key=lambda tx: tx.calculate_fee())
            return transaction.calculate_fee() > lowest_fee_tx.calculate_fee()
    
    def _remove_lowest_fee_transaction(self) -> None:
        """Remove the transaction with the lowest fee."""
        with self.lock:
            if not self.transactions:
                return
                
            lowest_fee_tx = min(self.transactions.values(), key=lambda tx: tx.calculate_fee())
            del self.transactions[lowest_fee_tx.tx_hash]


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.mempool = MemPool()
        self.total_supply = INITIAL_SUPPLY
        self.reserve = self.total_supply  # Tokens reserved for redistribution
        self.nodes = set()  # Set of participating nodes
        self.proof_of_compute = ProofOfCompute()
        self.create_genesis_block()
        self.accounts: Dict[str, int] = {"ONI Reserve": self.reserve}
        self.lock = threading.Lock()
        self.tx_index = {}  # tx_hash -> (block_index, tx_index)
        
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
                       gas_price: float = BASE_GAS_PRICE, gas_limit: int = 100000,
                       model_update: Optional[Dict] = None, data: Optional[str] = None, 
                       private_key: Optional[str] = None) -> bool:
        """
        Add a transaction to the mempool.
        
        Args:
            sender: Address of the sender
            recipient: Address of the recipient
            amount: Amount to transfer
            gas_price: Price per unit of gas
            gas_limit: Maximum gas allowed for transaction
            model_update: Optional AI model update data
            data: Optional transaction data
            private_key: Optional private key for signing
            
        Returns:
            bool: True if transaction was added successfully
        """
        # Validate sender has sufficient balance
        if sender != "ONI Reserve" and self.accounts.get(sender, 0) < amount:
            logger.warning(f"Insufficient balance for transaction: {sender} -> {recipient}, {amount} ONI")
            return False
        
        # Create transaction
        transaction = Transaction(sender, recipient, amount, gas_price, gas_limit, model_update, data)
        
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
        
        # Add to mempool
        return self.mempool.add_transaction(transaction)

    def mine_block(self, miner_address: str) -> Optional[Block]:
        """
        Create a new block by mining (performing proof of compute).
        
        Args:
            miner_address: Address to receive the mining reward
            
        Returns:
            Block: The newly mined block, or None if mining failed
        """
        with self.lock:
            # Get transactions from mempool
            transactions = self.mempool.get_transactions()
            
            if not transactions:
                logger.info("No transactions to mine")
                return None
                
            # Add mining reward transaction
            reward_tx = Transaction("ONI Reserve", miner_address, BLOCK_REWARD)
            transactions.append(reward_tx)
            
            # Adjust difficulty if needed
            if len(self.chain) % DIFFICULTY_ADJUSTMENT_INTERVAL == 0:
                new_difficulty = self.proof_of_compute.adjust_difficulty(self.chain)
                current_difficulty = self.proof_of_compute.difficulty
                self.proof_of_compute.difficulty = new_difficulty
                logger.info(f"Adjusted difficulty from {current_difficulty} to {new_difficulty}")
            
            # Create new block
            new_block = Block(
                index=len(self.chain),
                previous_hash=self.last_block.hash,
                timestamp=time.time(),
                transactions=transactions,
                difficulty=self.proof_of_compute.difficulty
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
            for tx in transactions:
                # Deduct from sender
                if tx.sender in self.accounts:
                    fee = tx.calculate_fee()
                    self.accounts[tx.sender] = max(0, self.accounts[tx.sender] - tx.amount - fee)
                
                # Add to recipient
                if tx.recipient in self.accounts:
                    self.accounts[tx.recipient] += tx.amount
                else:
                    self.accounts[tx.recipient] = tx.amount
                    
                # Update reserve
                if tx.sender == "ONI Reserve":
                    self.reserve -= tx.amount
                
                # Add to transaction index
                self.tx_index[tx.tx_hash] = (new_block.index, transactions.index(tx))
            
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
        with self.lock:
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
                
                # Rebuild transaction index
                self.tx_index = {}
                for block_idx, block in enumerate(self.chain):
                    for tx_idx, tx in enumerate(block.transactions):
                        self.tx_index[tx.tx_hash] = (block_idx, tx_idx)
                
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
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get a transaction by its hash."""
        if tx_hash in self.tx_index:
            block_idx, tx_idx = self.tx_index[tx_hash]
            block = self.get_block_by_index(block_idx)
            if block and 0 <= tx_idx < len(block.transactions):
                return block.transactions[tx_idx]
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
                        "tx_hash": tx.tx_hash,
                        "timestamp": tx.timestamp,
                        "sender": tx.sender,
                        "recipient": tx.recipient,
                        "amount": tx.amount,
                        "fee": tx.calculate_fee(),
                        "has_model_update": tx.model_update is not None
                    })
                    
        return transactions
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the blockchain."""
        total_txs = sum(len(block.transactions) for block in self.chain)
        avg_block_time = 0
        if len(self.chain) > 1:
            total_time = self.chain[-1].timestamp - self.chain[0].timestamp
            avg_block_time = total_time / (len(self.chain) - 1) if len(self.chain) > 1 else 0
            
        return {
            "length": len(self.chain),
            "total_supply": self.total_supply,
            "circulating_supply": self.total_supply - self.reserve,
            "reserve": self.reserve,
            "difficulty": self.proof_of_compute.difficulty,
            "mempool_size": self.mempool.size(),
            "total_transactions": total_txs,
            "total_accounts": len(self.accounts),
            "avg_block_time": avg_block_time,
            "last_block_time": self.last_block.timestamp if self.chain else 0,
            "target_block_time": TARGET_BLOCK_TIME,
            "gas_price": BASE_GAS_PRICE
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
        self.sync_lock = threading.Lock()
        self.mining_thread = None
        self.is_mining = False
        self.tx_queue = queue.Queue()
        self.tx_processor_thread = None
        self.is_processing_txs = False
        
    def start(self):
        """Start the node's background threads."""
        self.start_tx_processor()
        logger.info(f"ONI Node {self.node_id} started")
        
    def stop(self):
        """Stop the node's background threads."""
        self.is_mining = False
        self.is_processing_txs = False
        if self.mining_thread and self.mining_thread.is_alive():
            self.mining_thread.join(timeout=2)
        if self.tx_processor_thread and self.tx_processor_thread.is_alive():
            self.tx_processor_thread.join(timeout=2)
        logger.info(f"ONI Node {self.node_id} stopped")
        
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
        
    def submit_transaction(self, sender: str, recipient: str, amount: int, 
                          gas_price: float = BASE_GAS_PRICE, gas_limit: int = 100000,
                          data: Optional[str] = None) -> str:
        """Submit a regular transaction to the network."""
        transaction = Transaction(sender, recipient, amount, gas_price, gas_limit, None, data)
        transaction.sign_transaction(self.private_key)
        
        # Add to transaction queue for processing
        self.tx_queue.put(transaction)
        
        return transaction.tx_hash
        
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
        transaction = Transaction(
            sender="ONI Reserve",
            recipient=self.node_id,
            amount=BLOCK_REWARD,  # Base reward, will be adjusted based on quality
            gas_price=BASE_GAS_PRICE,
            gas_limit=200000,  # Higher gas limit for model updates
            model_update=model_update
        )
        transaction.sign_transaction(self.private_key)
        
        # Add to transaction queue for processing
        self.tx_queue.put(transaction)
        
        logger.info(f"Model update submitted for {model_id} v{version}")
        return True
    
    def start_tx_processor(self):
        """Start the transaction processor thread."""
        if self.tx_processor_thread is None or not self.tx_processor_thread.is_alive():
            self.is_processing_txs = True
            self.tx_processor_thread = threading.Thread(target=self._process_transactions)
            self.tx_processor_thread.daemon = True
            self.tx_processor_thread.start()
            logger.info("Transaction processor started")
    
    def _process_transactions(self):
        """Process transactions from the queue."""
        while self.is_processing_txs:
            try:
                # Get transaction from queue with timeout
                transaction = self.tx_queue.get(timeout=1)
                
                # Add to blockchain
                success = self.blockchain.add_transaction(
                    sender=transaction.sender,
                    recipient=transaction.recipient,
                    amount=transaction.amount,
                    gas_price=transaction.gas_price,
                    gas_limit=transaction.gas_limit,
                    model_update=transaction.model_update,
                    data=transaction.data,
                    private_key=self.private_key if transaction.sender == self.node_id else None
                )
                
                if success:
                    # Broadcast transaction to network
                    self.broadcast_transaction(transaction)
                    logger.info(f"Transaction processed: {transaction.tx_hash}")
                else:
                    logger.warning(f"Failed to process transaction: {transaction.tx_hash}")
                
                # Mark task as done
                self.tx_queue.task_done()
                
                # Start mining if not already mining and mempool has enough transactions
                if not self.is_mining and self.blockchain.mempool.size() >= 10:
                    self.start_mining()
                    
            except queue.Empty:
                # No transactions in queue
                pass
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
    
    def start_mining(self):
        """Start the mining thread."""
        if self.mining_thread is None or not self.mining_thread.is_alive():
            self.is_mining = True
            self.mining_thread = threading.Thread(target=self._mine_continuously)
            self.mining_thread.daemon = True
            self.mining_thread.start()
            logger.info("Mining thread started")
    
    def stop_mining(self):
        """Stop the mining thread."""
        self.is_mining = False
        if self.mining_thread and self.mining_thread.is_alive():
            self.mining_thread.join(timeout=2)
            logger.info("Mining thread stopped")
    
    def _mine_continuously(self):
        """Continuously mine blocks."""
        while self.is_mining:
            # Check if there are transactions to mine
            if self.blockchain.mempool.size() > 0:
                # Mine a block
                new_block = self.mine_pending_transactions()
                
                if new_block:
                    # Broadcast the new block
                    self.broadcast_block(new_block)
            else:
                # No transactions to mine, sleep for a bit
                time.sleep(1)
    
    def mine_pending_transactions(self) -> Optional[Block]:
        """Mine pending transactions into a new block."""
        with self.sync_lock:
            new_block = self.blockchain.mine_block(self.node_id)
            return new_block
    
    def validate_and_add_block(self, block: Block) -> bool:
        """
        Validate a block received from a peer and add it to the chain if valid.
        
        Args:
            block: The block to validate and add
            
        Returns:
            bool: True if the block was added successfully
        """
        with self.sync_lock:
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
            
            # Update transaction index
            for tx_idx, tx in enumerate(block.transactions):
                self.blockchain.tx_index[tx.tx_hash] = (block.index, tx_idx)
                
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
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get status information about the node."""
        return {
            "node_id": self.node_id,
            "peers": len(self.peers),
            "mempool_size": self.blockchain.mempool.size(),
            "is_mining": self.is_mining,
            "chain_stats": self.blockchain.get_chain_stats(),
            "model_cache_size": len(self.model_cache),
            "account_balance": self.blockchain.get_balance(self.node_id)
        }


# Example usage
if __name__ == "__main__":
    # Create a node
    node = ONINode("node1", "private_key_1")
    
    # Start the node
    node.start()
    
    try:
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
        
        # Submit a regular transaction
        node.submit_transaction(
            sender=node.node_id,
            recipient="user1",
            amount=5
        )
        
        # Wait for mining to complete
        time.sleep(10)
        
        # Get node status
        status = node.get_node_status()
        print(f"Node status: {json.dumps(status, indent=2)}")
        
    finally:
        # Stop the node
        node.stop()