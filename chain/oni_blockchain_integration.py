import os
import json
import hashlib
import time
import logging
import requests
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
import queue
from pathlib import Path

# Import blockchain client
from chain.oni_blockchain_client import ONIBlockchainClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ONI-Blockchain-Integration")

class ONIBlockchainIntegration:
    """
    Integration layer between ONI system and blockchain network.
    Handles model updates, training contributions, and reward distribution.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the blockchain integration.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize blockchain client
        self.client = ONIBlockchainClient(
            node_url=self.config.get("node_url", "http://localhost:5000"),
            api_key=self.config.get("api_key"),
            private_key=self.config.get("private_key")
        )
        
        # Initialize model cache
        self.model_cache_dir = Path(self.config.get("model_cache_dir", "./model_cache"))
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize contribution tracking
        self.contributions_file = Path(self.config.get("contributions_file", "./contributions.json"))
        self.contributions = self._load_contributions()
        
        # Initialize transaction queue
        self.tx_queue = queue.Queue()
        self.tx_processor_thread = None
        self.is_processing_txs = False
        
        # Start background sync thread if enabled
        if self.config.get("auto_sync", False):
            self.sync_thread = threading.Thread(target=self._background_sync, daemon=True)
            self.sync_thread.start()
        else:
            self.sync_thread = None
            
        # Start transaction processor if enabled
        if self.config.get("auto_process_tx", True):
            self.start_tx_processor()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or environment variables."""
        config = {
            "node_url": os.environ.get("ONI_BLOCKCHAIN_NODE_URL", "http://localhost:5000"),
            "api_key": os.environ.get("ONI_BLOCKCHAIN_API_KEY"),
            "private_key": os.environ.get("ONI_BLOCKCHAIN_PRIVATE_KEY"),
            "model_cache_dir": os.environ.get("ONI_MODEL_CACHE_DIR", "./model_cache"),
            "contributions_file": os.environ.get("ONI_CONTRIBUTIONS_FILE", "./contributions.json"),
            "auto_sync": os.environ.get("ONI_AUTO_SYNC", "false").lower() == "true",
            "sync_interval": int(os.environ.get("ONI_SYNC_INTERVAL", "300")),  # 5 minutes
            "auto_mine": os.environ.get("ONI_AUTO_MINE", "false").lower() == "true",
            "mine_interval": int(os.environ.get("ONI_MINE_INTERVAL", "600")),  # 10 minutes
            "auto_process_tx": os.environ.get("ONI_AUTO_PROCESS_TX", "true").lower() == "true",
            "batch_transactions": os.environ.get("ONI_BATCH_TRANSACTIONS", "true").lower() == "true",
            "batch_size": int(os.environ.get("ONI_BATCH_SIZE", "20")),
            "gas_price": float(os.environ.get("ONI_GAS_PRICE", "0.00001"))
        }
        
        # Override with config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        return config
    
    def _load_contributions(self) -> Dict:
        """Load contribution history from file."""
        if self.contributions_file.exists():
            try:
                with open(self.contributions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load contributions file: {e}")
        
        # Initialize empty contributions structure
        return {
            "training_sessions": {},
            "model_updates": {},
            "feedback": {},
            "rewards": {},
            "last_sync": 0
        }
    
    def _save_contributions(self) -> None:
        """Save contribution history to file."""
        try:
            with open(self.contributions_file, 'w') as f:
                json.dump(self.contributions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save contributions file: {e}")
    
    def _background_sync(self) -> None:
        """Background thread for periodic blockchain synchronization."""
        while True:
            try:
                # Sync with blockchain
                self.sync_with_blockchain()
                
                # Mine pending transactions if enabled
                if self.config.get("auto_mine", False):
                    self.mine_pending_transactions()
                
                # Sleep for sync interval
                time.sleep(self.config.get("sync_interval", 300))
            except Exception as e:
                logger.error(f"Error in background sync: {e}")
                time.sleep(60)  # Sleep for a minute on error
    
    def start_tx_processor(self) -> None:
        """Start the transaction processor thread."""
        if self.tx_processor_thread is None or not self.tx_processor_thread.is_alive():
            self.is_processing_txs = True
            self.tx_processor_thread = threading.Thread(target=self._process_transactions)
            self.tx_processor_thread.daemon = True
            self.tx_processor_thread.start()
            logger.info("Transaction processor started")
    
    def stop_tx_processor(self) -> None:
        """Stop the transaction processor thread."""
        self.is_processing_txs = False
        if self.tx_processor_thread and self.tx_processor_thread.is_alive():
            self.tx_processor_thread.join(timeout=5)
            logger.info("Transaction processor stopped")
    
    def _process_transactions(self) -> None:
        """Process transactions from the queue."""
        while self.is_processing_txs:
            try:
                # Get transaction from queue with timeout
                tx_data = self.tx_queue.get(timeout=1)
                
                # Submit transaction
                if self.config.get("batch_transactions", True):
                    # Use client's batch functionality
                    tx_hash = self.client.submit_transaction(
                        recipient=tx_data["recipient"],
                        amount=tx_data["amount"],
                        gas_price=tx_data.get("gas_price", self.config.get("gas_price", 0.00001)),
                        gas_limit=tx_data.get("gas_limit", 100000),
                        model_update=tx_data.get("model_update"),
                        data=tx_data.get("data"),
                        batch=True
                    )
                else:
                    # Submit immediately
                    tx_hash = self.client.submit_transaction(
                        recipient=tx_data["recipient"],
                        amount=tx_data["amount"],
                        gas_price=tx_data.get("gas_price", self.config.get("gas_price", 0.00001)),
                        gas_limit=tx_data.get("gas_limit", 100000),
                        model_update=tx_data.get("model_update"),
                        data=tx_data.get("data"),
                        batch=False
                    )
                
                # Record transaction
                if tx_hash:
                    logger.info(f"Transaction submitted: {tx_hash}")
                    
                    # Update local record if this is a model update
                    if tx_data.get("model_update"):
                        model_id = tx_data["model_update"]["model_id"]
                        version = tx_data["model_update"]["version"]
                        model_hash = tx_data["model_update"]["data_hash"]
                        
                        self.contributions["model_updates"][model_hash] = {
                            "model_id": model_id,
                            "version": version,
                            "tx_hash": tx_hash,
                            "timestamp": time.time(),
                            "status": "pending"
                        }
                        self._save_contributions()
                
                # Mark task as done
                self.tx_queue.task_done()
                
            except queue.Empty:
                # No transactions in queue
                pass
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
    
    def sync_with_blockchain(self) -> bool:
        """
        Synchronize local state with blockchain.
        
        Returns:
            bool: True if synchronization was successful
        """
        try:
            # Get chain stats
            stats = self.client.get_chain_stats()
            
            # Get model updates since last sync
            updates = self.client.get_model_updates()
            
            # Update local state
            for update in updates:
                update_id = update.get("update_id")
                if update_id and update_id not in self.contributions["model_updates"]:
                    self.contributions["model_updates"][update_id] = update
                elif update_id:
                    # Update existing record with blockchain data
                    self.contributions["model_updates"][update_id].update({
                        "block_index": update.get("block_index"),
                        "block_hash": update.get("block_hash"),
                        "status": "confirmed"
                    })
            
            # Update last sync time
            self.contributions["last_sync"] = time.time()
            
            # Save contributions
            self._save_contributions()
            
            logger.info(f"Synchronized with blockchain: {len(updates)} new updates")
            return True
        except Exception as e:
            logger.error(f"Failed to sync with blockchain: {e}")
            return False
    
    def queue_transaction(self, recipient: str, amount: int, 
                         gas_price: Optional[float] = None, 
                         gas_limit: int = 100000,
                         model_update: Optional[Dict] = None, 
                         data: Optional[str] = None) -> None:
        """
        Queue a transaction for processing.
        
        Args:
            recipient: Address of the recipient
            amount: Amount to transfer
            gas_price: Price per unit of gas (optional)
            gas_limit: Maximum gas allowed for transaction
            model_update: Optional AI model update data
            data: Optional transaction data
        """
        # Use configured gas price if not provided
        if gas_price is None:
            gas_price = self.config.get("gas_price", 0.00001)
            
        # Add to transaction queue
        self.tx_queue.put({
            "recipient": recipient,
            "amount": amount,
            "gas_price": gas_price,
            "gas_limit": gas_limit,
            "model_update": model_update,
            "data": data
        })
        
        # Start transaction processor if not already running
        if not self.is_processing_txs:
            self.start_tx_processor()
    
    def submit_model_update(self, model_id: str, version: str, 
                           model_path: str, metrics: Dict) -> bool:
        """
        Submit a model update to the blockchain.
        
        Args:
            model_id: Identifier for the model
            version: Version of the model update
            model_path: Path to the model file or directory
            metrics: Performance metrics for the update
            
        Returns:
            bool: True if the update was accepted
        """
        try:
            # Submit model update
            success = self.client.submit_model_update(
                model_id=model_id,
                version=version,
                model_path=model_path,
                metrics=metrics
            )
            
            if success:
                # Calculate model hash for tracking
                model_hash = self._calculate_model_hash(model_path)
                
                # Record contribution locally
                update_record = {
                    "model_id": model_id,
                    "version": version,
                    "model_hash": model_hash,
                    "metrics": metrics,
                    "timestamp": time.time(),
                    "status": "pending"
                }
                
                self.contributions["model_updates"][model_hash] = update_record
                self._save_contributions()
                
                logger.info(f"Submitted model update for {model_id} v{version}")
                return True
            else:
                logger.warning(f"Failed to submit model update for {model_id} v{version}")
                return False
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model file or directory."""
        path = Path(model_path)
        
        if path.is_file():
            # Hash single file
            return self._hash_file(path)
        elif path.is_dir():
            # Hash directory contents
            file_hashes = []
            for file_path in sorted(path.glob('**/*')):
                if file_path.is_file():
                    file_hashes.append(self._hash_file(file_path))
            
            # Combine file hashes
            combined_hash = hashlib.sha256(''.join(file_hashes).encode()).hexdigest()
            return combined_hash
        else:
            raise ValueError(f"Invalid model path: {model_path}")
    
    def _hash_file(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
    
    def record_training_session(self, session_id: str, model_id: str, 
                               trainer_id: str, config: Dict) -> bool:
        """
        Record a training session on the blockchain.
        
        Args:
            session_id: Unique identifier for the session
            model_id: Identifier for the model being trained
            trainer_id: Identifier for the trainer
            config: Training configuration
            
        Returns:
            bool: True if the session was recorded successfully
        """
        try:
            # Create session record
            session_record = {
                "session_id": session_id,
                "model_id": model_id,
                "trainer_id": trainer_id,
                "config": config,
                "start_time": time.time(),
                "status": "active",
                "updates": [],
                "metrics": {}
            }
            
            # Record locally
            self.contributions["training_sessions"][session_id] = session_record
            self._save_contributions()
            
            # Queue transaction to blockchain
            self.queue_transaction(
                recipient="ONI_NETWORK",
                amount=0,
                model_update={
                    "type": "training_session",
                    "session_id": session_id,
                    "model_id": model_id,
                    "trainer_id": trainer_id,
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"Recorded training session {session_id} for {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error recording training session: {e}")
            return False
    
    def update_training_progress(self, session_id: str, step: int, 
                               metrics: Dict, model_path: Optional[str] = None) -> bool:
        """
        Update training progress on the blockchain.
        
        Args:
            session_id: Identifier for the training session
            step: Current training step
            metrics: Training metrics
            model_path: Optional path to model checkpoint
            
        Returns:
            bool: True if the update was recorded successfully
        """
        try:
            # Check if session exists
            if session_id not in self.contributions["training_sessions"]:
                logger.warning(f"Training session {session_id} not found")
                return False
                
            session = self.contributions["training_sessions"][session_id]
            
            # Update session record
            session["current_step"] = step
            session["metrics"] = metrics
            session["last_update"] = time.time()
            
            # If model checkpoint provided, submit as model update
            if model_path:
                model_id = session["model_id"]
                version = f"{session['model_id']}_step_{step}"
                
                success = self.submit_model_update(
                    model_id=model_id,
                    version=version,
                    model_path=model_path,
                    metrics=metrics
                )
                
                if success:
                    # Record update in session
                    model_hash = self._calculate_model_hash(model_path)
                    session["updates"].append({
                        "step": step,
                        "model_hash": model_hash,
                        "timestamp": time.time()
                    })
            
            # Save contributions
            self._save_contributions()
            
            logger.info(f"Updated training progress for session {session_id} at step {step}")
            return True
        except Exception as e:
            logger.error(f"Error updating training progress: {e}")
            return False
    
    def complete_training_session(self, session_id: str, final_metrics: Dict, 
                                final_model_path: Optional[str] = None) -> bool:
        """
        Mark a training session as completed on the blockchain.
        
        Args:
            session_id: Identifier for the training session
            final_metrics: Final training metrics
            final_model_path: Optional path to final model
            
        Returns:
            bool: True if the session was completed successfully
        """
        try:
            # Check if session exists
            if session_id not in self.contributions["training_sessions"]:
                logger.warning(f"Training session {session_id} not found")
                return False
                
            session = self.contributions["training_sessions"][session_id]
            
            # Update session record
            session["status"] = "completed"
            session["end_time"] = time.time()
            session["metrics"] = final_metrics
            
            # If final model provided, submit as model update
            if final_model_path:
                model_id = session["model_id"]
                version = f"{session['model_id']}_final"
                
                success = self.submit_model_update(
                    model_id=model_id,
                    version=version,
                    model_path=final_model_path,
                    metrics=final_metrics
                )
                
                if success:
                    # Record update in session
                    model_hash = self._calculate_model_hash(final_model_path)
                    session["final_model_hash"] = model_hash
            
            # Save contributions
            self._save_contributions()
            
            # Queue transaction to blockchain
            self.queue_transaction(
                recipient="ONI_NETWORK",
                amount=0,
                model_update={
                    "type": "training_completion",
                    "session_id": session_id,
                    "model_id": session["model_id"],
                    "trainer_id": session["trainer_id"],
                    "metrics": final_metrics,
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"Completed training session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error completing training session: {e}")
            return False
    
    def record_feedback(self, session_id: str, feedback_id: str, 
                      user_id: str, rating: float, compassion_score: float) -> bool:
        """
        Record human feedback on the blockchain.
        
        Args:
            session_id: Identifier for the training session
            feedback_id: Unique identifier for the feedback
            user_id: Identifier for the user providing feedback
            rating: Feedback rating (1-10)
            compassion_score: Compassion score (0-1)
            
        Returns:
            bool: True if the feedback was recorded successfully
        """
        try:
            # Validate inputs
            if rating < 1 or rating > 10:
                logger.warning(f"Invalid rating: {rating}, must be between 1 and 10")
                return False
                
            if compassion_score < 0 or compassion_score > 1:
                logger.warning(f"Invalid compassion score: {compassion_score}, must be between 0 and 1")
                return False
                
            # Create feedback record
            feedback_record = {
                "feedback_id": feedback_id,
                "session_id": session_id,
                "user_id": user_id,
                "rating": rating,
                "compassion_score": compassion_score,
                "timestamp": time.time(),
                "status": "pending"
            }
            
            # Record locally
            self.contributions["feedback"][feedback_id] = feedback_record
            self._save_contributions()
            
            # Queue transaction to blockchain
            self.queue_transaction(
                recipient="ONI_NETWORK",
                amount=0,
                model_update={
                    "type": "feedback",
                    "feedback_id": feedback_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "rating": int(rating * 10),  # Scale to integer
                    "compassion_score": int(compassion_score * 100),  # Scale to integer
                    "timestamp": time.time()
                }
            )
            
            logger.info(f"Recorded feedback {feedback_id} for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def mine_pending_transactions(self) -> bool:
        """
        Mine pending transactions into a new block.
        
        Returns:
            bool: True if mining was successful
        """
        try:
            return self.client.mine_block()
        except Exception as e:
            logger.error(f"Error mining transactions: {e}")
            return False
    
    def get_contributor_rewards(self, contributor_id: str) -> float:
        """
        Get total rewards for a contributor.
        
        Args:
            contributor_id: Identifier for the contributor
            
        Returns:
            float: Total rewards in ONI tokens
        """
        try:
            # Check local cache first
            if contributor_id in self.contributions["rewards"]:
                return self.contributions["rewards"][contributor_id]
                
            # Query blockchain
            balance = self.client.get_balance(contributor_id)
            
            # Update local cache
            self.contributions["rewards"][contributor_id] = balance
            self._save_contributions()
            
            return balance
        except Exception as e:
            logger.error(f"Error getting contributor rewards: {e}")
            return 0.0
    
    def get_training_sessions(self, contributor_id: Optional[str] = None) -> List[Dict]:
        """
        Get training sessions for a contributor.
        
        Args:
            contributor_id: Optional identifier to filter by contributor
            
        Returns:
            List[Dict]: List of training sessions
        """
        sessions = []
        
        for session_id, session in self.contributions["training_sessions"].items():
            if contributor_id and session["trainer_id"] != contributor_id:
                continue
                
            sessions.append(session)
            
        return sessions
    
    def get_model_updates(self, model_id: Optional[str] = None) -> List[Dict]:
        """
        Get model updates from the blockchain.
        
        Args:
            model_id: Optional model ID to filter updates
            
        Returns:
            List[Dict]: List of model updates
        """
        updates = []
        
        for update_id, update in self.contributions["model_updates"].items():
            if model_id and update["model_id"] != model_id:
                continue
                
            updates.append(update)
            
        return updates
    
    def get_feedback(self, session_id: Optional[str] = None) -> List[Dict]:
        """
        Get feedback for a session.
        
        Args:
            session_id: Optional session ID to filter feedback
            
        Returns:
            List[Dict]: List of feedback records
        """
        feedback = []
        
        for feedback_id, record in self.contributions["feedback"].items():
            if session_id and record["session_id"] != session_id:
                continue
                
            feedback.append(record)
            
        return feedback
    
    def create_proof_of_contribute(self, contributor_id: str, contribution_type: str, 
                                 compute_hours: float, quality_score: float) -> Dict:
        """
        Create a proof of contribute record.
        
        Args:
            contributor_id: Identifier for the contributor
            contribution_type: Type of contribution (training, feedback, etc.)
            compute_hours: Hours of compute contributed
            quality_score: Quality score for the contribution (0-1)
            
        Returns:
            Dict: Proof of contribute record
        """
        # Create proof record
        proof = {
            "contributor_id": contributor_id,
            "contribution_type": contribution_type,
            "compute_hours": compute_hours,
            "quality_score": quality_score,
            "timestamp": time.time(),
            "hash": hashlib.sha256(f"{contributor_id}:{contribution_type}:{compute_hours}:{quality_score}:{time.time()}".encode()).hexdigest()
        }
        
        # Queue transaction to blockchain
        self.queue_transaction(
            recipient="ONI_NETWORK",
            amount=0,
            model_update={
                "type": "proof_of_contribute",
                "contributor_id": contributor_id,
                "contribution_type": contribution_type,
                "compute_hours": compute_hours,
                "quality_score": quality_score,
                "timestamp": time.time()
            }
        )
        
        logger.info(f"Created proof of contribute for {contributor_id}")
        return proof
    
    def verify_proof_of_contribute(self, proof: Dict) -> bool:
        """
        Verify a proof of contribute.
        
        Args:
            proof: Proof of contribute record
            
        Returns:
            bool: True if the proof is valid
        """
        try:
            # Verify proof hash
            expected_hash = hashlib.sha256(
                f"{proof['contributor_id']}:{proof['contribution_type']}:{proof['compute_hours']}:{proof['quality_score']}:{proof['timestamp']}".encode()
            ).hexdigest()
            
            if proof['hash'] != expected_hash:
                logger.warning(f"Invalid proof hash: {proof['hash']} != {expected_hash}")
                return False
                
            # Verify on blockchain
            # In a real implementation, this would verify the proof exists on chain
            # For this example, we'll just return True
            return True
        except Exception as e:
            logger.error(f"Error verifying proof: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the blockchain integration."""
        # Save contributions
        self._save_contributions()
        
        # Stop transaction processor
        self.stop_tx_processor()
        
        # Stop batch processor in client
        self.client.stop_batch_processor()
        
        # Stop background thread if running
        if self.sync_thread and self.sync_thread.is_alive():
            # Can't directly stop thread, but we can set a flag
            self.config["auto_sync"] = False
            logger.info("Blockchain integration shutdown initiated")


# Example usage
if __name__ == "__main__":
    # Create blockchain integration
    integration = ONIBlockchainIntegration()
    
    # Record a training session
    session_id = f"session_{int(time.time())}"
    model_id = "oni-nlp-v1"
    trainer_id = "trainer_1"
    
    integration.record_training_session(
        session_id=session_id,
        model_id=model_id,
        trainer_id=trainer_id,
        config={
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
    )
    
    # Update training progress
    for step in range(1, 11):
        integration.update_training_progress(
            session_id=session_id,
            step=step,
            metrics={
                "loss": 1.0 - (step * 0.1),
                "accuracy": step * 0.1,
                "step": step
            }
        )
        
        # Simulate training time
        time.sleep(1)
    
    # Complete training session
    integration.complete_training_session(
        session_id=session_id,
        final_metrics={
            "loss": 0.1,
            "accuracy": 0.9,
            "steps": 10
        }
    )
    
    # Record feedback
    integration.record_feedback(
        session_id=session_id,
        feedback_id=f"feedback_{int(time.time())}",
        user_id="user_1",
        rating=8.5,
        compassion_score=0.9
    )
    
    # Mine pending transactions
    integration.mine_pending_transactions()
    
    # Get contributor rewards
    rewards = integration.get_contributor_rewards(trainer_id)
    print(f"Rewards for {trainer_id}: {rewards} ONI")
    
    # Shutdown
    integration.shutdown()