import requests
import json
import hashlib
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import os
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONIBlockchainClient:
    """Client for interacting with the ONI blockchain network."""
    
    def __init__(self, node_url: str, api_key: Optional[str] = None, private_key: Optional[str] = None):
        """
        Initialize the ONI blockchain client.
        
        Args:
            node_url: URL of the ONI blockchain node
            api_key: Optional API key for authentication
            private_key: Optional private key for signing transactions
        """
        self.node_url = node_url
        self.api_key = api_key
        self.private_key = private_key
        self.public_key = self._derive_public_key(private_key) if private_key else None
        self.session = requests.Session()
        
        # Set up headers
        self.headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            self.headers["X-API-Key"] = api_key
            
        # Transaction queue for batching
        self.tx_queue = queue.Queue()
        self.batch_thread = None
        self.is_batching = False
        self.batch_size = 20
        self.batch_timeout = 2  # seconds
        
        # Cache for responses
        self.cache = {}
        self.cache_timeout = 30  # seconds
    
    def _derive_public_key(self, private_key: str) -> str:
        """Derive public key from private key (simplified)."""
        return hashlib.sha256(private_key.encode()).hexdigest()
    
    def _sign_data(self, data: Dict) -> str:
        """Sign data with private key (simplified)."""
        if not self.private_key:
            raise ValueError("Private key required for signing")
            
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256((data_str + self.private_key).encode()).hexdigest()
    
    def _generate_nonce(self) -> str:
        """Generate a unique nonce for transactions."""
        return str(uuid.uuid4())
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get a cached response if it exists and is not expired."""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
            # Remove expired cache entry
            del self.cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """Cache a response with the current timestamp."""
        self.cache[cache_key] = (response, time.time())
    
    def get_balance(self, address: Optional[str] = None) -> int:
        """
        Get the balance of an address.
        
        Args:
            address: The address to check, or None to use the client's address
            
        Returns:
            int: The balance in ONI tokens
        """
        address = address or self.public_key
        if not address:
            raise ValueError("Address required")
            
        cache_key = f"balance_{address}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response["balance"]
            
        try:
            response = self.session.get(
                f"{self.node_url}/balance/{address}",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the response
            self._cache_response(cache_key, result)
            
            return result["balance"]
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    def submit_transaction(self, recipient: str, amount: int, 
                          gas_price: float = 0.00001, gas_limit: int = 100000,
                          model_update: Optional[Dict] = None, data: Optional[str] = None,
                          batch: bool = False) -> Union[str, bool]:
        """
        Submit a transaction to the blockchain.
        
        Args:
            recipient: Address of the recipient
            amount: Amount to transfer
            gas_price: Price per unit of gas
            gas_limit: Maximum gas allowed for transaction
            model_update: Optional AI model update data
            data: Optional transaction data
            batch: Whether to batch this transaction with others
            
        Returns:
            str: Transaction hash if successful, or False if failed
        """
        if not self.public_key:
            raise ValueError("Public key required for transactions")
            
        transaction = {
            "sender": self.public_key,
            "recipient": recipient,
            "amount": amount,
            "gas_price": gas_price,
            "gas_limit": gas_limit,
            "model_update": model_update,
            "data": data,
            "nonce": self._generate_nonce(),
            "timestamp": time.time()
        }
        
        # Sign transaction if private key available
        if self.private_key:
            transaction["signature"] = self._sign_data(transaction)
        
        # If batching is enabled, add to queue and return transaction ID
        if batch:
            tx_id = hashlib.sha256(json.dumps(transaction, sort_keys=True).encode()).hexdigest()
            self.tx_queue.put((tx_id, transaction))
            
            # Start batch processor if not already running
            self._ensure_batch_processor()
            
            return tx_id
        
        # Otherwise, submit immediately
        try:
            response = self.session.post(
                f"{self.node_url}/transactions/new",
                headers=self.headers,
                json=transaction,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                logger.info(f"Transaction submitted: {self.public_key} -> {recipient}, {amount} ONI")
                return result.get("transaction_hash", "")
            else:
                logger.warning(f"Transaction failed: {result.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Failed to submit transaction: {e}")
            return False
    
    def _ensure_batch_processor(self) -> None:
        """Ensure the batch processor thread is running."""
        if self.batch_thread is None or not self.batch_thread.is_alive():
            self.is_batching = True
            self.batch_thread = threading.Thread(target=self._process_transaction_batch)
            self.batch_thread.daemon = True
            self.batch_thread.start()
            logger.info("Transaction batch processor started")
    
    def _process_transaction_batch(self) -> None:
        """Process transactions in batches."""
        while self.is_batching:
            batch = []
            batch_ids = []
            
            # Collect transactions for the batch
            try:
                # Get first transaction with timeout
                tx_id, tx = self.tx_queue.get(timeout=self.batch_timeout)
                batch.append(tx)
                batch_ids.append(tx_id)
                self.tx_queue.task_done()
                
                # Get more transactions without blocking
                while len(batch) < self.batch_size:
                    try:
                        tx_id, tx = self.tx_queue.get_nowait()
                        batch.append(tx)
                        batch_ids.append(tx_id)
                        self.tx_queue.task_done()
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                # No transactions in queue
                time.sleep(0.1)
                continue
                
            # Submit batch if we have transactions
            if batch:
                try:
                    response = self.session.post(
                        f"{self.node_url}/transactions/batch",
                        headers=self.headers,
                        json={"transactions": batch},
                        timeout=20
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get("success"):
                        logger.info(f"Batch of {len(batch)} transactions submitted successfully")
                    else:
                        logger.warning(f"Batch submission failed: {result.get('message')}")
                        
                except Exception as e:
                    logger.error(f"Failed to submit transaction batch: {e}")
    
    def stop_batch_processor(self) -> None:
        """Stop the batch processor thread."""
        self.is_batching = False
        if self.batch_thread and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=self.batch_timeout + 1)
            logger.info("Transaction batch processor stopped")
    
    def submit_model_update(self, model_id: str, version: str, 
                           model_path: str, metrics: Dict) -> bool:
        """
        Submit an AI model update to the blockchain.
        
        Args:
            model_id: Identifier for the model
            version: Version of the model update
            model_path: Path to the model file or directory
            metrics: Performance metrics for the update
            
        Returns:
            bool: True if the update was accepted
        """
        # Calculate model data hash
        model_hash = self._hash_model_file(model_path)
        
        # Create model update package
        model_update = {
            "model_id": model_id,
            "version": version,
            "update_type": "weights",
            "timestamp": time.time(),
            "metrics": metrics,
            "data_hash": model_hash
        }
        
        # Submit transaction with model update
        tx_hash = self.submit_transaction(
            recipient="ONI_NETWORK",  # Special recipient for model updates
            amount=0,  # No direct transfer, reward will be calculated by network
            model_update=model_update
        )
        
        return bool(tx_hash)
    
    def _hash_model_file(self, model_path: str) -> str:
        """
        Calculate hash of model file or directory.
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            str: SHA-256 hash of the model data
        """
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
    
    def mine_block(self) -> bool:
        """
        Request the node to mine a new block.
        
        Returns:
            bool: True if mining was successful
        """
        try:
            response = self.session.post(
                f"{self.node_url}/mine",
                headers=self.headers,
                json={"miner": self.public_key} if self.public_key else {},
                timeout=30  # Longer timeout for mining
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                logger.info(f"Block mined: {result.get('block_index')} with hash {result.get('block_hash')}")
                return True
            else:
                logger.warning(f"Mining failed: {result.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Failed to mine block: {e}")
            return False
    
    def get_chain(self) -> List[Dict]:
        """
        Get the full blockchain.
        
        Returns:
            List[Dict]: The blockchain as a list of blocks
        """
        cache_key = "chain"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response["chain"]
            
        try:
            response = self.session.get(
                f"{self.node_url}/chain",
                headers=self.headers,
                timeout=30  # Longer timeout for full chain
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the response
            self._cache_response(cache_key, result)
            
            return result["chain"]
        except Exception as e:
            logger.error(f"Failed to get chain: {e}")
            return []
    
    def get_pending_transactions(self) -> List[Dict]:
        """
        Get pending transactions.
        
        Returns:
            List[Dict]: List of pending transactions
        """
        try:
            response = self.session.get(
                f"{self.node_url}/transactions/pending",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()["transactions"]
        except Exception as e:
            logger.error(f"Failed to get pending transactions: {e}")
            return []
    
    def get_model_updates(self, model_id: Optional[str] = None) -> List[Dict]:
        """
        Get model updates from the blockchain.
        
        Args:
            model_id: Optional model ID to filter updates
            
        Returns:
            List[Dict]: List of model updates
        """
        cache_key = f"model_updates_{model_id or 'all'}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response["updates"]
            
        try:
            url = f"{self.node_url}/models/updates"
            if model_id:
                url += f"/{model_id}"
                
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the response
            self._cache_response(cache_key, result)
            
            return result["updates"]
        except Exception as e:
            logger.error(f"Failed to get model updates: {e}")
            return []
    
    def download_model_update(self, update_id: str, output_path: str) -> bool:
        """
        Download a model update.
        
        Args:
            update_id: ID of the update to download
            output_path: Path to save the downloaded update
            
        Returns:
            bool: True if download was successful
        """
        try:
            response = self.session.get(
                f"{self.node_url}/models/download/{update_id}",
                headers=self.headers,
                stream=True,
                timeout=60  # Longer timeout for downloads
            )
            response.raise_for_status()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Downloaded model update {update_id} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download model update: {e}")
            return False
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the blockchain.
        
        Returns:
            Dict[str, Any]: Blockchain statistics
        """
        cache_key = "stats"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response["stats"]
            
        try:
            response = self.session.get(
                f"{self.node_url}/stats",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the response
            self._cache_response(cache_key, result)
            
            return result["stats"]
        except Exception as e:
            logger.error(f"Failed to get chain stats: {e}")
            return {}
    
    def verify_proof_of_compute(self, block_hash: str, nonce: int) -> bool:
        """
        Verify a proof of compute.
        
        Args:
            block_hash: Hash of the block
            nonce: Nonce used for the proof
            
        Returns:
            bool: True if the proof is valid
        """
        try:
            response = self.session.post(
                f"{self.node_url}/verify",
                headers=self.headers,
                json={"block_hash": block_hash, "nonce": nonce},
                timeout=10
            )
            response.raise_for_status()
            return response.json()["valid"]
        except Exception as e:
            logger.error(f"Failed to verify proof: {e}")
            return False
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """
        Get a transaction by its hash.
        
        Args:
            tx_hash: Hash of the transaction
            
        Returns:
            Dict: Transaction details, or None if not found
        """
        cache_key = f"tx_{tx_hash}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
            
        try:
            response = self.session.get(
                f"{self.node_url}/transaction/{tx_hash}",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the response
            self._cache_response(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get transaction: {e}")
            return None
    
    def get_gas_price(self) -> float:
        """
        Get the current gas price.
        
        Returns:
            float: Current gas price
        """
        try:
            response = self.session.get(
                f"{self.node_url}/gas-price",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()["gas_price"]
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return 0.00001  # Default gas price
    
    def estimate_gas(self, transaction: Dict) -> int:
        """
        Estimate gas for a transaction.
        
        Args:
            transaction: Transaction details
            
        Returns:
            int: Estimated gas
        """
        try:
            response = self.session.post(
                f"{self.node_url}/estimate-gas",
                headers=self.headers,
                json=transaction,
                timeout=10
            )
            response.raise_for_status()
            return response.json()["gas"]
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}")
            return 100000  # Default gas estimate


# Example usage
if __name__ == "__main__":
    # Create a client
    client = ONIBlockchainClient(
        node_url="http://localhost:5000",
        private_key="example_private_key"
    )
    
    # Get balance
    balance = client.get_balance()
    print(f"Balance: {balance} ONI")
    
    # Submit a model update
    model_metrics = {
        "accuracy": 0.85,
        "loss": 0.15,
        "convergence_rate": 0.75,
        "ethical_score": 0.9
    }
    
    success = client.submit_model_update(
        model_id="oni-nlp-v1",
        version="1.0.0",
        model_path="./models/oni-nlp-v1",
        metrics=model_metrics
    )
    
    if success:
        print("Model update submitted successfully")
    else:
        print("Failed to submit model update")
    
    # Mine a block
    if client.mine_block():
        print("Block mined successfully")
    else:
        print("Failed to mine block")
    
    # Get chain stats
    stats = client.get_chain_stats()
    print(f"Chain stats: {json.dumps(stats, indent=2)}")