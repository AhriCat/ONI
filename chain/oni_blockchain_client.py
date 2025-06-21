import requests
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ONI-Blockchain-Client")

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
    
    def _derive_public_key(self, private_key: str) -> str:
        """Derive public key from private key (simplified)."""
        return hashlib.sha256(private_key.encode()).hexdigest()
    
    def _sign_data(self, data: Dict) -> str:
        """Sign data with private key (simplified)."""
        if not self.private_key:
            raise ValueError("Private key required for signing")
            
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256((data_str + self.private_key).encode()).hexdigest()
    
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
            
        try:
            response = self.session.get(
                f"{self.node_url}/balance/{address}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["balance"]
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    def submit_transaction(self, recipient: str, amount: int, 
                          model_update: Optional[Dict] = None) -> bool:
        """
        Submit a transaction to the blockchain.
        
        Args:
            recipient: Address of the recipient
            amount: Amount to transfer
            model_update: Optional AI model update data
            
        Returns:
            bool: True if transaction was submitted successfully
        """
        if not self.public_key:
            raise ValueError("Public key required for transactions")
            
        transaction = {
            "sender": self.public_key,
            "recipient": recipient,
            "amount": amount,
            "model_update": model_update,
            "timestamp": time.time()
        }
        
        # Sign transaction if private key available
        if self.private_key:
            transaction["signature"] = self._sign_data(transaction)
        
        try:
            response = self.session.post(
                f"{self.node_url}/transactions/new",
                headers=self.headers,
                json=transaction
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                logger.info(f"Transaction submitted: {self.public_key} -> {recipient}, {amount} ONI")
                return True
            else:
                logger.warning(f"Transaction failed: {result.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Failed to submit transaction: {e}")
            return False
    
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
        return self.submit_transaction(
            recipient="ONI_NETWORK",  # Special recipient for model updates
            amount=0,  # No direct transfer, reward will be calculated by network
            model_update=model_update
        )
    
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
                json={"miner": self.public_key} if self.public_key else {}
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
        try:
            response = self.session.get(
                f"{self.node_url}/chain",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["chain"]
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
                headers=self.headers
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
        try:
            url = f"{self.node_url}/models/updates"
            if model_id:
                url += f"/{model_id}"
                
            response = self.session.get(
                url,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["updates"]
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
                stream=True
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
        try:
            response = self.session.get(
                f"{self.node_url}/stats",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["stats"]
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
                json={"block_hash": block_hash, "nonce": nonce}
            )
            response.raise_for_status()
            return response.json()["valid"]
        except Exception as e:
            logger.error(f"Failed to verify proof: {e}")
            return False


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