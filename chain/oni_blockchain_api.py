from flask import Flask, request, jsonify
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Union
import threading
import os
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename

# Import ONI blockchain components
from chain.oni_proof_of_compute import Blockchain, Transaction, Block, ONINode, MemPool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ONI-Blockchain-API")

# Initialize Flask app
app = Flask(__name__)

# Initialize blockchain node
node = None
node_lock = threading.Lock()

# Initialize blockchain
def initialize_blockchain(node_id: str, private_key: str):
    global node
    with node_lock:
        node = ONINode(node_id, private_key)
        node.start()  # Start background threads
        logger.info(f"Initialized blockchain node: {node_id}")

# API routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "node_id": node.node_id if node else None,
        "version": "1.0.0"
    })

@app.route('/balance/<address>', methods=['GET'])
def get_balance(address: str):
    """Get balance for an address."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    balance = node.blockchain.get_balance(address)
    return jsonify({
        "address": address,
        "balance": balance,
        "timestamp": time.time()
    })

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    """Create a new transaction."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    data = request.get_json()
    required_fields = ['sender', 'recipient', 'amount']
    
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
        
    # Get optional fields
    gas_price = data.get('gas_price', 0.00001)
    gas_limit = data.get('gas_limit', 100000)
    model_update = data.get('model_update')
    tx_data = data.get('data')
    private_key = data.get('private_key')
    
    # Create transaction
    success = node.blockchain.add_transaction(
        sender=data['sender'],
        recipient=data['recipient'],
        amount=data['amount'],
        gas_price=gas_price,
        gas_limit=gas_limit,
        model_update=model_update,
        data=tx_data,
        private_key=private_key
    )
    
    if success:
        # Get the transaction from mempool
        tx_hash = None
        for tx in node.blockchain.mempool.transactions.values():
            if (tx.sender == data['sender'] and 
                tx.recipient == data['recipient'] and 
                tx.amount == data['amount']):
                tx_hash = tx.tx_hash
                break
        
        # Broadcast transaction to network
        if tx_hash:
            transaction = node.blockchain.mempool.transactions[tx_hash]
            node.broadcast_transaction(transaction)
        
        return jsonify({
            "success": True,
            "message": "Transaction added to pending transactions",
            "transaction_hash": tx_hash,
            "timestamp": time.time()
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to add transaction",
            "timestamp": time.time()
        }), 400

@app.route('/transactions/batch', methods=['POST'])
def batch_transactions():
    """Process a batch of transactions."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    data = request.get_json()
    if 'transactions' not in data or not isinstance(data['transactions'], list):
        return jsonify({"error": "Invalid batch format"}), 400
        
    results = []
    for tx_data in data['transactions']:
        required_fields = ['sender', 'recipient', 'amount']
        if not all(field in tx_data for field in required_fields):
            results.append({
                "success": False,
                "message": "Missing required fields",
                "transaction_data": tx_data
            })
            continue
            
        # Get optional fields
        gas_price = tx_data.get('gas_price', 0.00001)
        gas_limit = tx_data.get('gas_limit', 100000)
        model_update = tx_data.get('model_update')
        tx_data_content = tx_data.get('data')
        private_key = tx_data.get('private_key')
        
        # Create transaction
        success = node.blockchain.add_transaction(
            sender=tx_data['sender'],
            recipient=tx_data['recipient'],
            amount=tx_data['amount'],
            gas_price=gas_price,
            gas_limit=gas_limit,
            model_update=model_update,
            data=tx_data_content,
            private_key=private_key
        )
        
        if success:
            # Get the transaction from mempool
            tx_hash = None
            for tx in node.blockchain.mempool.transactions.values():
                if (tx.sender == tx_data['sender'] and 
                    tx.recipient == tx_data['recipient'] and 
                    tx.amount == tx_data['amount']):
                    tx_hash = tx.tx_hash
                    break
            
            # Broadcast transaction to network
            if tx_hash:
                transaction = node.blockchain.mempool.transactions[tx_hash]
                node.broadcast_transaction(transaction)
            
            results.append({
                "success": True,
                "message": "Transaction added to pending transactions",
                "transaction_hash": tx_hash
            })
        else:
            results.append({
                "success": False,
                "message": "Failed to add transaction",
                "transaction_data": tx_data
            })
    
    return jsonify({
        "success": True,
        "results": results,
        "timestamp": time.time()
    })

@app.route('/mine', methods=['POST'])
def mine():
    """Mine a new block."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    data = request.get_json()
    miner = data.get('miner', node.node_id)
    
    # Start mining in a separate thread to avoid blocking
    def mine_async():
        new_block = node.mine_pending_transactions()
        if new_block:
            # Broadcast block to network
            node.broadcast_block(new_block)
    
    threading.Thread(target=mine_async).start()
    
    return jsonify({
        "success": True,
        "message": "Mining started",
        "miner": miner,
        "timestamp": time.time()
    })

@app.route('/chain', methods=['GET'])
def get_chain():
    """Get the full blockchain."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    chain_data = []
    for block in node.blockchain.chain:
        block_data = {
            "index": block.index,
            "timestamp": block.timestamp,
            "previous_hash": block.previous_hash,
            "hash": block.hash,
            "nonce": block.nonce,
            "merkle_root": block.merkle_root,
            "difficulty": block.difficulty,
            "gas_used": block.gas_used,
            "size": block.size,
            "transactions": [
                {
                    "tx_hash": tx.tx_hash,
                    "sender": tx.sender,
                    "recipient": tx.recipient,
                    "amount": tx.amount,
                    "gas_price": tx.gas_price,
                    "gas_limit": tx.gas_limit,
                    "timestamp": tx.timestamp,
                    "has_model_update": tx.model_update is not None,
                    "has_data": tx.data is not None
                } for tx in block.transactions
            ]
        }
        chain_data.append(block_data)
        
    return jsonify({
        "chain": chain_data,
        "length": len(chain_data),
        "timestamp": time.time()
    })

@app.route('/transactions/pending', methods=['GET'])
def get_pending_transactions():
    """Get pending transactions."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    transactions = [
        {
            "tx_hash": tx.tx_hash,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "gas_price": tx.gas_price,
            "gas_limit": tx.gas_limit,
            "fee": tx.calculate_fee(),
            "timestamp": tx.timestamp,
            "has_model_update": tx.model_update is not None,
            "has_data": tx.data is not None
        } for tx in node.blockchain.mempool.transactions.values()
    ]
    
    return jsonify({
        "transactions": transactions,
        "count": len(transactions),
        "timestamp": time.time()
    })

@app.route('/transaction/<tx_hash>', methods=['GET'])
def get_transaction(tx_hash: str):
    """Get a transaction by its hash."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    # Check mempool first
    if node.blockchain.mempool.contains_transaction(tx_hash):
        tx = node.blockchain.mempool.transactions[tx_hash]
        return jsonify({
            "tx_hash": tx.tx_hash,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "gas_price": tx.gas_price,
            "gas_limit": tx.gas_limit,
            "fee": tx.calculate_fee(),
            "timestamp": tx.timestamp,
            "status": "pending",
            "has_model_update": tx.model_update is not None,
            "has_data": tx.data is not None,
            "model_update": tx.model_update,
            "data": tx.data
        })
    
    # Check blockchain
    tx = node.blockchain.get_transaction(tx_hash)
    if tx:
        block_idx, tx_idx = node.blockchain.tx_index[tx_hash]
        block = node.blockchain.get_block_by_index(block_idx)
        
        return jsonify({
            "tx_hash": tx.tx_hash,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "gas_price": tx.gas_price,
            "gas_limit": tx.gas_limit,
            "fee": tx.calculate_fee(),
            "timestamp": tx.timestamp,
            "status": "confirmed",
            "block_index": block_idx,
            "block_hash": block.hash,
            "has_model_update": tx.model_update is not None,
            "has_data": tx.data is not None,
            "model_update": tx.model_update,
            "data": tx.data
        })
    
    return jsonify({"error": "Transaction not found"}), 404

@app.route('/models/updates', methods=['GET'])
@app.route('/models/updates/<model_id>', methods=['GET'])
def get_model_updates(model_id=None):
    """Get model updates from the blockchain."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    updates = []
    
    # Search for model updates in the blockchain
    for block in node.blockchain.chain:
        for tx in block.transactions:
            if tx.model_update:
                # Filter by model_id if provided
                if model_id and tx.model_update.get("model_id") != model_id:
                    continue
                    
                updates.append({
                    "block_index": block.index,
                    "block_hash": block.hash,
                    "tx_hash": tx.tx_hash,
                    "timestamp": tx.timestamp,
                    "sender": tx.sender,
                    "recipient": tx.recipient,
                    "update_id": tx.model_update.get("data_hash"),
                    "model_id": tx.model_update.get("model_id"),
                    "version": tx.model_update.get("version"),
                    "metrics": tx.model_update.get("metrics")
                })
    
    return jsonify({
        "updates": updates,
        "count": len(updates),
        "timestamp": time.time()
    })

@app.route('/models/download/<update_id>', methods=['GET'])
def download_model_update(update_id):
    """Download a model update."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    # Get model update from cache
    update_data = node.get_model_update(update_id)
    
    if not update_data:
        return jsonify({
            "error": "Model update not found",
            "update_id": update_id
        }), 404
        
    # In a real implementation, this would serve the file
    # For this example, we'll just return the data
    return jsonify({
        "update_id": update_id,
        "data": update_data,
        "timestamp": time.time()
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get blockchain statistics."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    stats = node.blockchain.get_chain_stats()
    
    return jsonify({
        "stats": stats,
        "timestamp": time.time()
    })

@app.route('/verify', methods=['POST'])
def verify_proof():
    """Verify a proof of compute."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    data = request.get_json()
    
    if 'block_hash' not in data or 'nonce' not in data:
        return jsonify({"error": "Missing required fields"}), 400
        
    # Create a dummy block to verify
    dummy_block = Block(
        index=0,
        previous_hash="0",
        timestamp=time.time(),
        transactions=[],
        nonce=data['nonce']
    )
    
    # Verify the proof
    is_valid = node.blockchain.proof_of_compute.validate_proof(
        dummy_block,
        data['block_hash']
    )
    
    return jsonify({
        "valid": is_valid,
        "block_hash": data['block_hash'],
        "nonce": data['nonce'],
        "timestamp": time.time()
    })

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    """Register new nodes in the network."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    data = request.get_json()
    
    if 'nodes' not in data:
        return jsonify({"error": "Missing nodes list"}), 400
        
    for node_address in data['nodes']:
        node.register_peer(node_address)
        
    return jsonify({
        "success": True,
        "message": f"Added {len(data['nodes'])} nodes",
        "total_nodes": len(node.peers),
        "timestamp": time.time()
    })

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    """Resolve conflicts between nodes."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    # In a real implementation, this would fetch chains from peers
    # For this example, we'll assume no conflicts
    replaced = False
    
    return jsonify({
        "success": True,
        "message": "Chain is authoritative" if not replaced else "Chain was replaced",
        "chain_length": len(node.blockchain.chain),
        "timestamp": time.time()
    })

@app.route('/gas-price', methods=['GET'])
def get_gas_price():
    """Get the current gas price."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    # In a real implementation, this would calculate based on network congestion
    # For this example, we'll return a fixed value
    return jsonify({
        "gas_price": node.blockchain.chain[-1].transactions[0].gas_price if node.blockchain.chain and node.blockchain.chain[-1].transactions else 0.00001,
        "timestamp": time.time()
    })

@app.route('/estimate-gas', methods=['POST'])
def estimate_gas():
    """Estimate gas for a transaction."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    data = request.get_json()
    required_fields = ['sender', 'recipient', 'amount']
    
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
        
    # Create a dummy transaction to estimate gas
    tx = Transaction(
        sender=data['sender'],
        recipient=data['recipient'],
        amount=data['amount'],
        gas_price=data.get('gas_price', 0.00001),
        gas_limit=data.get('gas_limit', 100000),
        model_update=data.get('model_update'),
        data=data.get('data')
    )
    
    gas = tx.calculate_gas()
    
    return jsonify({
        "gas": gas,
        "fee": gas * tx.gas_price,
        "timestamp": time.time()
    })

@app.route('/proof-of-compute/submit', methods=['POST'])
def submit_proof_of_compute():
    """Submit proof of compute for a model update."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    # Check if form data or JSON
    if request.content_type and 'multipart/form-data' in request.content_type:
        data = request.form
        required_fields = ['model_id', 'version', 'metrics', 'compute_time']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Check if model file was uploaded
        if 'model_file' not in request.files:
            return jsonify({"error": "Missing model file"}), 400
            
        model_file = request.files['model_file']
        if model_file.filename == '':
            return jsonify({"error": "Empty model filename"}), 400
            
        # Save model file temporarily
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        
        filename = secure_filename(model_file.filename)
        model_path = temp_dir / f"{data['model_id']}_{data['version']}_{int(time.time())}_{filename}"
        model_file.save(model_path)
        
        try:
            # Parse metrics
            metrics = json.loads(data['metrics']) if isinstance(data['metrics'], str) else data['metrics']
            
            # Submit model update
            success = node.submit_model_update(
                model_id=data['model_id'],
                version=data['version'],
                update_data={"path": str(model_path)},
                metrics=metrics
            )
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Proof of compute submitted successfully",
                    "timestamp": time.time()
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to submit proof of compute",
                    "timestamp": time.time()
                }), 400
        finally:
            # Clean up temporary file
            if model_path.exists():
                os.remove(model_path)
    else:
        # JSON data
        data = request.get_json()
        required_fields = ['model_id', 'version', 'metrics', 'update_data']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Submit model update
        success = node.submit_model_update(
            model_id=data['model_id'],
            version=data['version'],
            update_data=data['update_data'],
            metrics=data['metrics']
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Proof of compute submitted successfully",
                "timestamp": time.time()
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to submit proof of compute",
                "timestamp": time.time()
            }), 400

@app.route('/node/status', methods=['GET'])
def get_node_status():
    """Get node status."""
    if not node:
        return jsonify({"error": "Node not initialized"}), 500
        
    status = node.get_node_status()
    
    return jsonify({
        "status": status,
        "timestamp": time.time()
    })

# Start the blockchain node
def start_blockchain_node(node_id: str, private_key: str, host: str = '0.0.0.0', port: int = 5000):
    """
    Start the blockchain node and API server.
    
    Args:
        node_id: ID for the blockchain node
        private_key: Private key for the node
        host: Host to bind the API server
        port: Port to bind the API server
    """
    # Initialize blockchain node
    initialize_blockchain(node_id, private_key)
    
    # Start API server
    app.run(host=host, port=port, threaded=True)

# Shutdown handler
def shutdown_node():
    """Shutdown the blockchain node."""
    global node
    if node:
        node.stop()
        logger.info("Blockchain node shutdown complete")

# Register shutdown handler
import atexit
atexit.register(shutdown_node)

if __name__ == "__main__":
    # Generate a simple private key if not provided
    private_key = os.environ.get("ONI_PRIVATE_KEY", hashlib.sha256(str(time.time()).encode()).hexdigest())
    
    # Generate node ID from private key
    node_id = hashlib.sha256(private_key.encode()).hexdigest()[:8]
    
    # Get host and port from environment
    host = os.environ.get("ONI_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ONI_API_PORT", "5000"))
    
    logger.info(f"Starting ONI blockchain node {node_id} on {host}:{port}")
    
    # Start blockchain node and API server
    start_blockchain_node(node_id, private_key, host, port)