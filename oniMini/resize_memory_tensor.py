import torch
import pickle
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

def migrate_memory(old_memory_file, new_memory_file, new_dim_size=1536):
    """
    Migrate memory embeddings from one dimension size to another using zero-padding.
    
    Args:
        old_memory_file (str): Path to the existing memory pickle file
        new_memory_file (str): Path to save the migrated memory
        new_dim_size (int): New embedding dimension size (default: 2048)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load existing memory
        with open(old_memory_file, "rb") as f:
            old_memory = pickle.load(f)
            
        if not old_memory:
            logger.warning("Old memory file is empty")
            return
            
        # Create new memory dictionary
        new_memory = {}
        
        # Process each embedding
        for text, embedding in old_memory.items():
            try:
                # Get original embedding size
                orig_size = embedding.size(-1)
                
                # Create new zero-padded embedding
                new_embedding = torch.zeros(new_dim_size, device=embedding.device)
                
                # Copy original values
                new_embedding[:orig_size] = embedding[:orig_size]
                
                # Store in new memory
                new_memory[text] = new_embedding
                
            except Exception as e:
                logger.error(f"Error processing embedding for text '{text}': {str(e)}")
                continue
        
        # Save new memory
        with open(new_memory_file, "wb") as f:
            # Ensure all tensors are on CPU before saving
            cpu_memory = {k: v.cpu() for k, v in new_memory.items()}
            pickle.dump(cpu_memory, f)
            
        logger.info(f"Successfully migrated {len(new_memory)} embeddings to dimension size {new_dim_size}")
        
        return new_memory
        
    except Exception as e:
        logger.error(f"Failed to migrate memory: {str(e)}")
        return None

def verify_migration(old_file, new_file):
    """
    Verify the migration was successful by comparing the memory files.
    
    Args:
        old_file (str): Path to the original memory file
        new_file (str): Path to the migrated memory file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load both memories
        with open(old_file, "rb") as f:
            old_memory = pickle.load(f)
        with open(new_file, "rb") as f:
            new_memory = pickle.load(f)
            
        # Check keys match
        if set(old_memory.keys()) != set(new_memory.keys()):
            logger.error("Key mismatch between old and new memory")
            return False
            
        # Check dimensions
        old_dim = next(iter(old_memory.values())).size(-1)
        new_dim = next(iter(new_memory.values())).size(-1)
        
        logger.info(f"Old dimension: {old_dim}, New dimension: {new_dim}")
        logger.info(f"Number of entries: {len(old_memory)}")
        
        # Verify content preservation
        for key in old_memory:
            old_emb = old_memory[key]
            new_emb = new_memory[key]
            
            # Check if original values are preserved
            if not torch.allclose(old_emb, new_emb[:old_emb.size(-1)]):
                logger.error(f"Content mismatch for key: {key}")
                return False
                
        logger.info("Migration verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False
