import os
import json
import numpy as np
import random
import PyPDF2
import sqlite3
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import threading
import time
import pygame  # For rendering
import math
from memory.episodic_memory import EpisodicBuffer, EpisodicEmbeddingLayer
from memory.fading_memory import FadingMemorySystem
from memory.heuristic_manager import HeuristicManager
from memory.hopfield import SparceHopfieldNetwork, ModernContinuousHopfieldNetwork
from memory.mem_handler import MemoryInterferenceHandler
from memory.memory_consolidator import MemoryCosolidator
from memory.semantic_memory import SemanticMemoryLayer, TextPatternFinder
from memory.snapshot_memory import SnapshotMemorySystem
from memory.spatial_memory import SpatialMemoryModule
from memory.volatile_memory import VolatileMemory
# ===========================
# Memory Manager
# ===========================

class Memory:
    def __init__(self, tokenizer, working_memory_capacity: int = 5, ltm_capacity: int = 10000000000000):
        self.tokenizer = tokenizer
        self.working_memory_capacity = working_memory_capacity
        self.context = {}
        self.ltm_capacity = ltm_capacity
        self.working_memory = []  # Short-term working memory as a list
        self.semantic_memory = {}  # Store generalized knowledge
        self.ltm = ltm  # Long-term memory list
        self.ltm_summary = {}  # Summary or knowledge graph of LTM
        self.episodic_memory_path = 'C:/Users/jonny/Documents/PATH/ONI/ltm/episodes/'
        self.semantic_memory_path = os.path.join('C:/Users/jonny/Documents/PATH/ONI/ltm_path/', 'semantic_memory.json')
        self.ltm_summary_path = os.path.join('C:/Users/jonny/Documents/PATH/ONI/ltm_path/', "ltm_data.json")
        self.load_long_term_memory()
        self.episodic_embeddings = {}  # To store episodic embeddings
        self.semantic_embeddings = {}  # To store semantic embeddings
        self.episodic_layer = EpisodicEmbeddingLayer(input_dim=8192, output_dim=8192)
        self.semantic_layer = SemanticMemoryLayer(input_dim=8192, output_dim=8192)
        self.episodic_layer.to(device)  # Adjust device as needed
        self.semantic_layer.to(device)  # Adjust device as needed
        
        # New memory components
        self.episodic_buffer = EpisodicBuffer(hidden_dim=896, buffer_size=working_memory_capacity)
        self.continuous_hopfield = ModernContinuousHopfieldNetwork(hidden_dim=896)
        self.memory_consolidator = MemoryConsolidator()
        self.interference_handler = MemoryInterferenceHandler()
        
        # Sleep state tracking
        self.is_sleeping = False
        self.last_sleep_time = time.time()
        self.sleep_interval = 3600  # Default: consolidate every hour
        self.fading = FadingMemorySystem()
        self.snapshot = SnapshotMemory()
    def cleanup(self):
        """Release any held resources."""
        self.memory.data.zero_()
        torch.cuda.empty_cache()
        
    def update_context(self, key: str, value: str):
        self.context[key] = value

    def get_context(self) -> Dict[str, str]:
        return self.context

    def handle_media(self, data: str) -> Optional[Dict[str, str]]:
        """Handle different types of media files for episodic memory."""
        media_extensions = {
            '.mov': 'video',
            '.mp4': 'video',
            '.avi': 'video',
            '.wav': 'audio',
            '.mp3': 'audio',
            '.txt': 'plaintext',
            '.pdf': 'PDF',
            '.doc': 'Document File',
            '.odt': 'Open Document TXT',
            '.py': 'python file',
            '.html': 'website',
            '.js': 'javascript',
            '.css': 'styles'
        }
        file_extension = os.path.splitext(data)[1]
        media_type = media_extensions.get(file_extension, 'unknown')

        if media_type in ['video', 'audio']:
            return {'type': media_type, 'path': data}
        return None

    def load_long_term_memory(self):
        """Load semantic memory and LTM summary from files."""
        self.semantic_memory = self._load_json(self.semantic_memory_path)
        self.ltm_summary = self._load_json(self.ltm_summary_path)

    @staticmethod
    def _load_json(file_path: str) -> Dict:
        """Helper function to load JSON data from a file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return {}
        return {}

    def update_episodic_memory(self, data: bytes, data_type: str, key: str):
        """Update episodic memory by storing media files."""
        file_path = os.path.join(self.episodic_memory_path, f"{key}_{data_type}")
        os.makedirs(self.episodic_memory_path, exist_ok=True)

        # Store the media file
        with open(file_path, 'wb') as file:
            file.write(data)

        # Update LTM summary
        self.ltm_summary[key] = {"data_type": data_type, "path": file_path}

        # Convert the data into a tensor for embedding (assuming the data is in a suitable format for embedding)
        data_tensor = torch.tensor(list(data), dtype=torch.float32).unsqueeze(0)  # Example tensor conversion

        # Update the episodic embedding layer
        embedding = self.episodic_layer(data_tensor)
        self.episodic_embeddings[key] = embedding

    def retrieve_from_episodic(self, key: str) -> Optional[str]:
        """Retrieve data from episodic memory."""
        return self.ltm_summary.get(key, {}).get('path', None)

    def update_semantic_memory(self, data: str, data_type: str, key: str):
        """Update semantic memory by storing processed semantic information and updating embeddings."""
        file_path = os.path.join(self.semantic_memory_path, f"{key}_{data_type}")

        # Check if the semantic_memory_path is a directory or a file
        if not os.path.isdir(self.semantic_memory_path):
            # If it's a file, raise an error or handle accordingly
            if os.path.exists(self.semantic_memory_path):
                raise FileExistsError(f"A file exists at the path: {self.semantic_memory_path}. Unable to create directory.")
            else:
                os.makedirs(self.semantic_memory_path, exist_ok=True)

        # Store the semantic data file
        with open(file_path, 'w') as file:
            file.write(data)

        # Update LTM summary
        self.ltm_summary[key] = {'data_type': data_type, 'path': file_path}

        # Convert the data into a tensor for embedding
        data_tensor = torch.tensor(list(data), dtype=torch.float32).unsqueeze(0)  # Example tensor conversion

        # Update the semantic embedding layer
        embedding = self.semantic_layer(data_tensor)
        self.semantic_embeddings[key] = embedding

    def lookup_token(self, token: str) -> int:
        """Lookup the index of a token in the semantic memory."""
        return self.semantic_memory.get(token, -1)

    def meditate(self):
        """Compress and refine semantic and working memory."""
        unique_data = set(list(self.semantic_memory.values()) + self.working_memory)
        self.semantic_memory = {token: idx for idx, token in enumerate(unique_data)}

    def sleep(self):
        """Main sleep function to stop all processes and consolidate memories."""
        print("AI is going to sleep...")
        self.is_sleeping = True
        
        # Consolidate memories
        self.meditate()
        
        # Perform memory consolidation
        episodic_memories = self._get_episodic_memories()
        semantic_memories = self._get_semantic_memories()
        
        consolidated_episodic, consolidated_semantic = self.memory_consolidator.consolidate_memories(
            episodic_memories, semantic_memories
        )
        
        # Update memories with consolidated versions
        self._update_consolidated_memories(consolidated_episodic, consolidated_semantic)
        
        # Save to disk
        self.save_long_term_memory()
        
        # Update sleep state
        self.is_sleeping = False
        self.last_sleep_time = time.time()
        
        print("AI has woken up with refreshed memories.")

    def save_long_term_memory(self):
        """Save semantic memory and LTM summary."""
        self._save_json(self.semantic_memory_path, self.semantic_memory)
        self._save_json(self.ltm_summary_path, self.ltm_summary)
        torch.cuda.empty_cache()

    @staticmethod
    def _save_json(file_path: str, data: Dict):
        """Helper function to save JSON data to a file."""
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def handle_pdf(self, pdf_path: str):
        """Extract text from PDF and add to semantic memory."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    self.update_semantic_memory(text, data_type='PDF', key=f'pdf_page_{page_num}')
            print(f"PDF '{os.path.basename(pdf_path)}' added to semantic memory.")
        except Exception as e:
            print(f"Error processing PDF: {e}")

    def _get_mp4_data(self, directory: str) -> Dict[str, bytes]:
        mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
        mp4_data = {}

        for file in mp4_files:
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                mp4_data[file] = f.read()

        return mp4_data

    def update_memory(self, stm_data: str, episodic_data: bytes = None, episodic_key: str = None, semantic_data: str = None):
        """Update all memory stores with new data."""
        # Update working memory
        self.working_memory.append(stm_data)
        if len(self.working_memory) > self.working_memory_capacity:
            self.working_memory.pop(0)  # Remove oldest entry to maintain capacity
            
        # Update episodic buffer with new data
        if episodic_data is not None or semantic_data is not None:
            # Convert data to tensors
            episodic_tensor = None
            semantic_tensor = None
            
            if episodic_data is not None:
                episodic_tensor = torch.tensor(list(episodic_data), dtype=torch.float32).unsqueeze(0)
                
            if semantic_data is not None:
                semantic_tensor = torch.tensor(list(semantic_data), dtype=torch.float32).unsqueeze(0)
            
            # Create query from current context
            context_str = " ".join(self.context.values())
            context_tensor = torch.tensor(list(context_str), dtype=torch.float32).unsqueeze(0)
            
            # Update episodic buffer
            self.episodic_buffer(context_tensor, episodic_tensor, semantic_tensor)
            
            # Store in continuous Hopfield network for associative retrieval
            if episodic_tensor is not None:
                self.continuous_hopfield.store(episodic_tensor)
            if semantic_tensor is not None:
                self.continuous_hopfield.store(semantic_tensor)
        
        # Update traditional memory stores
        if episodic_data and episodic_key:
            self.update_episodic_memory(episodic_data, data_type='audio', key=episodic_key)
        if semantic_data and episodic_key:
            self.update_semantic_memory(semantic_data, data_type='text', key=episodic_key)
            
        # Check if it's time to sleep
        if time.time() - self.last_sleep_time > self.sleep_interval:
            self.sleep()
        else:
            # Just save without full consolidation
            self.save_long_term_memory()

    def categorize_and_store(self, db_path: str = 'personality.db'):
        """Categorize memory items and store them in a SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personalities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                race TEXT,
                origin TEXT,
                age INTEGER,
                type TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                personality_id INTEGER,
                timestamp TEXT,
                input_text TEXT,
                response_text TEXT,
                FOREIGN KEY (personality_id) REFERENCES personalities(id)
            )
        """)
        conn.commit()
        conn.close()

    def get_experiences(self) -> Dict[str, bytes]:
        """Retrieve MP4 data from episodes."""
        return self._get_mp4_data(self.episodic_memory_path)

    def integrate_insights(self, insights: Dict, model):
        """Integrate insights into the system's decision-making process."""
        # Placeholder for integration logic
        pass

    def reflect_on_experience(self, experiences: Dict[str, bytes], model):
        """
        Analyze experiences and integrate insights.

        Args:
            experiences (dict): Dictionary of experiences.
            model: The AGI model for integration.
        """
        # Analyze experiences (e.g., identify patterns, successes, areas for improvement)
        insights = self.analyze_experiences(experiences)
        self.integrate_insights(insights, model)

    def analyze_experiences(self, experiences: Dict[str, bytes]) -> Dict:
        """
        Analyze experiences to extract insights.

        Args:
            experiences (dict): Dictionary of experiences.

        Returns:
            dict: Extracted insights.
        """
        # Placeholder for analysis logic
        return {"patterns": [], "improvements": []}
        
    def _get_episodic_memories(self) -> Dict[str, Any]:
        """Get all episodic memories for consolidation"""
        # This is a simplified implementation
        # In a real system, this would retrieve all episodic memories from storage
        episodic_memories = {}
        
        # Convert episodic embeddings to memories
        for key, embedding in self.episodic_embeddings.items():
            episodic_memories[key] = {
                'embedding': embedding.detach().cpu().numpy().tolist(),
                'path': self.ltm_summary.get(key, {}).get('path', ''),
                'data_type': self.ltm_summary.get(key, {}).get('data_type', ''),
                'access_count': 1,  # Default value
                'last_access': time.time(),
                'timestamp': time.time()
            }
        
        return episodic_memories
    
    def _get_semantic_memories(self) -> Dict[str, Any]:
        """Get all semantic memories for consolidation"""
        # This is a simplified implementation
        # In a real system, this would retrieve all semantic memories from storage
        semantic_memories = {}
        
        # Convert semantic embeddings to memories
        for key, embedding in self.semantic_embeddings.items():
            semantic_memories[key] = {
                'embedding': embedding.detach().cpu().numpy().tolist(),
                'content': key,  # Use key as content for simplicity
                'access_count': 1,  # Default value
                'last_access': time.time(),
                'timestamp': time.time()
            }
        
        # Add semantic memory dictionary items
        for token, idx in self.semantic_memory.items():
            if isinstance(token, str):
                semantic_memories[f"token_{idx}"] = {
                    'content': token,
                    'index': idx,
                    'access_count': 1,  # Default value
                    'last_access': time.time(),
                    'timestamp': time.time()
                }
        
        return semantic_memories
    
    def _update_consolidated_memories(self, episodic: Dict[str, Any], semantic: Dict[str, Any]):
        """Update memory stores with consolidated memories"""
        # Update episodic embeddings
        for key, memory in episodic.items():
            if 'embedding' in memory:
                embedding = torch.tensor(memory['embedding'])
                self.episodic_embeddings[key] = embedding
        
        # Update semantic embeddings
        for key, memory in semantic.items():
            if 'embedding' in memory:
                embedding = torch.tensor(memory['embedding'])
                self.semantic_embeddings[key] = embedding
            elif 'content' in memory and 'index' in memory:
                self.semantic_memory[memory['content']] = memory['index']
    
    def retrieve_associative_memory(self, query: str) -> Dict[str, Any]:
        """
        Retrieve memories associatively using the continuous Hopfield network.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with retrieved memories
        """
        # Convert query to tensor
        query_tensor = torch.tensor(list(query), dtype=torch.float32).unsqueeze(0)
        
        # Retrieve from Hopfield network
        retrieved = self.continuous_hopfield.retrieve(query_tensor)
        
        # Convert retrieved tensor to string (simplified)
        retrieved_data = retrieved.detach().cpu().numpy().tolist()[0]
        
        # Find closest matching memories
        closest_memories = self._find_closest_memories(retrieved_data)
        
        return {
            'query': query,
            'retrieved_memories': closest_memories
        }
    
    def _find_closest_memories(self, retrieved_data: List[float]) -> List[Dict[str, Any]]:
        """
        Find closest matching memories to the retrieved data.
        
        Args:
            retrieved_data: Retrieved data vector
            
        Returns:
            List of closest memories
        """
        retrieved_tensor = torch.tensor(retrieved_data)
        closest_memories = []
        
        # Check episodic embeddings
        for key, embedding in self.episodic_embeddings.items():
            similarity = F.cosine_similarity(
                retrieved_tensor.unsqueeze(0),
                embedding.flatten().unsqueeze(0)
            ).item()
            
            if similarity > 0.7:  # Threshold for similarity
                closest_memories.append({
                    'key': key,
                    'type': 'episodic',
                    'similarity': similarity,
                    'path': self.ltm_summary.get(key, {}).get('path', '')
                })
        
        # Check semantic embeddings
        for key, embedding in self.semantic_embeddings.items():
            similarity = F.cosine_similarity(
                retrieved_tensor.unsqueeze(0),
                embedding.flatten().unsqueeze(0)
            ).item()
            
            if similarity > 0.7:  # Threshold for similarity
                closest_memories.append({
                    'key': key,
                    'type': 'semantic',
                    'similarity': similarity,
                    'content': key  # Use key as content for simplicity
                })
        
        # Sort by similarity (highest first)
        closest_memories.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top 5 memories
        return closest_memories[:5]
    
    def check_sleep_needed(self) -> bool:
        """
        Check if memory consolidation (sleep) is needed.
        
        Returns:
            bool: True if sleep is needed, False otherwise
        """
        # Check if it's been long enough since last sleep
        time_since_sleep = time.time() - self.last_sleep_time
        if time_since_sleep > self.sleep_interval:
            return True
            
        # Check if working memory is getting full
        if len(self.working_memory) >= self.working_memory_capacity * 0.9:
            return True
            
        # Check if there are many new memories since last consolidation
        new_memory_count = len(self.episodic_embeddings) + len(self.semantic_embeddings)
        if new_memory_count > 100:  # Arbitrary threshold
            return True
            
        return False
