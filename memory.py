import os
import json
import numpy as np
import random
import PyPDF2
import sqlite3
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import threading
import time
import pygame  # For rendering
import math

# ===========================
# Memory Components
# ===========================

class FadingMemorySystem(nn.Module):
    def __init__(self, hidden_dim: int, decay_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        self.memory_state = None
        
    def forward(self, x):
        if self.memory_state is None:
            self.memory_state = torch.zeros_like(x)
            
        # Apply exponential decay to existing memory
        self.memory_state = self.memory_state * math.exp(-self.decay_rate)
        
        # Update memory with new information
        self.memory_state = self.memory_state + x
        
        return self.memory_state
class SnapshotMemorySystem(nn.Module):
    def __init__(self, hidden_dim: int, memory_size: int = 8192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.current_index = 0
        
        # Initialize memory bank as a Parameter instead of a buffer
        # This ensures proper handling of gradients
        self.register_parameter(
            'memory_bank',
            nn.Parameter(torch.zeros(memory_size, hidden_dim), requires_grad=True)
        )
        
    def update(self, snapshot: torch.Tensor):
        """
        Update memory bank with new snapshots
        Args:
            snapshot: Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Get mean representation across sequence length
        snapshot = snapshot.mean(dim=1)  # (batch_size, hidden_dim)
        batch_size = snapshot.size(0)
        
        # Calculate indices for the batch
        indices = torch.arange(
            self.current_index,
            self.current_index + batch_size
        ) % self.memory_size
        
        # Use scatter_ instead of direct assignment
        # Create a temporary tensor of the same size as memory_bank
        new_memory = self.memory_bank.clone()
        new_memory.data[indices] = snapshot.detach().clone()
        
        # Update the parameter
        self.memory_bank = nn.Parameter(new_memory)
        
        # Update current index
        self.current_index = (self.current_index + batch_size) % self.memory_size
        
    def get_snapshots(self, num_snapshots: Optional[int] = None) -> torch.Tensor:
        """
        Retrieve snapshots from memory
        Args:
            num_snapshots: Number of most recent snapshots to retrieve
        Returns:
            Tensor of shape (num_snapshots, hidden_dim)
        """
        if num_snapshots is None:
            num_snapshots = self.memory_size
            
        num_snapshots = min(num_snapshots, self.memory_size)
        
        # Calculate indices of most recent snapshots
        indices = torch.arange(
            self.current_index - num_snapshots,
            self.current_index
        ) % self.memory_size
        
        return self.memory_bank[indices]
            
class SpatialMemoryModule:
    def __init__(self, room_size: Tuple[int, int], overlap: float = 0.2, max_memory: int = 100):
        """
        Initializes the spatial memory with parameters defining room size, overlap, and memory constraints.

        Args:
            room_size (tuple): Dimensions of each room (width, height).
            overlap (float): Fractional overlap between adjacent rooms.
            max_memory (int): Maximum number of rooms to retain in memory.
        """
        self.room_width, self.room_height = room_size
        self.overlap = overlap
        self.current_position = (0, 0)  # Starting at origin
        self.memory = {}  # Dictionary to store room data indexed by position
        self.max_memory = max_memory

    def get_current_room_key(self) -> Tuple[int, int]:
        """
        Determines the key for the current room based on Oni's position.

        Returns:
            tuple: Coordinates representing the current room.
        """
        x, y = self.current_position
        room_x = int(x // (self.room_width * (1 - self.overlap)))
        room_y = int(y // (self.room_height * (1 - self.overlap)))
        return (room_x, room_y)

    def update_position(self, new_position: Tuple[int, int]) -> bool:
        """
        Updates Oni's current position and determines if a new room needs to be loaded.

        Args:
            new_position (tuple): New (x, y) coordinates.

        Returns:
            bool: True if a new room is entered, False otherwise.
        """
        old_room = self.get_current_room_key()
        self.current_position = new_position
        new_room = self.get_current_room_key()
        if new_room != old_room:
            return True
        return False

    def load_room(self, room_key: Tuple[int, int], room_data: Dict):
        """
        Loads data for a new room into memory.

        Args:
            room_key (tuple): Coordinates representing the room.
            room_data (dict): Data associated with the room.
        """
        self.memory[room_key] = room_data
        # If memory exceeds max_memory, remove the least recently used room
        if len(self.memory) > self.max_memory:
            oldest_room = next(iter(self.memory))
            del self.memory[oldest_room]

    def get_current_room_data(self) -> Optional[Dict]:
        """
        Retrieves data for the current room.

        Returns:
            dict or None: Data of the current room or None if not loaded.
        """
        room_key = self.get_current_room_key()
        return self.memory.get(room_key, None)


class HeuristicManager:
    def __init__(self, heuristic_function, max_priority: int = 100):
        """
        Initializes the Heuristic Manager.

        Args:
            heuristic_function (callable): Function to compute priority based on room key.
            max_priority (int): Maximum number of rooms to prioritize.
        """
        self.heuristic_function = heuristic_function
        self.priority_queue = []
        self.max_priority = max_priority

    def add_room(self, room_key: Tuple[int, int]):
        priority = self.heuristic_function(room_key)
        heapq.heappush(self.priority_queue, (priority, room_key))
        # Ensure the queue doesn't exceed max_priority
        if len(self.priority_queue) > self.max_priority:
            heapq.heappop(self.priority_queue)

    def get_next_room(self) -> Optional[Tuple[int, int]]:
        if self.priority_queue:
            return heapq.heappop(self.priority_queue)[1]
        return None


class EpisodicEmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, compression_rate: float = 0.9):
        super(EpisodicEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compression_rate = compression_rate

        # Encoder for different media types (multi-modal)
        self.media_encoder = nn.ModuleDict({
            'text': nn.Linear(input_dim, output_dim),
            'image': nn.Linear(input_dim, output_dim),
            'audio': nn.Linear(input_dim, output_dim),
            # Add other media types as needed
        })

        # Chain compression mechanism
        self.compression_layer = nn.Linear(output_dim, int(output_dim * self.compression_rate))

        # Infinite space handler
        self.embeddings = []

    def forward(self, x, media_type: str):
        if media_type not in self.media_encoder:
            raise ValueError(f"Unsupported media type: {media_type}")

        # Encode input
        x = self.media_encoder[media_type](x)

        # Chain compression
        x = self.compression_layer(x)

        # Store embedding (infinite space handler)
        self.embeddings.append(x)

        return x


class SemanticMemoryLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(SemanticMemoryLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Text encoder
        self.text_encoder = nn.Linear(input_dim, output_dim)

        # Shared connection to episodic layer
        self.episodic_embedding_layer = None  # Set externally

    def forward(self, text_input: torch.Tensor, media_reference: Optional[torch.Tensor] = None):
        # Encode text
        text_embedding = self.text_encoder(text_input)

        # If media reference is provided, connect with episodic embedding
        if media_reference is not None and self.episodic_embedding_layer is not None:
            media_embedding = self.episodic_embedding_layer(media_reference, media_type='image')  # Example for image
            combined_embedding = torch.cat((text_embedding, media_embedding), dim=-1)
            return combined_embedding

        return text_embedding


class SparseHopfieldNetwork:
    def __init__(self, size: int, sparsity: float = 0.1):
        self.size = size
        self.sparsity = sparsity
        self.weights = np.zeros((size, size))
        self.create_sparse_connections()

    def create_sparse_connections(self):
        num_connections = int(self.size * self.size * self.sparsity)
        for _ in range(num_connections):
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            while i == j or self.weights[i, j] != 0:
                i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            self.weights[i, j] = 1
            self.weights[j, i] = 1  # Ensure symmetry

    def train(self, patterns: List[List[int]]):
        for p in patterns:
            p = np.array(p)
            outer_product = np.outer(p, p)
            self.weights += outer_product
        self.weights = self.weights / len(patterns)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern: List[int], steps: int = 10) -> np.ndarray:
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
        return pattern


class TextPatternFinder:
    def __init__(self, tokenizer, min_pattern_length: int = 3, max_pattern_length: int = 10, min_occurrences: int = 2):
        self.tokenizer = tokenizer
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_occurrences = min_occurrences
        self.corpus_patterns = defaultdict(list)
        self.ltm_patterns = defaultdict(list)
        self.hopfield_network = None

    def find_patterns(self, corpus: List[str], ltm: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        self._find_patterns_in_text(corpus, self.corpus_patterns)
        self._find_patterns_in_text(ltm, self.ltm_patterns)
        return self.corpus_patterns, self.ltm_patterns

    def _find_patterns_in_text(self, text: List[str], pattern_dict: Dict[str, List[int]]):
        tokens = self.tokenizer.tokenize(' '.join(text))
        num_tokens = len(tokens)

        for i in range(num_tokens):
            for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                if i + length > num_tokens:
                    break
                pattern = ' '.join(tokens[i:i+length])
                pattern_dict[pattern].append(i)

        # Remove patterns that do not meet the minimum occurrence threshold
        for pattern, occurrences in list(pattern_dict.items()):
            if len(occurrences) < self.min_occurrences:
                del pattern_dict[pattern]

    def consolidate_patterns(self) -> Dict[str, List[int]]:
        combined_patterns = {**self.corpus_patterns, **self.ltm_patterns}
        unique_patterns = {p: combined_patterns[p] for p in combined_patterns if p in self.corpus_patterns and p in self.ltm_patterns}
        return unique_patterns

    def use_hopfield_network(self, unique_patterns: Dict[str, List[int]]):
        pattern_vectors = [self._pattern_to_vector(p) for p in unique_patterns.keys()]
        self.hopfield_network = SparseHopfieldNetwork(size=len(unique_patterns))
        self.hopfield_network.train(pattern_vectors)

    def _pattern_to_vector(self, pattern: str) -> List[int]:
        pattern_tokens = self.tokenizer.tokenize(pattern)
        pattern_vector = [0] * self.tokenizer.vocab_size
        for token in pattern_tokens:
            index = self.tokenizer.token_to_id(token)
            if 0 <= index < self.tokenizer.vocab_size:
                pattern_vector[index] = 1
        return pattern_vector

    def update_hopfield_network(self, new_patterns: Dict[str, List[int]]):
        new_pattern_vectors = [self._pattern_to_vector(p) for p in new_patterns.keys()]
        self.hopfield_network.train(new_pattern_vectors)


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
        self.semantic_layer = SemanticMemoryLayer(input_dim=1024, output_dim=1024)
        self.episodic_layer.to(device)  # Adjust device as needed
        self.semantic_layer.to(device)  # Adjust device as needed
   
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
        self.ltm_summary[key] = {'data_type': data_type, 'path': file_path}

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
        self.meditate()
        self.save_long_term_memory()
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
        self.working_memory.append(stm_data)
        if len(self.working_memory) > self.working_memory_capacity:
            self.working_memory.pop(0)  # Remove oldest entry to maintain capacity
        if episodic_data and episodic_key:
            self.update_episodic_memory(episodic_data, data_type='audio', key=episodic_key)
        if semantic_data and episodic_key:
            self.update_semantic_memory(semantic_data, data_type='text', key=episodic_key)
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
