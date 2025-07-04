#!/usr/bin/env python3
"""
ONI-MindLink Enhanced BCI Interface v3.0
Integrates real-time neural network brain intent decoding with PyAutoGUI controller actions.

Features:
- Bidirectional brain-computer interface with advanced signal processing.
- Real-time neural intent classification via a PyTorch model (EEGIntentNet).
- Action execution through Oni's modular PyAutoGUI-based controller.
- Comprehensive brain data storage and analytics via SQLite.
- Online training support for adaptive, real-time model improvement.
- Modular architecture connecting mental states, signals, and actions.

Author: ONI-MindLink Enhanced Team
License: Pantheum License (matching ONI)
"""

import numpy as np
import sqlite3
import json
import time
import asyncio
import threading
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

# Assuming the controller is in the specified path within the project structure
from ControllerPyAutoGui import KeyboardMouseController

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- BCI Data Structures and Enums ---
class BrainWaveType(Enum):
    DELTA = (0.5, 4.0); THETA = (4.0, 8.0); ALPHA = (8.0, 13.0); BETA = (13.0, 30.0); GAMMA = (30.0, 100.0); HIGH_GAMMA = (100.0, 200.0)

class MentalState(Enum):
    FOCUSED = "focused"; RELAXED = "relaxed"; MEDITATIVE = "meditative"; CREATIVE = "creative"; STRESSED = "stressed"; ALERT = "alert"; NEUTRAL = "neutral"; SEARCHING = "searching"; VISUALIZING = "visualizing"; COMMAND_INTENT = "command_intent"

class BrainSignalType(Enum):
    INTENTION = "intention"; EMOTION = "emotion"; MEMORY = "memory"; ATTENTION = "attention"; MOTOR_IMAGERY = "motor_imagery"; COMMAND = "command"

@dataclass
class EnhancedEEGReading:
    timestamp: float; session_id: str; mental_state: MentalState; confidence: float;
    spectral_features: Dict[str, float]; temporal_features: Dict[str, float]

@dataclass
class BrainCommand:
    command_id: str; command_type: str; parameters: Dict[str, Any]; confidence: float; timestamp: float; mental_context: MentalState

# --- Neural Network and Control Classes (from nnsforbci.txt) ---
class EEGIntentNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_intents: int):
        super(EEGIntentNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_intents))

    def forward(self, x):
        return self.model(x)

class RealTimeIntentController:
    """Connects neural intent predictions to actual computer actions."""
    def __init__(self, model: EEGIntentNet, controller: KeyboardMouseController, intent_to_action_map: Dict[str, str]):
        self.model = model
        self.controller = controller
        self.intent_to_action_map = intent_to_action_map
        self.softmax = nn.Softmax(dim=1)
        self.model.eval() # Set model to evaluation mode

    def predict_intent(self, features: Dict[str, float]) -> Tuple[str, float]:
        if not features: return "none", 0.0
        x = torch.tensor([list(features.values())], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
        probs = self.softmax(logits)
        confidence, intent_idx = torch.max(probs, dim=1)
        intent = list(self.intent_to_action_map.keys())[intent_idx.item()]
        return intent, confidence.item()

    def act_on_intent(self, intent: str, confidence: float, threshold: float = 0.7):
        if confidence < threshold: return
        action = self.intent_to_action_map.get(intent)
        if action:
            logger.info(f"Executing action for intent '{intent}' with confidence {confidence:.2f}")
            self.controller.execute_sequence(action)

class OnlineTrainer:
    def __init__(self, model: EEGIntentNet, lr: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, features: List[float], label_idx: int):
        self.model.train() # Set model to training mode
        x = torch.tensor([features], dtype=torch.float32)
        y = torch.tensor([label_idx], dtype=torch.long)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval() # Return to evaluation mode
        return loss.item()

# --- Data Storage and Management ---
class BrainDataStorage:
    """Manages the SQLite database for storing EEG data and commands."""
    def __init__(self, db_path: str = "brain_data_v3.db"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS eeg_readings (
                            id INTEGER PRIMARY KEY, timestamp REAL, session_id TEXT, mental_state TEXT,
                            confidence REAL, features TEXT)''')
            conn.commit()

    def store_eeg_reading(self, reading: EnhancedEEGReading, features: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO eeg_readings (timestamp, session_id, mental_state, confidence, features) VALUES (?, ?, ?, ?, ?)",
                         (reading.timestamp, reading.session_id, reading.mental_state.value, reading.confidence, json.dumps(features)))
            conn.commit()

# --- Core BCI Interface ---
class EnhancedMindLinkInterface:
    def __init__(self, storage: BrainDataStorage):
        self.storage = storage
        self.is_streaming = False
        self.current_session_id = None
        self.intent_controller: Optional[RealTimeIntentController] = None
        self.last_action_time = 0
        # Simulation parameters
        self.sampling_rate = 256
        self.channels = 8
        self.signal_buffer = np.zeros((self.channels, self.sampling_rate * 2))

    def connect(self) -> bool:
        self.current_session_id = str(uuid.uuid4())
        logger.info(f"Connected to Enhanced Mind-link (Session: {self.current_session_id})")
        return True

    def set_intent_controller(self, intent_controller: RealTimeIntentController):
        self.intent_controller = intent_controller

    def start_streaming(self):
        if self.is_streaming: return
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._enhanced_streaming_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Started enhanced EEG streaming.")

    def stop_streaming(self):
        self.is_streaming = False
        if hasattr(self, 'stream_thread'): self.stream_thread.join()
        logger.info("Stopped enhanced EEG streaming.")

    def _enhanced_streaming_loop(self):
        while self.is_streaming:
            try:
                # Simulate receiving a new chunk of data
                raw_chunk = np.random.randn(self.channels, 20)
                self.signal_buffer = np.roll(self.signal_buffer, -raw_chunk.shape[1], axis=1)
                self.signal_buffer[:, -raw_chunk.shape[1]:] = raw_chunk

                eeg_reading = self._process_enhanced_signals(self.signal_buffer)

                if eeg_reading and self.intent_controller:
                    features = {**eeg_reading.spectral_features, **eeg_reading.temporal_features}
                    self.storage.store_eeg_reading(eeg_reading, features)

                    intent, confidence = self.intent_controller.predict_intent(features)
                    
                    # Throttle actions to avoid spamming the OS
                    if confidence > 0.75 and time.time() - self.last_action_time > 1.5:
                        self.intent_controller.act_on_intent(intent, confidence)
                        self.last_action_time = time.time()

                time.sleep(0.1)  # 10Hz inference rate
            except Exception as e:
                logger.error(f"Critical streaming error: {e}", exc_info=True)
                self.is_streaming = False
                break
    
    def _process_enhanced_signals(self, raw_data: np.ndarray) -> Optional[EnhancedEEGReading]:
        spectral_features = self._extract_spectral_features(raw_data)
        temporal_features = self._extract_temporal_features(raw_data)
        
        # Simple rule-based mental state classification for demo
        if spectral_features.get('beta_power', 0) > 0.4:
            mental_state, confidence = MentalState.FOCUSED, spectral_features['beta_power']
        elif spectral_features.get('alpha_power', 0) > 0.4:
            mental_state, confidence = MentalState.RELAXED, spectral_features['alpha_power']
        else:
            mental_state, confidence = MentalState.NEUTRAL, 0.5
        
        return EnhancedEEGReading(
            timestamp=time.time(), session_id=self.current_session_id,
            mental_state=mental_state, confidence=confidence,
            spectral_features=spectral_features, temporal_features=temporal_features
        )

    def _extract_spectral_features(self, data: np.ndarray) -> Dict[str, float]:
        features = {}
        nyquist = 0.5 * self.sampling_rate
        fft_vals = np.abs(fft(data))
        fft_freq = fftfreq(data.shape[1], 1/self.sampling_rate)
        
        for wave in BrainWaveType:
            freq_ix = np.where((fft_freq >= wave.value[0]) & (fft_freq <= wave.value[1]))[0]
            power = np.mean(fft_vals[:, freq_ix]) if freq_ix.size > 0 else 0
            features[f'{wave.name.lower()}_power'] = power
            
        total_power = sum(features.values())
        if total_power > 0:
            for key in features: features[key] /= total_power
        return features

    def _extract_temporal_features(self, data: np.ndarray) -> Dict[str, float]:
        # Calculate simple, fast temporal features for real-time use
        return {
            'mean_abs_val': float(np.mean(np.abs(data))),
            'std_dev': float(np.std(data)),
            'zero_crossings': float(np.sum(np.diff(np.sign(data)) != 0, axis=1).mean()),
            'variance': float(np.var(data))
        }

# --- Main Application Orchestrator ---
def main():
    logger.info("Initializing ONI-MindLink BCI v3.0...")

    # 1. Define Intents and Actions
    intent_to_action_map = {
        "intent_copy": "copy_text", "intent_paste": "paste_text",
        "intent_new_tab": "open_new_tab", "intent_save": "save_file",
        "intent_forward_start": "game_move_forward", "intent_forward_stop": "game_stop_forward",
        "intent_jump": "game_jump", "intent_sprint_start": "sprint_start"
    }
    num_intents = len(intent_to_action_map)
    # The number of input features must match the combined length of spectral and temporal feature dicts
    # BrainWaveType (6) + temporal features (4) = 10
    input_size = len(BrainWaveType) + 4 

    # 2. Instantiate all components
    controller = KeyboardMouseController()
    storage = BrainDataStorage()
    model = EEGIntentNet(input_size=input_size, hidden_size=64, num_intents=num_intents)
    intent_controller = RealTimeIntentController(model, controller, intent_to_action_map)
    mindlink_interface = EnhancedMindLinkInterface(storage)

    # 3. Link the MindLink interface to the intent controller
    mindlink_interface.set_intent_controller(intent_controller)
    
    # 4. Start the BCI
    if not mindlink_interface.connect():
        logger.error("Could not connect to MindLink device. Exiting.")
        return

    try:
        mindlink_interface.start_streaming()
        print("\n" + "="*60)
        print("ONI-MINDLINK BCI V3.0 IS NOW ACTIVE")
        print("Listening for brain intents... Press Ctrl+C to shut down.")
        print("="*60)
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        mindlink_interface.stop_streaming()
        logger.info("Application has been shut down gracefully.")

if __name__ == "__main__":
    main()
