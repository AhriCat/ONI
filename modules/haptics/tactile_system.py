import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

class TouchType(Enum):
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    TEXTURE = "texture"
    PAIN = "pain"

class EmotionCategory(Enum):
    COMFORT = "comfort"
    PAIN = "pain"
    PLEASURE = "pleasure"
    NEUTRAL = "neutral"
    ALERT = "alert"

@dataclass
class TactileSignal:
    """Enhanced tactile signal with validation and utility methods"""
    pressure: float  # 0-1 normalized
    temperature: float  # Celsius
    vibration: float  # Hz
    location: Tuple[float, float]  # (x, y) body coordinates
    duration: float  # seconds
    context: Dict[str, Union[str, float]]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        self._validate()
    
    def _validate(self):
        """Validate signal parameters"""
        assert 0 <= self.pressure <= 1, "Pressure must be normalized 0-1"
        assert -50 <= self.temperature <= 100, "Temperature out of realistic range"
        assert 0 <= self.vibration <= 1000, "Vibration frequency out of range"
        assert len(self.location) == 2, "Location must be (x, y) coordinates"
    
    def to_tensor(self) -> torch.Tensor:
        """Convert signal to tensor for neural network processing"""
        return torch.tensor([
            self.pressure,
            self.temperature / 100.0,  # normalize
            self.vibration / 1000.0,   # normalize
            self.location[0],
            self.location[1],
            self.duration
        ], dtype=torch.float32)
    
    def get_intensity(self) -> float:
        """Calculate overall signal intensity"""
        return (self.pressure + self.vibration/1000.0) / 2.0

class TouchEncoder(nn.Module):
    """CNN-based encoder for tactile patterns"""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class TouchEmotionMapper(nn.Module):
    """Maps tactile signals to emotional states using valence-arousal model"""
    def __init__(self, signal_dim: int = 6, context_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.encoder = TouchEncoder(signal_dim, hidden_dim, 64)
        self.context_encoder = nn.LSTM(context_dim, 32, batch_first=True)
        self.arousal_valence_predictor = nn.Sequential(
            nn.Linear(64 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # valence, arousal
        )
        
    def forward(self, tactile_seq: torch.Tensor, context_seq: torch.Tensor) -> torch.Tensor:
        # Encode tactile signals
        batch_size, seq_len = tactile_seq.shape[:2]
        tactile_flat = tactile_seq.reshape(-1, tactile_seq.shape[-1])
        emb = self.encoder(tactile_flat)
        emb = emb.reshape(batch_size, seq_len, -1)
        emb = torch.mean(emb, dim=1)  # Average over sequence
        
        # Encode context
        ctx_out, _ = self.context_encoder(context_seq)
        ctx = ctx_out[:, -1, :]  # Use last hidden state
        
        # Predict emotion
        combined = torch.cat([emb, ctx], dim=-1)
        emotion = self.arousal_valence_predictor(combined)
        
        # Apply sigmoid to constrain to [-1, 1] range
        emotion = torch.tanh(emotion)
        return emotion  # [valence, arousal]

class ReflexArc:
    """Immediate response system for tactile signals"""
    def __init__(self, pain_thresholds: Dict[str, float] = None):
        self.thresholds = pain_thresholds or {
            'severe': 0.8,
            'moderate': 0.5,
            'mild': 0.2
        }
        self.response_history = []
    
    def evaluate(self, signal: TactileSignal) -> str:
        """Evaluate signal and return immediate response"""
        response_time = time.time()
        
        # Check for emergency conditions
        if signal.pressure > self.thresholds['severe']:
            response = "emergency_stop"
        elif signal.pressure > self.thresholds['moderate']:
            response = "withdraw"
        elif signal.temperature > 60 or signal.temperature < -10:
            response = "temperature_alert"
        elif signal.vibration > 500:
            response = "vibration_alert"
        else:
            response = "log_touch"
        
        # Log response
        self.response_history.append({
            'signal': signal,
            'response': response,
            'reaction_time': response_time - signal.timestamp
        })
        
        return response
    
    def get_response_stats(self) -> Dict:
        """Get statistics about response patterns"""
        if not self.response_history:
            return {}
        
        responses = [r['response'] for r in self.response_history]
        reaction_times = [r['reaction_time'] for r in self.response_history]
        
        return {
            'total_responses': len(responses),
            'response_types': {r: responses.count(r) for r in set(responses)},
            'avg_reaction_time': np.mean(reaction_times),
            'max_reaction_time': np.max(reaction_times)
        }

class TouchTranslator(nn.Module):
    """Cross-domain touch translation between robot, VR, and human domains"""
    def __init__(self, signal_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        
        # Domain-specific encoders
        self.robot_encoder = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        self.vr_encoder = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        self.human_encoder = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        
        # Shared representation transformer
        self.shared_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
            num_layers=2
        )
        
        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, signal_dim),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
    
    def forward(self, signal: torch.Tensor, source_domain: str) -> torch.Tensor:
        """Translate signal from source domain to universal representation"""
        if source_domain == "robot":
            x = self.robot_encoder(signal)
        elif source_domain == "vr":
            x = self.vr_encoder(signal)
        elif source_domain == "human":
            x = self.human_encoder(signal)
        else:
            raise ValueError(f"Unknown domain: {source_domain}")
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply transformer
        shared = self.shared_transformer(x)
        
        # Decode to universal format
        output = self.decoder(shared.squeeze(1))
        return output

class MultimodalPerceptualSystem:
    """Integrated system combining vision, audio, and touch"""
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.vision_model = self._create_vision_backbone()
        self.audio_model = self._create_audio_backbone()
        self.touch_model = TouchEncoder()
        self.fusion_layer = nn.Linear(192, 128)  # 64*3 -> 128
        
    def _create_vision_backbone(self) -> nn.Module:
        """Create vision processing backbone"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 64, 64)
        )
    
    def _create_audio_backbone(self) -> nn.Module:
        """Create audio processing backbone"""
        return nn.Sequential(
            nn.Linear(1024, 256),  # Assuming 1024 audio features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64)
        )
    
    def encode_inputs(self, image: torch.Tensor, sound: torch.Tensor, 
                     touch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode multimodal inputs"""
        v = self.vision_model(image)
        a = self.audio_model(sound)
        t = self.touch_model(touch)
        return v, a, t
    
    def fuse_modalities(self, v: torch.Tensor, a: torch.Tensor, 
                       t: torch.Tensor) -> torch.Tensor:
        """Fuse multimodal representations"""
        combined = torch.cat([v, a, t], dim=-1)
        return self.fusion_layer(combined)

class EmotionalLatentLearner(nn.Module):
    """Learn emotional representations from multimodal input"""
    def __init__(self, input_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.latent_projector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        
        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(EmotionCategory))
        )
    
    def forward(self, multimodal_features: torch.Tensor, 
                physiological_state: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional physiological state"""
        if physiological_state is not None:
            features = torch.cat([multimodal_features, physiological_state], dim=-1)
        else:
            features = multimodal_features
        
        # Project to latent space
        latent = self.latent_projector(features)
        
        # Generate outputs
        contrastive_features = self.contrastive_head(latent)
        emotion_logits = self.emotion_classifier(latent)
        
        return {
            'latent': latent,
            'contrastive': contrastive_features,
            'emotion_logits': emotion_logits,
            'emotion_probs': F.softmax(emotion_logits, dim=-1)
        }

class TactilePerceptionSystem:
    """Complete tactile perception system integrating all components"""
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.emotion_mapper = TouchEmotionMapper()
        self.reflex_arc = ReflexArc()
        self.touch_translator = TouchTranslator()
        self.multimodal_system = MultimodalPerceptualSystem(device)
        self.emotional_learner = EmotionalLatentLearner()
        
    def process_tactile_signal(self, signal: TactileSignal) -> Dict:
        """Process a single tactile signal through the complete pipeline"""
        results = {}
        
        # Immediate reflex response
        reflex_response = self.reflex_arc.evaluate(signal)
        results['reflex_response'] = reflex_response
        
        # Convert to tensor for neural processing
        signal_tensor = signal.to_tensor().unsqueeze(0)
        
        # Emotion mapping (requires context sequence - using dummy for single signal)
        dummy_context = torch.zeros(1, 5, 10)  # batch_size=1, seq_len=5, context_dim=10
        emotion_coords = self.emotion_mapper(signal_tensor.unsqueeze(1), dummy_context)
        results['emotion'] = {
            'valence': emotion_coords[0, 0].item(),
            'arousal': emotion_coords[0, 1].item()
        }
        
        # Domain translation
        translated = self.touch_translator(signal_tensor, 'human')
        results['translated_signal'] = translated
        
        return results
    
    def process_multimodal_input(self, image: torch.Tensor, sound: torch.Tensor, 
                                touch: torch.Tensor) -> Dict:
        """Process multimodal input including tactile information"""
        # Encode each modality
        v, a, t = self.multimodal_system.encode_inputs(image, sound, touch)
        
        # Fuse modalities
        fused = self.multimodal_system.fuse_modalities(v, a, t)
        
        # Learn emotional representation
        emotional_output = self.emotional_learner(fused)
        
        return {
            'visual_features': v,
            'audio_features': a,
            'tactile_features': t,
            'fused_features': fused,
            'emotional_output': emotional_output
        }

# Example usage and testing
def demo_tactile_system():
    """Demonstrate the tactile perception system"""
    print("Initializing Tactile Perception System...")
    system = TactilePerceptionSystem()
    
    # Create sample tactile signals
    signals = [
        TactileSignal(0.3, 25.0, 10.0, (0.5, 0.3), 0.1, {'mood': 'calm'}),
        TactileSignal(0.9, 45.0, 100.0, (0.2, 0.8), 0.5, {'mood': 'alert'}),
        TactileSignal(0.1, 20.0, 5.0, (0.7, 0.1), 0.05, {'mood': 'relaxed'})
    ]
    
    print("\nProcessing tactile signals...")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Pressure: {signal.pressure}, Temp: {signal.temperature}°C")
        print(f"  Location: {signal.location}, Duration: {signal.duration}s")
        
        results = system.process_tactile_signal(signal)
        print(f"  Reflex Response: {results['reflex_response']}")
        print(f"  Emotion - Valence: {results['emotion']['valence']:.3f}, "
              f"Arousal: {results['emotion']['arousal']:.3f}")
    
    # Show reflex arc statistics
    print(f"\nReflex Arc Statistics:")
    stats = system.reflex_arc.get_response_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    demo_tactile_system()


