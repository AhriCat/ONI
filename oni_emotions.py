import torch
import torch.nn as nn
from transformers import pipeline
import numpy as np
from collections import deque
device = torch.device('cuda')
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device = 'cuda')


class EmotionalState:
    def __init__(self):
        self.emotion_memory = deque(maxlen=5)
        self.current_emotion = None
        self.emotion_intensity = 0.0
        self.emotion_decay = 0.95
        
    def update_state(self, new_emotion, intensity=1.0):
        self.emotion_memory.append(self.current_emotion)
        self.current_emotion = new_emotion
        self.emotion_intensity = intensity
        
    def decay_emotion(self):
        self.emotion_intensity *= self.emotion_decay

class EmotionalLayer(nn.Module):
    # Class-level emotion mapping
    EMOTION_VA_MAP = {
        'admiration': (0.6, 0.3),
        'amusement': (0.8, 0.4),
        'anger': (-0.6, 0.8),
        'annoyance': (-0.4, 0.4),
        'approval': (0.4, 0.2),
        'caring': (0.6, 0.3),
        'confusion': (-0.2, 0.3),
        'curiosity': (0.3, 0.4),
        'desire': (0.5, 0.6),
        'disappointment': (-0.5, -0.2),
        'disapproval': (-0.4, 0.3),
        'disgust': (-0.7, 0.4),
        'embarrassment': (-0.4, 0.4),
        'excitement': (0.7, 0.8),
        'fear': (-0.7, 0.7),
        'gratitude': (0.7, 0.3),
        'grief': (-0.8, -0.4),
        'joy': (0.8, 0.6),
        'love': (0.9, 0.5),
        'nervousness': (-0.3, 0.7),
        'optimism': (0.6, 0.4),
        'pride': (0.7, 0.5),
        'realization': (0.2, 0.3),
        'relief': (0.4, -0.2),
        'remorse': (-0.6, -0.3),
        'sadness': (-0.7, -0.3),
        'surprise': (0.3, 0.7),
        'neutral': (0.0, 0.0)
    }

    def __init__(self, hidden_dim):
        super(EmotionalLayer, self).__init__()
        self.valence = nn.Parameter(torch.tensor(0.0))
        self.arousal = nn.Parameter(torch.tensor(0.0))
        self.emotional_state = EmotionalState()
        self.emotion_embedding = nn.Embedding(len(self.EMOTION_VA_MAP), hidden_dim)
        self.transition_matrix = nn.Parameter(torch.randn(len(self.EMOTION_VA_MAP), len(self.EMOTION_VA_MAP)))
        self.device = device
        
    def map_emotion_to_valence_arousal(self, emotion_label):
        return self.EMOTION_VA_MAP.get(emotion_label, (0.0, 0.0))

    def compute_emotion_influence(self, x, emotion_dict):
        # Convert emotion dictionary to tensor
        emotion_vector = torch.zeros(len(self.EMOTION_VA_MAP))
        
        # Fill the emotion vector with probabilities
        for label, score in emotion_dict.items():
            if label in self.EMOTION_VA_MAP:
                idx = list(self.EMOTION_VA_MAP.keys()).index(label)
                emotion_vector[idx] = score
                
        # Normalize probabilities
        emotion_vector = torch.softmax(emotion_vector, dim=0)
        
        # Compute next state
        transition_weights = torch.softmax(self.transition_matrix, dim=1)
        next_state_probs = torch.matmul(emotion_vector, transition_weights)
        
        # Update emotional state
        max_emotion_idx = torch.argmax(next_state_probs)
        self.emotional_state.update_state(max_emotion_idx.item(), next_state_probs[max_emotion_idx].item())
        
        return next_state_probs

    def forward(self, x, emotion_input=None):
        if emotion_input is not None:
            # Process emotional input from classifier
            emotion_dict = {}
            for item in emotion_input:
                emotion_dict[item['label']] = item['score']
            
            emotional_influence = self.compute_emotion_influence(x, emotion_dict)
            
            # Find dominant emotion
            dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
            v, a = self.map_emotion_to_valence_arousal(dominant_emotion)
            
            # Update valence and arousal
            self.valence.data = torch.tensor(v, dtype=self.valence.dtype)
            self.arousal.data = torch.tensor(a, dtype=self.arousal.dtype)
        
        # Apply emotional modulation
        arousal_gain = torch.sigmoid(self.arousal)
        emotional_modulation = 1.0 + (self.valence * arousal_gain)
        
        # Apply emotional state influence
        if self.emotional_state.current_emotion is not None:
            emotion_embedding = self.emotion_embedding(torch.tensor(self.emotional_state.current_emotion))
            emotional_context = emotion_embedding * self.emotional_state.emotion_intensity
            x = x + emotional_context
        
        return x * emotional_modulation

class EnergyModule(nn.Module):
    def __init__(self, init_energy=100):
        super(EnergyModule, self).__init__()
        self.energy = nn.Parameter(torch.tensor(init_energy, dtype=torch.float32), requires_grad=False)
        self.max_energy = init_energy
        self.recovery_rate = 0.1
        
    def forward(self, x, arousal, valence, emotional_state):
        energy_loss = torch.norm(x).item() * 0.01 * (1 + emotional_state.emotion_intensity)
        new_energy = self.energy - energy_loss

        if valence > 0:
            recovery = self.recovery_rate * (1 + valence) * (self.max_energy - new_energy)
            new_energy = min(new_energy + recovery, self.max_energy)
        
        fatigue_threshold = 0.2 * self.max_energy
        if new_energy < fatigue_threshold:
            fatigue_factor = (new_energy / fatigue_threshold)
            x = x * fatigue_factor
            
        self.energy.data = new_energy.clone().detach()
        return x
    
class EmotionalFeedbackModule(nn.Module):
    def __init__(self, base_learning_rate=0.001, emotion_sensitivity=0.5):
        super(EmotionalFeedbackModule, self).__init__()
        self.base_lr = base_learning_rate
        self.emotion_sensitivity = emotion_sensitivity
        self.feedback_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available else 'cpu')

    def compute_feedback_weight(self, emotional_state, model = None):
        """
        Compute weight adjustment based on emotional valence and intensity
        Positive emotions -> positive feedback (reward)
        Negative emotions -> negative feedback (punishment)
        """
        valence = emotional_state['valence']
        intensity = emotional_state['intensity']
        
        model = None 
        
        feedback = valence * intensity * self.emotion_sensitivity
        feedback_tensor = tokenizer.encode(str(feedback))  # Convert to tensor
        feedback_weight = torch.tanh(feedback_tensor * self.base_lr)
        if model is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)
        else:
            return feedback
        def feedback_hook(optimizer):
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.data += feedback_weight * param.grad
                    
        optimizer.add_hook(feedback_hook)
        return feedback_weight
    
    def apply_emotional_feedback(self, module, emotional_state):
        """
        Apply emotional feedback to module parameters
        """
        feedback_weight = self.compute_feedback_weight(emotional_state)
        
        # Store feedback for analysis
        self.feedback_history.append({
            'emotion': emotional_state['current_emotion'],
            'feedback_weight': feedback_weight,
            'intensity': emotional_state['intensity']
        })
        
        # Apply feedback to parameters
        with torch.no_grad():
            for param in module.parameters():
                if param.requires_grad:
                    # Positive feedback reinforces current weights
                    # Negative feedback pushes weights toward zero
                    if feedback_weight > 0:
                        param.data += feedback_weight * param.data
                    else:
                        param.data += feedback_weight * (param.data - param.data.mean())
                        
    def get_feedback_stats(self):
        """
        Return statistics about feedback history
        """
        if not self.feedback_history:
            return None
            
        recent_feedback = self.feedback_history[-10:]
        return {
            'average_feedback': sum(f['feedback_weight'] for f in recent_feedback) / len(recent_feedback),
            'strongest_positive': max((f for f in recent_feedback if f['feedback_weight'] > 0), 
                                   key=lambda x: x['feedback_weight'], default=None),
            'strongest_negative': min((f for f in recent_feedback if f['feedback_weight'] < 0), 
                                   key=lambda x: x['feedback_weight'], default=None)
        }

class EmotionalEnergyModel(nn.Module):
    def __init__(self, hidden_dim, init_energy=100):
        super(EmotionalEnergyModel, self).__init__()
        self.emotion_layer = EmotionalLayer(hidden_dim)
        self.energy_module = EnergyModule(init_energy)
        self.classifier = classifier
        self.feedback_module = EmotionalFeedbackModule()
        
    def get_emotional_state(self):
        current_emotion_idx = self.emotion_layer.emotional_state.current_emotion
        if current_emotion_idx is not None:
            emotion_list = list(self.emotion_layer.EMOTION_VA_MAP.keys())
            current_emotion = emotion_list[current_emotion_idx] if current_emotion_idx < len(emotion_list) else "unknown"
        else:
            current_emotion = "none"
            
        return {
            'current_emotion': current_emotion,
            'intensity': self.emotion_layer.emotional_state.emotion_intensity,
            'valence': self.emotion_layer.valence.item(),
            'arousal': self.emotion_layer.arousal.item(),
            'energy': self.energy_module.energy.item()
        }
        
    def forward(self, x):
        if isinstance(x, str):
            # Process text input
            model_outputs = self.classifier([x])[0]  # Get first item since we're processing single string
            processed_tensor = torch.randn(1, 896)  # Placeholder tensor
            output = self.emotion_layer(processed_tensor, model_outputs)
            emotional_state = self.get_emotional_state()
            self.feedback_module.apply_emotional_feedback(self.emotion_layer, emotional_state)
            return {
                'classification': model_outputs,
                'emotional_state': self.get_emotional_state(),
                'feedback_stats': self.feedback_module.get_feedback_stats()
            }
            
        else:
            # Process tensor input
            x = self.emotion_layer(x)
            x = self.energy_module(
                x, 
                self.emotion_layer.arousal.item(),
                self.emotion_layer.valence.item(),
                self.emotion_layer.emotional_state
            )
            return x

# Example usage
emoMod = EmotionalEnergyModel(hidden_dim=896)

# Test with tensor input
input_tensor = torch.randn(1, 10)
output_tensor = emoMod(input_tensor)

# Test with text input
output_text = emoMod("I love you, you are so awesome! I'm so proud of you")

print(f"Tensor output: {output_tensor}")
print(f"Text analysis: {output_text}")
print(f"Current emotional state: {emoMod.get_emotional_state()}")
