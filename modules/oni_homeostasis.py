import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.oni_Kan import KanSpikingNeuron
from modules.oni_Dyn import DynamicSynapse 

class HomeostaticController(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HomeostaticController, self).__init__()

        # Spiking neuron layer
        self.kan = KanSpikingNeuron(input_dim)

        # Hebbian-based dynamic synapse adjustment
        self.dynamic_synapse = DynamicSynapse(input_dim, hidden_dim)

        # Predictive processing mechanism
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Feedback memory
        self.history = nn.Parameter(torch.zeros(hidden_dim))

        # Advanced monitoring systems
        self.anomaly_detector = AnomalyDetectionModule(hidden_dim)
        self.stability_regulator = StabilityRegulationModule(hidden_dim)
        self.resource_allocator = AdaptiveScheduler(hidden_dim)

    def forward(self, x, system_state):
        """Processes the input, regulates internal states, and adapts dynamically."""
        feedback_signal = self.calculate_feedback(system_state + self.history)

        # Step 1: Spiking Neuron Processing
        x = self.kan(x)

        # Step 2: Predictive Processing with RNN
        x, _ = self.rnn(x.unsqueeze(0))  # Ensure batch dimension
        x = x.squeeze(0)  # Remove batch dimension after processing

        # Step 3: Synaptic Adaptation
        adjusted_output = self.dynamic_synapse(x + feedback_signal)

        # Step 4: Anomaly Detection & Stability Regulation
        if self.anomaly_detector.detect(x):
            adjusted_output = self.stability_regulator.apply(adjusted_output)

        # Step 5: Resource Allocation
        adjusted_output = self.resource_allocator.allocate(adjusted_output)

        # Update history using a non-linear projection (tanh ensures stable memory updates)
        self.history = torch.tanh(self.history + adjusted_output)

        return adjusted_output, feedback_signal

    def calculate_feedback(self, system_state):
        """Computes feedback for self-regulation based on system state deviations."""
        return torch.sin(system_state) * 0.1  # Nonlinear modulation instead of simple random noise

# --- Supportive Mechanisms ---

class AnomalyDetectionModule(nn.Module):
    """Detects anomalies using an autoencoder approach to measure deviation from normal states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

    def detect(self, x):
        """Returns True if input deviates significantly from its expected reconstruction."""
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        loss = torch.mean((x - reconstructed) ** 2)  # Reconstruction error as anomaly score
        return loss > 0.05  # Threshold-based anomaly detection

class StabilityRegulationModule(nn.Module):
    """Applies stability adjustments using Hebbian Learning-inspired weight updates."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

    def apply(self, x):
        """Uses a Hebbian learning-inspired rule to adjust stability."""
        hebbian_adjustment = torch.matmul(x, self.weights)
        return x + 0.1 * torch.tanh(hebbian_adjustment)  # Regulates erratic fluctuations

class AdaptiveScheduler(nn.Module):
    """Dynamically allocates resources based on system load and demand."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.priority_scale = nn.Parameter(torch.ones(hidden_dim))  # Adaptive scaling factors

    def allocate(self, x):
        """Dynamically adjusts processing intensity based on importance."""
        scaled_x = x * self.priority_scale
        return F.normalize(scaled_x, p=2, dim=0)  # Ensures energy-efficient operation


