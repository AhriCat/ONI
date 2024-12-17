import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaCognitionModule(nn.Module):
    def __init__(self, hidden_dim):
        """
        MetaCognitionModule with contextual, nuanced conflict reasoning and interaction graphs.

        Args:
            hidden_dim (int): Dimensionality of the hidden input tensor.
        """
        super(MetaCognitionModule, self).__init__()
        
        # Reflection, confidence, and normalization layers
        self.self_reflection = nn.Linear(hidden_dim, hidden_dim)
        self.confidence_estimation = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dynamic storage for principles
        self.principles = nn.ParameterList()

        # Contextual projection layer for principles
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)

        # Alignment layer
        self.adaptive_alignment = nn.Linear(hidden_dim, hidden_dim)

    def add_principle(self, principle_vector):
        """
        Dynamically add a new principle to the module.

        Args:
            principle_vector (torch.Tensor): A tensor of shape (hidden_dim,).
        """
        if principle_vector.dim() != 1 or principle_vector.size(0) != self.self_reflection.in_features:
            raise ValueError("Principle vector must be of shape (hidden_dim,).")
        self.principles.append(nn.Parameter(principle_vector.clone(), requires_grad=True))

    def contextual_conflict_score(self, principle_a, principle_b, context):
        """
        Compute a nuanced conflict score between two principles in the given context.

        Args:
            principle_a (torch.Tensor): Tensor representing principle A.
            principle_b (torch.Tensor): Tensor representing principle B.
            context (torch.Tensor): The input context tensor.

        Returns:
            score (float): Conflict score, where higher values indicate stronger conflict.
        """
        # Project principles into the context space
        proj_a = self.context_projection(principle_a)
        proj_b = self.context_projection(principle_b)
        context_weight = F.normalize(context, p=2, dim=-1)  # Normalize context vector

        # Measure directional conflict weighted by context
        score = torch.dot(context_weight, proj_a - proj_b).abs()
        return score

    def detect_nuanced_conflicts(self, context, threshold=0.5):
        """
        Detect nuanced, context-aware conflicts between principles.

        Args:
            context (torch.Tensor): The input context tensor.
            threshold (float): Threshold for determining significant conflict.

        Returns:
            conflicts (list): List of conflicting principle index pairs and their scores.
        """
        conflicts = []
        num_principles = len(self.principles)
        if num_principles < 2:
            return conflicts

        # Compare all principle pairs
        for i in range(num_principles):
            for j in range(i + 1, num_principles):
                score = self.contextual_conflict_score(self.principles[i], self.principles[j], context)
                if score > threshold:
                    conflicts.append((i, j, score.item()))
        return conflicts

    def forward(self, x, conflict_threshold=0.5):
        """
        Forward pass with context-aware principle alignment and conflict detection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim)
            conflict_threshold (float): Threshold for detecting nuanced conflicts.

        Returns:
            x (torch.Tensor): Updated input tensor after reflection.
            confidence (torch.Tensor): Confidence score for self-reflection.
            conflicts (list): List of nuanced conflicts detected.
        """
        # Self-reflection step
        reflection = torch.tanh(self.self_reflection(x))

        # Aggregate principles dynamically
        if len(self.principles) > 0:
            principles = torch.stack(self.principles)  # Shape: (num_principles, hidden_dim)
            principle_weights = torch.softmax(torch.matmul(x, principles.T), dim=-1)  # Attention scores
            
            # Weighted principle alignment
            principle_alignment = torch.matmul(principle_weights, principles)  # Shape: (batch_size, hidden_dim)
            adaptive_reflection = self.adaptive_alignment(reflection + principle_alignment)
        else:
            adaptive_reflection = self.adaptive_alignment(reflection)

        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_estimation(adaptive_reflection))

        # Residual connection and layer normalization
        output = self.layer_norm(x + adaptive_reflection)

        # Detect nuanced conflicts
        conflicts = self.detect_nuanced_conflicts(x.mean(dim=0), threshold=conflict_threshold)

        return output, confidence, conflicts
