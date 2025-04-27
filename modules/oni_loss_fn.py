import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

class UpgradedDynamicAdvancedNLPLoss(nn.Module):
    def __init__(self, ignore_index, label_smoothing=0.1, focal_gamma=2.0, dice_smooth=1.0, 
                 ce_weight=1.0, focal_weight=1.0, dice_weight=1.0, tversky_weight=1.0,
                 tversky_alpha=0.3, tversky_beta=0.7, focal_tversky_gamma=0.75):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.dice_smooth = dice_smooth
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_tversky_gamma = focal_tversky_gamma
        self.epsilon = 1e-6
        self.accumulated_gradients = 0
        self.accumulation_steps = 12 # Adjust as needed

    def normalized_loss_weights(self):
        total_weight = self.ce_weight + self.focal_weight + self.dice_weight + self.tversky_weight
        if total_weight == 0:
            total_weight = self.epsilon
        return (self.ce_weight / total_weight, self.focal_weight / total_weight,
                self.dice_weight / total_weight, self.tversky_weight / total_weight)

    def clip_gradients(self, input, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(input.parameters(), max_norm)

    def forward(self, model_output, target, epoch=0, max_epochs=100, model=None):
        """
        Compute a dynamically weighted combination of different loss functions.

        Parameters:
            model_output (torch.Tensor): The model's predictions.
            target (torch.Tensor): The ground truth labels.
            epoch (int): The current epoch (default: 0).
            max_epochs (int): The maximum number of epochs for dynamic weight adjustment.
            model (nn.Module): The model, used for gradient clipping (optional).
        """
        # Reshape tensors and move target to the correct device
        model_output, target = self._reshape_tensors(model_output, target)
        target = target.to(model_output.device)

        # Debugging info (optional, can be commented out)
        debug_inputs(model_output, target)

        # Validity check on inputs
        self._check_input_validity(model_output)

        # Compute individual losses
        ce_loss = self.cross_entropy_loss(model_output, target)
        focal_loss = self.focal_loss(model_output, target, self.adaptive_temperature(model_output))
        dice_loss = self.dice_loss(model_output, target)
        tversky_loss = self.focal_tversky_loss(model_output, target)

        # Dynamically adjust weights based on epoch progress
        self.update_loss_weights(epoch, max_epochs)

        # Normalize weights to sum to 1
        normalized_weights = self.normalized_loss_weights()

        # Compute total loss as a weighted sum of the losses
        total_loss = (
            normalized_weights[0] * ce_loss +
            normalized_weights[1] * self.dynamic_weight(focal_loss) * focal_loss +
            normalized_weights[2] * self.dynamic_weight(dice_loss) * dice_loss +
            normalized_weights[3] * self.dynamic_weight(tversky_loss) * tversky_loss
        )

        # Perform gradient accumulation, clip gradients when accumulation is complete
        total_loss = total_loss / self.accumulation_steps
        total_loss.backward()

        self.accumulated_gradients += 1
        if self.accumulated_gradients == self.accumulation_steps:
            if model is not None:
                self.clip_gradients(model)  # Clip gradients
            self.accumulated_gradients = 0

        return total_loss


    
    def centralized_gradient(self, input):
        for param in input.parameters():
            if param.grad is not None:
                param.grad.data.add_(-param.grad.data.mean())

    def adaptive_temperature(self, input):
        prob_dist = F.softmax(input, dim=-1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + self.epsilon), dim=-1)
        return 1.0 + torch.sigmoid(entropy).mean().item()

    def focal_tversky_loss(self, input, target):
        input = F.softmax(input, dim=-1)
        target_one_hot = F.one_hot(target, num_classes=input.size(-1)).float()
        mask = (target != self.ignore_index).float().unsqueeze(-1)

        true_pos = (input * target_one_hot * mask).sum(dim=(1, 2))
        false_neg = (target_one_hot * (1 - input) * mask).sum(dim=(1, 2))
        false_pos = ((1 - target_one_hot) * input * mask).sum(dim=(1, 2))

        tversky = (true_pos + self.epsilon) / (true_pos + self.tversky_alpha * false_neg + self.tversky_beta * false_pos + self.epsilon)
        focal_tversky = (1 - tversky) ** self.focal_tversky_gamma

        return focal_tversky.mean()

    def mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _reshape_tensors(self, input, target):
        if input.dim() > 3:
            input = input.view(-1, input.size(-2), input.size(-1))
        if target.dim() > 2:
            target = target.view(-1, target.size(-1))
        return input, target

    def _check_input_validity(self, input):
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Input contains NaN or Inf values")

    def adaptive_label_smoothing(self, input):
        prob_dist = F.softmax(input, dim=-1)
        uncertainty = -torch.sum(prob_dist * torch.log(prob_dist + self.epsilon), dim=-1)  # Entropy
        dynamic_smoothing = self.label_smoothing * torch.sigmoid(uncertainty)
        return dynamic_smoothing

    def cross_entropy_loss(self, input, target):
        dynamic_smoothing = self.adaptive_label_smoothing(input)
        return F.cross_entropy(
            input.view(-1, input.size(-1)),
            target.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=dynamic_smoothing.mean().item()  # Use mean smoothing
        )

    def focal_loss(self, input, target, temperature=1.0):
        input_scaled = input / temperature
        ce_loss = F.cross_entropy(input_scaled.view(-1, input.size(-1)), target.view(-1),
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


    def dice_loss(self, input, target):
        input = F.softmax(input, dim=-1)
        target_one_hot = F.one_hot(target, num_classes=input.size(-1)).float()
        
        # Create a mask where ignore_index is marked as zero and valid classes are 1
        mask = (target != self.ignore_index).float().unsqueeze(-1)

        # Exclude the ignored indices in the intersection and denominator calculations
        intersection = (input * target_one_hot * mask).sum(dim=(1, 2))
        denominator = (input.sum(dim=(1, 2)) + target_one_hot.sum(dim=(1, 2))) * mask.sum(dim=(1, 2))
        
        # Avoid division by zero using the small epsilon value
        dice_score = (2. * intersection + self.dice_smooth) / (denominator + self.dice_smooth + self.epsilon)
        return 1 - dice_score.mean()
    
    def differentiable_augment(self, input):
        noise = torch.randn_like(input) * 0.01
        input_aug = input + noise
        return input_aug

    def multi_task_loss_regularization(self, input, task_weights, tasks):
        total_loss = 0.0
        for task, weight in zip(tasks, task_weights):
            loss = self.forward(input, task['target'])
            total_loss += weight * loss
        return total_loss

    @torch.jit.export
    def update_weights(self, ce_weight=None, focal_weight=None, dice_weight=None, tversky_weight = None):
        if ce_weight is not None:
            self.ce_weight = ce_weight
        if focal_weight is not None:
            self.focal_weight = focal_weight
        if dice_weight is not None:
            self.dice_weight = dice_weight
        if self.tversky_weight is not None:
            self.tversky_weight = tversky_weight

    def update_loss_weights(self, epoch, max_epochs):
        factor = epoch / max_epochs
        self.ce_weight = self.ce_weight * (1 - factor)
        self.focal_weight = self.focal_weight * factor
        self.dice_weight = self.dice_weight * factor
        self.tversky_weight = self.tversky_weight * factor

def debug_inputs(input, target):
    print(f"Input shape: {input.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Input device: {input.device}")
    print(f"Target device: {target.device}")
    print(f"Input dtype: {input.dtype}")
    print(f"Target dtype: {target.dtype}")
    print(f"Input min/max: {input.min().item():.4f}/{input.max().item():.4f}")
    print(f"Target unique values: {target.unique().tolist()}")

def create_loss_function(tokenizer):
    return UpgradedDynamicAdvancedNLPLoss(ignore_index=tokenizer.pad_token_id if tokenizer else -100)

# Example usage with learning rate scheduler
criterion = create_loss_function(tokenizer)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#scheduler = CosineAnnealingLR(optimizer, T_max=100)  # Adjust T_max based on your training schedule
