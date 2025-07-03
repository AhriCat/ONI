import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add the ONI-Public modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from haptics.tactile_system import (
    TactileSignal, TouchEmotionMapper, TouchTranslator, 
    MultimodalPerceptualSystem, EmotionalLatentLearner,
    TactilePerceptionSystem, TouchType, EmotionCategory
)

@dataclass
class TrainingConfig:
    """Configuration for training the tactile system"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    val_split: float = 0.2
    patience: int = 10
    min_delta: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Component-specific learning rates
    emotion_lr: float = 1e-3
    translator_lr: float = 5e-4
    multimodal_lr: float = 1e-3
    emotional_learner_lr: float = 1e-3
    
    # Loss weights
    emotion_weight: float = 1.0
    contrastive_weight: float = 0.5
    reconstruction_weight: float = 0.3
    classification_weight: float = 0.7

class TactileDataset(Dataset):
    """Dataset for tactile perception training"""
    def __init__(self, signals: List[TactileSignal], contexts: List[np.ndarray], 
                 emotions: List[Tuple[float, float]], labels: List[int],
                 multimodal_data: Optional[Dict] = None):
        self.signals = signals
        self.contexts = contexts
        self.emotions = emotions  # (valence, arousal) pairs
        self.labels = labels  # emotion category labels
        self.multimodal_data = multimodal_data or {}
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        context = torch.tensor(self.contexts[idx], dtype=torch.float32)
        emotion = torch.tensor(self.emotions[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Convert signal to tensor
        signal_tensor = signal.to_tensor()
        
        # Prepare multimodal data if available
        multimodal = {}
        if self.multimodal_data:
            multimodal = {
                'image': torch.tensor(self.multimodal_data.get('images', [np.zeros((3, 64, 64))])[idx], dtype=torch.float32),
                'sound': torch.tensor(self.multimodal_data.get('sounds', [np.zeros(1024)])[idx], dtype=torch.float32),
                'physiological': torch.tensor(self.multimodal_data.get('physiological', [np.zeros(10)])[idx], dtype=torch.float32)
            }
        
        return {
            'signal': signal_tensor,
            'context': context,
            'emotion': emotion,
            'label': label,
            'multimodal': multimodal
        }

class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning"""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask
        batch_size = features.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        pos_sim = torch.diag(exp_sim)
        neg_sim = exp_sim.sum(dim=1) - pos_sim
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        return loss.mean()

class TactileTrainer:
    """Main trainer class for the tactile perception system"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.emotion_mapper = TouchEmotionMapper().to(self.device)
        self.touch_translator = TouchTranslator().to(self.device)
        self.multimodal_system = MultimodalPerceptualSystem(config.device).to(self.device)
        self.emotional_learner = EmotionalLatentLearner().to(self.device)
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize loss functions
        self.setup_losses()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.log_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_optimizers(self):
        """Setup optimizers for different components"""
        self.emotion_optimizer = optim.AdamW(
            self.emotion_mapper.parameters(),
            lr=self.config.emotion_lr,
            weight_decay=self.config.weight_decay
        )
        
        self.translator_optimizer = optim.AdamW(
            self.touch_translator.parameters(),
            lr=self.config.translator_lr,
            weight_decay=self.config.weight_decay
        )
        
        self.multimodal_optimizer = optim.AdamW(
            list(self.multimodal_system.parameters()),
            lr=self.config.multimodal_lr,
            weight_decay=self.config.weight_decay
        )
        
        self.emotional_optimizer = optim.AdamW(
            self.emotional_learner.parameters(),
            lr=self.config.emotional_learner_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate schedulers
        self.emotion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.emotion_optimizer, mode='min', patience=5, factor=0.5
        )
        self.translator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.translator_optimizer, mode='min', patience=5, factor=0.5
        )
        
    def setup_losses(self):
        """Setup loss functions"""
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.emotion_mapper.train()
        self.touch_translator.train()
        self.multimodal_system.train()
        self.emotional_learner.train()
        
        total_loss = 0
        component_losses = {
            'emotion': 0, 'translation': 0, 'multimodal': 0, 'contrastive': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_loss = 0
            
            # Move batch to device
            signal = batch['signal'].to(self.device)
            context = batch['context'].to(self.device)
            emotion_target = batch['emotion'].to(self.device)
            label = batch['label'].to(self.device)
            
            # 1. Train emotion mapper
            self.emotion_optimizer.zero_grad()
            
            # Create sequence dimension for emotion mapping
            signal_seq = signal.unsqueeze(1)  # [batch, 1, features]
            context_seq = context.unsqueeze(1).repeat(1, 5, 1)  # [batch, 5, context_dim]
            
            emotion_pred = self.emotion_mapper(signal_seq, context_seq)
            emotion_loss = self.mse_loss(emotion_pred, emotion_target)
            
            emotion_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.emotion_mapper.parameters(), self.config.gradient_clip)
            self.emotion_optimizer.step()
            
            component_losses['emotion'] += emotion_loss.item()
            batch_loss += emotion_loss.item() * self.config.emotion_weight
            
            # 2. Train touch translator
            self.translator_optimizer.zero_grad()
            
            # Translate between domains
            translated = self.touch_translator(signal, 'human')
            translation_loss = self.mse_loss(translated, signal)
            
            translation_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.touch_translator.parameters(), self.config.gradient_clip)
            self.translator_optimizer.step()
            
            component_losses['translation'] += translation_loss.item()
            batch_loss += translation_loss.item() * self.config.reconstruction_weight
            
            # 3. Train multimodal system (if multimodal data available)
            if batch['multimodal'] and any(v.numel() > 0 for v in batch['multimodal'].values()):
                self.multimodal_optimizer.zero_grad()
                self.emotional_optimizer.zero_grad()
                
                # Get multimodal data
                image = batch['multimodal']['image'].to(self.device)
                sound = batch['multimodal']['sound'].to(self.device)
                physiological = batch['multimodal']['physiological'].to(self.device)
                
                # Encode multimodal inputs
                v, a, t = self.multimodal_system.encode_inputs(image, sound, signal)
                fused = self.multimodal_system.fuse_modalities(v, a, t)
                
                # Emotional learning
                emotional_output = self.emotional_learner(fused, physiological)
                
                # Compute losses
                contrastive_loss = self.contrastive_loss(
                    emotional_output['contrastive'], label
                )
                classification_loss = self.ce_loss(
                    emotional_output['emotion_logits'], label
                )
                
                multimodal_loss = (
                    contrastive_loss * self.config.contrastive_weight +
                    classification_loss * self.config.classification_weight
                )
                
                multimodal_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.multimodal_system.parameters()) + 
                    list(self.emotional_learner.parameters()),
                    self.config.gradient_clip
                )
                
                self.multimodal_optimizer.step()
                self.emotional_optimizer.step()
                
                component_losses['multimodal'] += multimodal_loss.item()
                component_losses['contrastive'] += contrastive_loss.item()
                batch_loss += multimodal_loss.item()
            
            total_loss += batch_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Emotion': f'{emotion_loss.item():.4f}',
                'Translation': f'{translation_loss.item():.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                step = self.epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Loss/Batch', batch_loss, step)
                self.writer.add_scalar('Loss/Emotion', emotion_loss.item(), step)
                self.writer.add_scalar('Loss/Translation', translation_loss.item(), step)
        
        # Average losses
        avg_loss = total_loss / len(train_loader)
        for key in component_losses:
            component_losses[key] /= len(train_loader)
        
        component_losses['total'] = avg_loss
        return component_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.emotion_mapper.eval()
        self.touch_translator.eval()
        self.multimodal_system.eval()
        self.emotional_learner.eval()
        
        total_loss = 0
        component_losses = {
            'emotion': 0, 'translation': 0, 'multimodal': 0, 'contrastive': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                signal = batch['signal'].to(self.device)
                context = batch['context'].to(self.device)
                emotion_target = batch['emotion'].to(self.device)
                label = batch['label'].to(self.device)
                
                # Emotion mapping
                signal_seq = signal.unsqueeze(1)
                context_seq = context.unsqueeze(1).repeat(1, 5, 1)
                emotion_pred = self.emotion_mapper(signal_seq, context_seq)
                emotion_loss = self.mse_loss(emotion_pred, emotion_target)
                
                # Translation
                translated = self.touch_translator(signal, 'human')
                translation_loss = self.mse_loss(translated, signal)
                
                component_losses['emotion'] += emotion_loss.item()
                component_losses['translation'] += translation_loss.item()
                
                batch_loss = (
                    emotion_loss.item() * self.config.emotion_weight +
                    translation_loss.item() * self.config.reconstruction_weight
                )
                total_loss += batch_loss
        
        avg_loss = total_loss / len(val_loader)
        for key in component_losses:
            component_losses[key] /= len(val_loader)
        
        component_losses['total'] = avg_loss
        return component_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'emotion_mapper_state_dict': self.emotion_mapper.state_dict(),
            'touch_translator_state_dict': self.touch_translator.state_dict(),
            'multimodal_system_state_dict': self.multimodal_system.state_dict(),
            'emotional_learner_state_dict': self.emotional_learner.state_dict(),
            'emotion_optimizer_state_dict': self.emotion_optimizer.state_dict(),
            'translator_optimizer_state_dict': self.translator_optimizer.state_dict(),
            'multimodal_optimizer_state_dict': self.multimodal_optimizer.state_dict(),
            'emotional_optimizer_state_dict': self.emotional_optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with val_loss: {val_loss:.6f}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.emotion_mapper.load_state_dict(checkpoint['emotion_mapper_state_dict'])
        self.touch_translator.load_state_dict(checkpoint['touch_translator_state_dict'])
        self.multimodal_system.load_state_dict(checkpoint['multimodal_system_state_dict'])
        self.emotional_learner.load_state_dict(checkpoint['emotional_learner_state_dict'])
        
        self.emotion_optimizer.load_state_dict(checkpoint['emotion_optimizer_state_dict'])
        self.translator_optimizer.load_state_dict(checkpoint['translator_optimizer_state_dict'])
        self.multimodal_optimizer.load_state_dict(checkpoint['multimodal_optimizer_state_dict'])
        self.emotional_optimizer.load_state_dict(checkpoint['emotional_optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        self.logger.info(f'Loaded checkpoint from epoch {self.epoch}')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        self.logger.info('Starting training...')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Config: {self.config}')
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_losses = self.train_epoch(train_loader)
            
            # Validation phase
            val_losses = self.validate(val_loader)
            
            # Update learning rate schedulers
            self.emotion_scheduler.step(val_losses['total'])
            self.translator_scheduler.step(val_losses['total'])
            
            # Track losses
            self.train_losses.append(train_losses['total'])
            self.val_losses.append(val_losses['total'])
            
            # Log epoch results
            epoch_time = time.time() - start_time
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_losses["total"]:.6f}, '
                f'Val Loss: {val_losses["total"]:.6f}, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val', val_losses['total'], epoch)
            self.writer.add_scalar('Loss/Train_Emotion', train_losses['emotion'], epoch)
            self.writer.add_scalar('Loss/Val_Emotion', val_losses['emotion'], epoch)
            
            # Early stopping check
            if val_losses['total'] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_losses['total'], is_best=True)
            else:
                self.patience_counter += 1
                
            # Regular checkpoint saving
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_losses['total'])
                
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
        
        self.logger.info('Training completed!')
        self.writer.close()
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_curves.png'))
        plt.show()

def generate_synthetic_data(num_samples: int = 1000) -> Tuple[List[TactileSignal], List[np.ndarray], List[Tuple[float, float]], List[int]]:
    """Generate synthetic tactile data for training"""
    signals = []
    contexts = []
    emotions = []
    labels = []
    
    for i in range(num_samples):
        # Generate random tactile signal
        pressure = np.random.uniform(0, 1)
        temp = np.random.uniform(15, 40)
        vibration = np.random.uniform(0, 200)
        location = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        duration = np.random.uniform(0.01, 2.0)
        
        # Generate context
        context = np.random.randn(5, 10)  # 5 timesteps, 10 features
        
        # Generate emotion based on signal characteristics
        valence = np.tanh((pressure - 0.5) * 2)  # More pressure = more negative
        arousal = np.tanh(vibration / 100)  # More vibration = more arousal
        
        # Generate emotion label
        if valence > 0.3 and arousal < 0.3:
            label = EmotionCategory.COMFORT.value
        elif valence < -0.3 and arousal > 0.3:
            label = EmotionCategory.PAIN.value
        elif valence > 0.3 and arousal > 0.3:
            label = EmotionCategory.PLEASURE.value
        elif abs(valence) < 0.3 and abs(arousal) < 0.3:
            label = EmotionCategory.NEUTRAL.value
        else:
            label = EmotionCategory.ALERT.value
        
        signal = TactileSignal(pressure, temp, vibration, location, duration, {'mood': 'synthetic'})
        
        signals.append(signal)
        contexts.append(context)
        emotions.append((valence, arousal))
        labels.append(list(EmotionCategory).index(EmotionCategory(label)))
    
    return signals, contexts, emotions, labels

def main():
    """Main training function"""
    # Configuration
    config = TrainingConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='../checkpoints/tactile_system',
        log_dir='../logs/tactile_system'
    )
    
    # Generate synthetic data
    print("Generating synthetic data...")
    signals, contexts, emotions, labels = generate_synthetic_data(2000)
    
    # Create multimodal data
    multimodal_data = {
        'images': np.random.randn(2000, 3, 64, 64),
        'sounds': np.random.randn(2000, 1024),
        'physiological': np.random.randn(2000, 10)
    }
    
    # Create dataset
    dataset = TactileDataset(signals, contexts, emotions, labels, multimodal_data)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize trainer
    trainer = TactileTrainer(config)
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
