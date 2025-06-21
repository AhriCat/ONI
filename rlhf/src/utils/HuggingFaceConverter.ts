import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs-extra';
import * as path from 'path';
import { Logger } from './Logger';

export class HuggingFaceConverter {
  private logger: Logger;

  constructor() {
    this.logger = new Logger('HuggingFaceConverter');
  }

  public async convertModel(
    model: tf.LayersModel,
    modelId: string,
    outputPath: string
  ): Promise<void> {
    try {
      this.logger.info(`Converting model ${modelId} to HuggingFace format`);

      // Save the TensorFlow.js model first
      const tfPath = path.join(outputPath, 'tensorflow');
      await fs.ensureDir(tfPath);
      await model.save(`file://${tfPath}`);

      // Extract model architecture and weights
      const architecture = await this.extractArchitecture(model);
      const weights = await this.extractWeights(model);

      // Convert to HuggingFace PyTorch format
      await this.createPyTorchWeights(weights, outputPath);
      
      // Create HuggingFace configuration files
      await this.createModelConfig(architecture, modelId, outputPath);
      await this.createTokenizerFiles(modelId, outputPath);
      await this.createModelingFile(architecture, modelId, outputPath);

      this.logger.info(`Model ${modelId} successfully converted to HuggingFace format`);
    } catch (error) {
      this.logger.error(`Failed to convert model ${modelId}:`, error);
      throw error;
    }
  }

  private async extractArchitecture(model: tf.LayersModel): Promise<any> {
    const architecture = {
      layers: [],
      inputShape: model.inputs[0].shape,
      outputShape: model.outputs[0].shape,
      totalParams: model.countParams()
    };

    for (let i = 0; i < model.layers.length; i++) {
      const layer = model.layers[i];
      architecture.layers.push({
        name: layer.name,
        className: layer.getClassName(),
        config: layer.getConfig(),
        inputShape: layer.inputShape,
        outputShape: layer.outputShape
      });
    }

    return architecture;
  }

  private async extractWeights(model: tf.LayersModel): Promise<Map<string, tf.Tensor>> {
    const weights = new Map<string, tf.Tensor>();

    for (const layer of model.layers) {
      for (const weight of layer.weights) {
        weights.set(weight.name, weight.read());
      }
    }

    return weights;
  }

  private async createPyTorchWeights(
    weights: Map<string, tf.Tensor>,
    outputPath: string
  ): Promise<void> {
    const weightsData: { [key: string]: number[] } = {};

    for (const [name, tensor] of weights) {
      const data = await tensor.data();
      weightsData[this.convertWeightName(name)] = Array.from(data);
    }

    // Create PyTorch state dict format
    const pytorchScript = `
import torch
import json
import numpy as np

# Load weight data
with open('weights_data.json', 'r') as f:
    weights_data = json.load(f)

# Convert to PyTorch tensors
state_dict = {}
for name, data in weights_data.items():
    # Reshape based on layer type and name
    tensor_data = np.array(data)
    
    # Determine shape based on weight name
    if 'kernel' in name or 'weight' in name:
        if len(tensor_data.shape) == 1:
            # Bias or 1D weight
            shape = tensor_data.shape
        elif 'dense' in name or 'linear' in name:
            # Dense layer: transpose for PyTorch format
            shape = tensor_data.shape
            if len(shape) == 2:
                tensor_data = tensor_data.T
    
    state_dict[name] = torch.from_numpy(tensor_data.reshape(shape))

# Save PyTorch state dict
torch.save(state_dict, 'pytorch_model.bin')
print(f"Converted {len(state_dict)} weight tensors to PyTorch format")
`;

    await fs.writeJson(path.join(outputPath, 'weights_data.json'), weightsData);
    await fs.writeFile(path.join(outputPath, 'convert_weights.py'), pytorchScript);

    this.logger.info(`Created PyTorch weight conversion script with ${weights.size} tensors`);
  }

  private convertWeightName(tfName: string): string {
    // Convert TensorFlow.js weight names to PyTorch format
    return tfName
      .replace(/\//g, '.')
      .replace(':0', '')
      .replace('kernel', 'weight')
      .replace('bias', 'bias');
  }

  private async createModelConfig(
    architecture: any,
    modelId: string,
    outputPath: string
  ): Promise<void> {
    const config = {
      "_name_or_path": modelId,
      "architectures": ["ONIModel"],
      "auto_map": {
        "AutoConfig": "configuration_oni.ONIConfig",
        "AutoModel": "modeling_oni.ONIModel",
        "AutoModelForCausalLM": "modeling_oni.ONIForCausalLM"
      },
      "model_type": "oni",
      "torch_dtype": "float32",
      "transformers_version": "4.21.0",
      
      // Model-specific configuration
      "vocab_size": this.extractVocabSize(architecture),
      "hidden_size": this.extractHiddenSize(architecture),
      "num_hidden_layers": this.extractNumLayers(architecture),
      "num_attention_heads": this.extractNumHeads(architecture),
      "intermediate_size": this.extractIntermediateSize(architecture),
      "max_position_embeddings": this.extractMaxLength(architecture),
      
      // ONI-specific configuration
      "use_emotional_processing": true,
      "use_compassion_framework": true,
      "use_metacognition": true,
      "use_memory_attention": true,
      
      // Training configuration
      "initializer_range": 0.02,
      "layer_norm_eps": 1e-12,
      "pad_token_id": 0,
      "bos_token_id": 6,
      "eos_token_id": 5,
      "sep_token_id": 3,
      "cls_token_id": 2,
      "mask_token_id": 4,
      
      // Generation configuration
      "use_cache": true,
      "is_encoder_decoder": false,
      "add_cross_attention": false,
      
      // ONI architecture details
      "oni_architecture": {
        "total_layers": architecture.layers.length,
        "total_parameters": architecture.totalParams,
        "input_shape": architecture.inputShape,
        "output_shape": architecture.outputShape,
        "layer_details": architecture.layers
      }
    };

    await fs.writeJson(path.join(outputPath, 'config.json'), config, { spaces: 2 });
  }

  private async createTokenizerFiles(modelId: string, outputPath: string): Promise<void> {
    // Create tokenizer.json
    const tokenizerConfig = {
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 4, "content": "[MASK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 5, "content": "[EOS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": {
        "type": "BertNormalizer",
        "clean_text": true,
        "handle_chinese_chars": true,
        "strip_accents": null,
        "lowercase": true
      },
      "pre_tokenizer": {
        "type": "BertPreTokenizer"
      },
      "post_processor": {
        "type": "BertProcessing",
        "sep": ["[SEP]", 3],
        "cls": ["[CLS]", 2]
      },
      "decoder": {
        "type": "BertDecoder"
      },
      "model": {
        "type": "BPE",
        "dropout": null,
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "vocab": {},
        "merges": []
      }
    };

    await fs.writeJson(path.join(outputPath, 'tokenizer.json'), tokenizerConfig, { spaces: 2 });

    // Create tokenizer_config.json
    const tokenizerConfigFile = {
      "do_lower_case": true,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "eos_token": "[EOS]",
      "bos_token": "[CLS]",
      "tokenize_chinese_chars": true,
      "strip_accents": null,
      "model_max_length": 4096,
      "special_tokens_map_file": null,
      "name_or_path": modelId,
      "tokenizer_class": "BertTokenizer"
    };

    await fs.writeJson(path.join(outputPath, 'tokenizer_config.json'), tokenizerConfigFile, { spaces: 2 });

    // Create special_tokens_map.json
    const specialTokensMap = {
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "pad_token": "[PAD]",
      "sep_token": "[SEP]",
      "unk_token": "[UNK]",
      "eos_token": "[EOS]",
      "bos_token": "[CLS]"
    };

    await fs.writeJson(path.join(outputPath, 'special_tokens_map.json'), specialTokensMap, { spaces: 2 });
  }

  private async createModelingFile(
    architecture: any,
    modelId: string,
    outputPath: string
  ): Promise<void> {
    const modelingCode = `
"""
ONI Model implementation for HuggingFace Transformers
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union

class ONIConfig(PretrainedConfig):
    model_type = "oni"
    
    def __init__(
        self,
        vocab_size=300000,
        hidden_size=896,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=3584,
        max_position_embeddings=4096,
        use_emotional_processing=True,
        use_compassion_framework=True,
        use_metacognition=True,
        use_memory_attention=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.use_emotional_processing = use_emotional_processing
        self.use_compassion_framework = use_compassion_framework
        self.use_metacognition = use_metacognition
        self.use_memory_attention = use_memory_attention

class ONIModel(PreTrainedModel):
    config_class = ONIConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Core embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ONILayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # ONI-specific modules
        if config.use_emotional_processing:
            self.emotional_layer = EmotionalProcessingLayer(config.hidden_size)
        
        if config.use_compassion_framework:
            self.compassion_layer = CompassionLayer(config.hidden_size)
        
        if config.use_metacognition:
            self.metacognition_layer = MetaCognitionLayer(config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        batch_size, seq_length = input_ids.shape
        
        # Embeddings
        inputs_embeds = self.embeddings(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Apply emotional processing if enabled
        if hasattr(self, 'emotional_layer'):
            hidden_states = self.emotional_layer(hidden_states)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply compassion framework if enabled
        if hasattr(self, 'compassion_layer'):
            hidden_states = self.compassion_layer(hidden_states)
        
        # Apply metacognition if enabled
        if hasattr(self, 'metacognition_layer'):
            hidden_states = self.metacognition_layer(hidden_states)
        
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states

class ONIForCausalLM(ONIModel):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        hidden_states = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )

class ONILayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + self.dropout(ff_output))
        
        return hidden_states

class EmotionalProcessingLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emotional_transform = nn.Linear(hidden_size, hidden_size)
        self.valence_head = nn.Linear(hidden_size, 1)
        self.arousal_head = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        emotional_features = torch.tanh(self.emotional_transform(hidden_states))
        valence = torch.tanh(self.valence_head(emotional_features))
        arousal = torch.sigmoid(self.arousal_head(emotional_features))
        
        # Modulate hidden states based on emotional state
        emotional_modulation = 1.0 + (valence * arousal)
        return hidden_states * emotional_modulation

class CompassionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.agency_head = nn.Linear(hidden_size, 1)
        self.capability_head = nn.Linear(hidden_size, 1)
        self.suffering_head = nn.Linear(hidden_size, 1)
        self.compassion_transform = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states):
        agency = torch.sigmoid(self.agency_head(hidden_states))
        capability = torch.sigmoid(self.capability_head(hidden_states))
        suffering = torch.sigmoid(self.suffering_head(hidden_states))
        
        # Compassion score: Agency + Capability - Suffering
        compassion_score = agency + capability - suffering
        compassion_features = torch.tanh(self.compassion_transform(hidden_states))
        
        # Apply compassion modulation
        return hidden_states + (compassion_score * compassion_features)

class MetaCognitionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.confidence_head = nn.Linear(hidden_size, 1)
        self.uncertainty_head = nn.Linear(hidden_size, 1)
        self.metacog_transform = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states):
        confidence = torch.sigmoid(self.confidence_head(hidden_states))
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden_states))
        
        # Meta-cognitive modulation
        metacog_features = torch.tanh(self.metacog_transform(hidden_states))
        metacog_modulation = confidence * (1 - uncertainty)
        
        return hidden_states + (metacog_modulation * metacog_features)
`;

    await fs.writeFile(path.join(outputPath, 'modeling_oni.py'), modelingCode);

    // Create configuration file
    const configCode = `
"""
ONI Configuration for HuggingFace Transformers
"""

from transformers import PretrainedConfig

class ONIConfig(PretrainedConfig):
    model_type = "oni"
    
    def __init__(
        self,
        vocab_size=300000,
        hidden_size=896,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=3584,
        max_position_embeddings=4096,
        use_emotional_processing=True,
        use_compassion_framework=True,
        use_metacognition=True,
        use_memory_attention=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.use_emotional_processing = use_emotional_processing
        self.use_compassion_framework = use_compassion_framework
        self.use_metacognition = use_metacognition
        self.use_memory_attention = use_memory_attention
`;

    await fs.writeFile(path.join(outputPath, 'configuration_oni.py'), configCode);
  }

  private extractVocabSize(architecture: any): number {
    // Try to find embedding layer and extract vocab size
    const embeddingLayer = architecture.layers.find((layer: any) => 
      layer.className === 'Embedding' || layer.name.includes('embedding')
    );
    
    return embeddingLayer?.config?.inputDim || 300000;
  }

  private extractHiddenSize(architecture: any): number {
    // Try to find the hidden dimension from layers
    const denseLayer = architecture.layers.find((layer: any) => 
      layer.className === 'Dense' || layer.name.includes('dense')
    );
    
    return denseLayer?.config?.units || 896;
  }

  private extractNumLayers(architecture: any): number {
    // Count transformer-like layers
    return architecture.layers.filter((layer: any) => 
      layer.className === 'Dense' || 
      layer.className === 'LSTM' || 
      layer.className === 'Attention'
    ).length || 6;
  }

  private extractNumHeads(architecture: any): number {
    // Try to find attention layer and extract number of heads
    const attentionLayer = architecture.layers.find((layer: any) => 
      layer.className === 'MultiHeadAttention' || layer.name.includes('attention')
    );
    
    return attentionLayer?.config?.numHeads || 8;
  }

  private extractIntermediateSize(architecture: any): number {
    // Typically 4x hidden size
    return this.extractHiddenSize(architecture) * 4;
  }

  private extractMaxLength(architecture: any): number {
    // Try to find max sequence length from input shape or config
    if (architecture.inputShape && architecture.inputShape.length > 1) {
      return architecture.inputShape[1] || 4096;
    }
    return 4096;
  }
}