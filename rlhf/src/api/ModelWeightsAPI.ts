import express from 'express';
import { Request, Response } from 'express';
import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs-extra';
import * as path from 'path';
import { createReadStream, createWriteStream } from 'fs';
import { pipeline } from 'stream';
import { promisify } from 'util';
import { gzip, brotliCompress } from 'zlib';
import { DownloadRequest, ModelWeights, ModelConfig, User } from '../types';
import { HuggingFaceConverter } from '../utils/HuggingFaceConverter';
import { Logger } from '../utils/Logger';
import { AuthMiddleware } from '../middleware/AuthMiddleware';
import { RateLimitMiddleware } from '../middleware/RateLimitMiddleware';

const pipelineAsync = promisify(pipeline);

export class ModelWeightsAPI {
  private app: express.Application;
  private logger: Logger;
  private hfConverter: HuggingFaceConverter;
  private modelsPath: string;

  constructor() {
    this.app = express();
    this.logger = new Logger('ModelWeightsAPI');
    this.hfConverter = new HuggingFaceConverter();
    this.modelsPath = process.env.MODELS_PATH || './models';
    
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware(): void {
    this.app.use(express.json());
    this.app.use(AuthMiddleware);
    this.app.use('/download', RateLimitMiddleware);
  }

  private setupRoutes(): void {
    // List available models
    this.app.get('/models', this.listModels.bind(this));
    
    // Get model information
    this.app.get('/models/:modelId', this.getModelInfo.bind(this));
    
    // Download model weights
    this.app.post('/models/:modelId/download', this.downloadWeights.bind(this));
    
    // Upload model weights
    this.app.post('/models/:modelId/upload', this.uploadWeights.bind(this));
    
    // Convert to HuggingFace format
    this.app.post('/models/:modelId/convert/huggingface', this.convertToHuggingFace.bind(this));
    
    // Get download status
    this.app.get('/downloads/:downloadId/status', this.getDownloadStatus.bind(this));
  }

  private async listModels(req: Request, res: Response): Promise<void> {
    try {
      const modelsDir = await fs.readdir(this.modelsPath);
      const models = [];

      for (const modelDir of modelsDir) {
        const modelPath = path.join(this.modelsPath, modelDir);
        const configPath = path.join(modelPath, 'config.json');
        
        if (await fs.pathExists(configPath)) {
          const config = await fs.readJson(configPath);
          models.push({
            id: modelDir,
            name: config.name,
            type: config.type,
            parameters: config.parameters,
            lastModified: (await fs.stat(modelPath)).mtime,
            size: await this.getDirectorySize(modelPath)
          });
        }
      }

      res.json({
        success: true,
        data: models,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error listing models:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to list models',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async getModelInfo(req: Request, res: Response): Promise<void> {
    try {
      const { modelId } = req.params;
      const modelPath = path.join(this.modelsPath, modelId);
      
      if (!await fs.pathExists(modelPath)) {
        return res.status(404).json({
          success: false,
          error: 'Model not found',
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }

      const configPath = path.join(modelPath, 'config.json');
      const config = await fs.readJson(configPath);
      
      const weightsPath = path.join(modelPath, 'weights');
      const availableFormats = await this.getAvailableFormats(weightsPath);
      
      const modelInfo = {
        ...config,
        availableFormats,
        size: await this.getDirectorySize(modelPath),
        lastModified: (await fs.stat(modelPath)).mtime
      };

      res.json({
        success: true,
        data: modelInfo,
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error getting model info:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to get model information',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async downloadWeights(req: Request, res: Response): Promise<void> {
    try {
      const { modelId } = req.params;
      const downloadRequest: DownloadRequest = req.body;
      const user = req.user as User;

      // Validate download request
      const validation = await this.validateDownloadRequest(downloadRequest, user);
      if (!validation.valid) {
        return res.status(400).json({
          success: false,
          error: validation.error,
          timestamp: new Date(),
          requestId: req.headers['x-request-id']
        });
      }

      // Generate download ID
      const downloadId = this.generateDownloadId();
      
      // Start async download preparation
      this.prepareDownload(downloadId, modelId, downloadRequest, user);

      res.json({
        success: true,
        data: {
          downloadId,
          status: 'preparing',
          estimatedTime: '5-10 minutes'
        },
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    } catch (error) {
      this.logger.error('Error initiating download:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to initiate download',
        timestamp: new Date(),
        requestId: req.headers['x-request-id']
      });
    }
  }

  private async prepareDownload(
    downloadId: string,
    modelId: string,
    request: DownloadRequest,
    user: User
  ): Promise<void> {
    try {
      this.logger.info(`Preparing download ${downloadId} for model ${modelId}`);
      
      const modelPath = path.join(this.modelsPath, modelId);
      const outputPath = path.join('./downloads', downloadId);
      
      await fs.ensureDir(outputPath);
      
      // Load model
      const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
      
      // Convert to requested format
      let convertedPath: string;
      
      switch (request.format) {
        case 'pytorch':
          convertedPath = await this.convertToPyTorch(model, outputPath);
          break;
        case 'tensorflow':
          convertedPath = await this.convertToTensorFlow(model, outputPath);
          break;
        case 'onnx':
          convertedPath = await this.convertToONNX(model, outputPath);
          break;
        case 'huggingface':
          convertedPath = await this.convertToHuggingFaceFormat(model, modelId, outputPath);
          break;
        default:
          throw new Error(`Unsupported format: ${request.format}`);
      }
      
      // Apply compression if requested
      if (request.compression !== 'none') {
        convertedPath = await this.compressWeights(convertedPath, request.compression);
      }
      
      // Apply quantization if requested
      if (request.precision !== 'fp32') {
        convertedPath = await this.quantizeWeights(convertedPath, request.precision);
      }
      
      // Update download status
      await this.updateDownloadStatus(downloadId, 'ready', convertedPath);
      
      this.logger.info(`Download ${downloadId} prepared successfully`);
      
      // Cleanup model from memory
      model.dispose();
      
    } catch (error) {
      this.logger.error(`Error preparing download ${downloadId}:`, error);
      await this.updateDownloadStatus(downloadId, 'failed', undefined, error.message);
    }
  }

  private async convertToHuggingFaceFormat(
    model: tf.LayersModel,
    modelId: string,
    outputPath: string
  ): Promise<string> {
    const hfPath = path.join(outputPath, 'huggingface');
    await fs.ensureDir(hfPath);
    
    // Convert model to HuggingFace format
    await this.hfConverter.convertModel(model, modelId, hfPath);
    
    // Create HuggingFace config
    const config = await this.createHuggingFaceConfig(modelId);
    await fs.writeJson(path.join(hfPath, 'config.json'), config, { spaces: 2 });
    
    // Create tokenizer config if applicable
    const tokenizerConfig = await this.createTokenizerConfig(modelId);
    if (tokenizerConfig) {
      await fs.writeJson(path.join(hfPath, 'tokenizer.json'), tokenizerConfig, { spaces: 2 });
    }
    
    // Create README
    const readme = await this.createHuggingFaceReadme(modelId);
    await fs.writeFile(path.join(hfPath, 'README.md'), readme);
    
    return hfPath;
  }

  private async createHuggingFaceConfig(modelId: string): Promise<any> {
    const modelPath = path.join(this.modelsPath, modelId);
    const configPath = path.join(modelPath, 'config.json');
    const originalConfig = await fs.readJson(configPath);
    
    return {
      architectures: [originalConfig.architecture || 'ONIModel'],
      model_type: 'oni',
      vocab_size: originalConfig.hyperparameters?.vocabSize || 300000,
      hidden_size: originalConfig.hyperparameters?.hiddenDim || 896,
      num_attention_heads: originalConfig.hyperparameters?.numHeads || 8,
      num_hidden_layers: originalConfig.hyperparameters?.numLayers || 6,
      max_position_embeddings: originalConfig.hyperparameters?.maxLength || 4096,
      type_vocab_size: 2,
      initializer_range: 0.02,
      layer_norm_eps: 1e-12,
      pad_token_id: 0,
      position_embedding_type: 'absolute',
      use_cache: true,
      classifier_dropout: null,
      torch_dtype: 'float32',
      transformers_version: '4.21.0',
      oni_version: '1.0.0',
      training_config: originalConfig.training_config,
      performance_metrics: originalConfig.performance_metrics
    };
  }

  private async createTokenizerConfig(modelId: string): Promise<any> {
    // This would create a tokenizer config compatible with HuggingFace
    // For now, return a basic BPE tokenizer config
    return {
      version: '1.0',
      truncation: null,
      padding: null,
      added_tokens: [
        { id: 0, content: '[PAD]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 1, content: '[UNK]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 2, content: '[CLS]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 3, content: '[SEP]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 4, content: '[MASK]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 5, content: '[EOS]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true }
      ],
      normalizer: {
        type: 'BertNormalizer',
        clean_text: true,
        handle_chinese_chars: true,
        strip_accents: null,
        lowercase: true
      },
      pre_tokenizer: {
        type: 'BertPreTokenizer'
      },
      post_processor: {
        type: 'BertProcessing',
        sep: ['[SEP]', 3],
        cls: ['[CLS]', 2]
      },
      decoder: {
        type: 'BertDecoder'
      },
      model: {
        type: 'BPE',
        dropout: null,
        unk_token: '[UNK]',
        continuing_subword_prefix: '##',
        end_of_word_suffix: null,
        fuse_unk: false,
        vocab: {},
        merges: []
      }
    };
  }

  private async createHuggingFaceReadme(modelId: string): Promise<string> {
    const modelPath = path.join(this.modelsPath, modelId);
    const configPath = path.join(modelPath, 'config.json');
    const config = await fs.readJson(configPath);
    
    return `---
license: pantheum
tags:
- oni
- agi
- rlhf
- compassion
- ethical-ai
language:
- en
pipeline_tag: text-generation
---

# ${config.name}

This is an ONI (Omni-Neural Intelligence) model trained with Reinforcement Learning from Human Feedback (RLHF) and integrated with a compassion framework for ethical AI behavior.

## Model Details

- **Model Type**: ${config.type}
- **Architecture**: ${config.architecture}
- **Parameters**: ${config.parameters?.toLocaleString() || 'Unknown'}
- **Training**: RLHF with Compassion Framework
- **License**: Pantheum License

## Features

- **Ethical AI**: Integrated compassion framework ensuring beneficial behavior
- **Multi-modal**: Supports text, vision, and audio processing
- **Self-aware**: Meta-cognitive capabilities for self-reflection
- **Emotionally intelligent**: Emotional processing and energy management

## Usage

\`\`\`python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("${config.huggingfaceRepo || `oni-models/${modelId}`}")
model = AutoModelForCausalLM.from_pretrained("${config.huggingfaceRepo || `oni-models/${modelId}`}")

inputs = tokenizer("Hello, how can I help you today?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
\`\`\`

## Training Data

This model was trained using:
- Human feedback data with compassion ratings
- Multi-modal datasets for comprehensive understanding
- Ethical constraint validation during training

## Ethical Considerations

This model includes built-in ethical safeguards:
- Compassion framework evaluation for all outputs
- Bias detection and mitigation
- Harmful content filtering
- Transparency in decision-making

## Performance

${config.performance_metrics ? Object.entries(config.performance_metrics).map(([key, value]) => `- **${key}**: ${value}`).join('\n') : 'Performance metrics available upon request.'}

## Citation

\`\`\`bibtex
@misc{oni-${modelId},
  title={ONI: Omni-Neural Intelligence with Compassion Framework},
  author={ONI Team},
  year={2024},
  url={https://github.com/ahricat/oni}
}
\`\`\`

## License

This model is released under the Pantheum License. See LICENSE file for details.
`;
  }

  private async convertToPyTorch(model: tf.LayersModel, outputPath: string): Promise<string> {
    // This would implement TensorFlow.js to PyTorch conversion
    // For now, save in TensorFlow format and provide conversion instructions
    const tfPath = path.join(outputPath, 'tensorflow');
    await model.save(`file://${tfPath}`);
    
    // Create conversion script
    const conversionScript = `
# Convert TensorFlow.js model to PyTorch
# Run this script in a Python environment with both TensorFlow and PyTorch installed

import tensorflow as tf
import torch
import json
import numpy as np

def convert_to_pytorch():
    # Load TensorFlow model
    model = tf.keras.models.load_model('${tfPath}')
    
    # Extract weights and convert to PyTorch format
    pytorch_state_dict = {}
    
    for layer in model.layers:
        for weight in layer.weights:
            weight_name = weight.name.replace(':', '_').replace('/', '_')
            weight_value = weight.numpy()
            pytorch_state_dict[weight_name] = torch.from_numpy(weight_value)
    
    # Save PyTorch state dict
    torch.save(pytorch_state_dict, 'model.pth')
    
    # Save model config
    config = {
        'architecture': 'ONIModel',
        'num_layers': len([l for l in model.layers if 'dense' in l.name.lower()]),
        'hidden_size': model.layers[-1].output_shape[-1],
        'vocab_size': model.layers[0].input_shape[-1] if hasattr(model.layers[0], 'input_shape') else 300000
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':
    convert_to_pytorch()
`;
    
    await fs.writeFile(path.join(outputPath, 'convert_to_pytorch.py'), conversionScript);
    return outputPath;
  }

  private async convertToTensorFlow(model: tf.LayersModel, outputPath: string): Promise<string> {
    const tfPath = path.join(outputPath, 'tensorflow');
    await model.save(`file://${tfPath}`);
    return tfPath;
  }

  private async convertToONNX(model: tf.LayersModel, outputPath: string): Promise<string> {
    // This would implement TensorFlow.js to ONNX conversion
    // For now, provide conversion instructions
    const onnxPath = path.join(outputPath, 'onnx');
    await fs.ensureDir(onnxPath);
    
    // Save TensorFlow model first
    const tfPath = path.join(onnxPath, 'tensorflow');
    await model.save(`file://${tfPath}`);
    
    // Create ONNX conversion script
    const conversionScript = `
# Convert TensorFlow model to ONNX
# Requires: pip install tf2onnx

import subprocess
import sys

def convert_to_onnx():
    cmd = [
        sys.executable, '-m', 'tf2onnx.convert',
        '--saved-model', '${tfPath}',
        '--output', 'model.onnx',
        '--opset', '11'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Successfully converted to ONNX format")
    else:
        print(f"Conversion failed: {result.stderr}")

if __name__ == '__main__':
    convert_to_onnx()
`;
    
    await fs.writeFile(path.join(onnxPath, 'convert_to_onnx.py'), conversionScript);
    return onnxPath;
  }

  private async compressWeights(weightsPath: string, compression: string): Promise<string> {
    const compressedPath = `${weightsPath}_compressed`;
    await fs.ensureDir(compressedPath);
    
    const files = await fs.readdir(weightsPath);
    
    for (const file of files) {
      const inputPath = path.join(weightsPath, file);
      const outputPath = path.join(compressedPath, `${file}.${compression}`);
      
      const readStream = createReadStream(inputPath);
      const writeStream = createWriteStream(outputPath);
      
      let compressionStream;
      switch (compression) {
        case 'gzip':
          compressionStream = gzip();
          break;
        case 'brotli':
          compressionStream = brotliCompress();
          break;
        default:
          throw new Error(`Unsupported compression: ${compression}`);
      }
      
      await pipelineAsync(readStream, compressionStream, writeStream);
    }
    
    return compressedPath;
  }

  private async quantizeWeights(weightsPath: string, precision: string): Promise<string> {
    // This would implement weight quantization
    // For now, just copy the weights and add a note about quantization
    const quantizedPath = `${weightsPath}_${precision}`;
    await fs.copy(weightsPath, quantizedPath);
    
    const quantizationNote = `
# Weight Quantization to ${precision}

This directory contains weights that should be quantized to ${precision} precision.
Use the following script to perform quantization:

\`\`\`python
import torch
import numpy as np

def quantize_weights(input_path, output_path, precision='${precision}'):
    # Load weights
    if input_path.endswith('.pth'):
        weights = torch.load(input_path)
        
        # Quantize based on precision
        if precision == 'fp16':
            quantized_weights = {k: v.half() for k, v in weights.items()}
        elif precision == 'int8':
            quantized_weights = {k: torch.quantize_per_tensor(v, scale=0.1, zero_point=0, dtype=torch.qint8) 
                               for k, v in weights.items()}
        elif precision == 'int4':
            # Custom int4 quantization
            quantized_weights = {}
            for k, v in weights.items():
                # Simplified int4 quantization
                min_val, max_val = v.min(), v.max()
                scale = (max_val - min_val) / 15  # 4-bit range: 0-15
                quantized = torch.round((v - min_val) / scale).clamp(0, 15).byte()
                quantized_weights[k] = {'data': quantized, 'scale': scale, 'min_val': min_val}
        
        torch.save(quantized_weights, output_path)
\`\`\`
`;
    
    await fs.writeFile(path.join(quantizedPath, 'quantization_instructions.md'), quantizationNote);
    return quantizedPath;
  }

  private async validateDownloadRequest(request: DownloadRequest, user: User): Promise<{valid: boolean, error?: string}> {
    // Check user permissions
    if (!user.permissions.includes('download_models')) {
      return { valid: false, error: 'Insufficient permissions to download models' };
    }
    
    // Check format support
    const supportedFormats = ['pytorch', 'tensorflow', 'onnx', 'huggingface'];
    if (!supportedFormats.includes(request.format)) {
      return { valid: false, error: `Unsupported format: ${request.format}` };
    }
    
    // Check compression support
    const supportedCompression = ['none', 'gzip', 'brotli'];
    if (!supportedCompression.includes(request.compression)) {
      return { valid: false, error: `Unsupported compression: ${request.compression}` };
    }
    
    // Check precision support
    const supportedPrecision = ['fp32', 'fp16', 'int8', 'int4'];
    if (!supportedPrecision.includes(request.precision)) {
      return { valid: false, error: `Unsupported precision: ${request.precision}` };
    }
    
    return { valid: true };
  }

  private generateDownloadId(): string {
    return `download_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private async updateDownloadStatus(
    downloadId: string,
    status: string,
    filePath?: string,
    error?: string
  ): Promise<void> {
    // This would update the download status in a database
    // For now, just log the status
    this.logger.info(`Download ${downloadId} status: ${status}`, { filePath, error });
  }

  private async getAvailableFormats(weightsPath: string): Promise<string[]> {
    const formats = [];
    
    if (await fs.pathExists(path.join(weightsPath, 'model.json'))) {
      formats.push('tensorflow');
    }
    
    // Check for other formats
    formats.push('pytorch', 'onnx', 'huggingface'); // These can be converted
    
    return formats;
  }

  private async getDirectorySize(dirPath: string): Promise<number> {
    let size = 0;
    const files = await fs.readdir(dirPath, { withFileTypes: true });
    
    for (const file of files) {
      const filePath = path.join(dirPath, file.name);
      if (file.isDirectory()) {
        size += await this.getDirectorySize(filePath);
      } else {
        const stats = await fs.stat(filePath);
        size += stats.size;
      }
    }
    
    return size;
  }

  public getApp(): express.Application {
    return this.app;
  }
}