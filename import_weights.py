import os
import torch
import joblib
import onnx
import tensorflow as tf
from tensorflow import keras
from transformers import AutoModel, AutoTokenizer

def import_weights(filepath, model=None):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.pt', '.pth', '.ckpt']:
        if model is None:
            raise ValueError("You must pass a PyTorch model instance to load weights into.")
        state_dict = torch.load(filepath, map_location='cpu')
        if 'state_dict' in state_dict:  # for Lightning or wrapped ckpt
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        return model

    elif ext == '.bin':
        print("Loading HuggingFace model...")
        return AutoModel.from_pretrained(filepath)

    elif ext == '.onnx':
        print("ONNX models are loaded using inference engines (e.g., onnxruntime, not torch).")
        model = onnx.load(filepath)
        onnx.checker.check_model(model)
        return model

    elif ext == '.pb':
        print("Loading TensorFlow frozen graph...")
        return tf.saved_model.load(filepath)

    elif ext in ['.h5', '.keras']:
        print("Loading Keras model...")
        return keras.models.load_model(filepath)

    elif ext in ['.pkl', '.joblib']:
        print("Loading scikit-learn model...")
        return joblib.load(filepath)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")
