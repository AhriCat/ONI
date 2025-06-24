import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import huggingface_hub
import inspect
from typing import Dict, Any, List, Union, Tuple
import os
import requests
import struct

class DynamicModuleInjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.injected_modules = nn.ModuleDict()
        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_name: str, source: str = "huggingface") -> nn.Module:
        """
        Loads a model from different sources using cached version if possible.
        """
        key = f"{source}:{model_name}"
        if key in self.loaded_models:
           return self.loaded_models[key]
        try:
            if source == "huggingface":
                config = AutoConfig.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, config=config)
                model.to(self.device)
                self.loaded_models[key] = model
                return model
            elif source == "modelgarden":
               try:
                   response = requests.get(f"https://your-modelgarden-api.com/models/{model_name}", timeout=10)
                   response.raise_for_status()
                   model_data = response.json()
                   model = self._load_model_from_model_garden(model_data)
                   model.to(self.device)
                   self.loaded_models[key] = model
                   return model
               except requests.exceptions.RequestException as e:
                  print(f"Error loading model from modelgarden {model_name}: {e}")
                  return None
            elif source == "file":
                if os.path.exists(model_name):
                  model = torch.load(model_name, map_location=self.device)
                  self.loaded_models[key] = model
                  return model
                else:
                  print(f"Could not find model file {model_name}")
                  return None
            elif source == "binary":
                # Placeholder for loading binary from file
                if os.path.exists(model_name):
                    with open(model_name, 'rb') as f:
                        binary_data = f.read()
                    try:
                      model = self._try_load_model(binary_data) # Check if the data is a model first
                      if model:
                          model.to(self.device)
                          self.loaded_models[key] = model
                          return self._create_function_wrapper(model, key)
                      else: # If it's not a model we still return the data as a tensor
                        tensor_data = torch.tensor(list(binary_data), dtype=torch.uint8)
                        return self._process_binary_model(tensor_data)
                    except Exception as e: # If loading the model fails, we process it as binary anyway.
                      tensor_data = torch.tensor(list(binary_data), dtype=torch.uint8)
                      return self._process_binary_model(tensor_data)
                else:
                    print(f"Could not find binary model file {model_name}")
                    return None
            else:
                print(f"Unknown model source {source}")
                return None
        except Exception as e:
            print(f"Error loading model {model_name} from {source}: {e}")
            return None

    def _try_load_model(self, binary_data):
        """
        Trys loading the model based on a binary string, catches any errors and returns None if it can't load it
        """
        try:
            buffer = io.BytesIO(binary_data)
            model = torch.load(buffer, map_location=self.device)
            return model
        except Exception as e:
          return None

    def _load_model_from_model_garden(self, model_data):
            """
            Placeholder for handling the model garden loading process.
            """
            if 'model_path' in model_data:
                model_path = model_data['model_path']
                try:
                  model = torch.load(model_path, map_location=self.device)
                  return model
                except Exception as e:
                    print(f"Error loading model file from modelgarden: {e}")
                    return None
            else:
                print("Model path not found in model garden.")
                return None
    def _create_function_wrapper(self, model: nn.Module, model_name: str):
      """
        Wraps the loaded model with a callable class, for handling different types of model calls.
      """
      class ModelWrapper(nn.Module):
        def __init__(self, model, model_name, device):
          super().__init__()
          self.model = model
          self.device = device
          self.model_name = model_name
        def forward(self, input_data: Dict[str, Any]) -> Union[Any, Tuple[Any, ...]]:
            try:
                with torch.no_grad():
                    # Determine the call type based on model type or input
                    # Currently only tries for generate, then forward, otherwise gives error
                    if hasattr(self.model, 'generate') and input_data:
                      if 'input_ids' in input_data:
                        generated = self.model.generate(**input_data, return_dict=True)
                        if hasattr(generated, 'sequences') and generated.sequences:
                           return generated.sequences
                        else:
                             return generated
                      else:
                           return self.model.generate(**input_data, return_dict=True)
                    else:
                      # Default: forward pass and return the last hidden layer
                      return self.model(**input_data).last_hidden_state
            except Exception as e:
              print(f"Error processing with {self.model_name}: {e}")
              return None
      return ModelWrapper(model, model_name, self.device)

    def _process_binary_model(self, data):
        """
          Handles data as a binary tensor if it is not identified as a model.
        """
        class BinaryWrapper(nn.Module):
            def __init__(self, data):
                super().__init__()
                self.data = data

            def forward(self) -> torch.Tensor:
                return self.data
        return BinaryWrapper(data)
    
    def inject_module(self, model_name: str, module_name: str, source: str = "huggingface"):
        """
        Loads and injects a model as a module, returning the module if successful.
        """
        if module_name in self.injected_modules:
            print(f"module {module_name} is already loaded")
            return self.injected_modules[module_name]
        # Load the external model
        model = self._load_model(model_name, source)
        if model is None:
            return None

        # Register the module
        self.injected_modules[module_name] = model
        return model

    def forward(self, module_name: str, input_data: Dict[str, Any]) -> Union[Any, None]:
      """
      Runs the module with the input data
      """
      if module_name in self.injected_modules:
        module = self.injected_modules[module_name]
        if isinstance(module, nn.Module):
            return module(input_data)
        elif hasattr(module, 'forward'):
          return module.forward()
        else:
           return module
      else:
          print(f"Module {module_name} not found in model")
          return None
