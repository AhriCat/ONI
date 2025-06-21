import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from collections import deque
import random
import threading
import time
import re  # Needed for regex in animate method
import pytesseract  # For OCR
import cv2  # For computer vision
from PIL import ImageGrab  # For screen capture
import pyautogui  # For mouse and keyboard automation
import pyaudio  # For audio processing
from selenium import webdriver  # For browser automation
from selenium.webdriver.common.by import By  # For locating elements
import matplotlib.pyplot as plt  # For plotting images
import PyPDF2  # Ensure that PDF reading works
import torch.autograd as autograd  # If used elsewhere in the code
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers.image_utils import load_image
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import modules with error handling
try:
    from modules import file_preprocessor
except ImportError:
    logger.warning("file_preprocessor not found, using fallback")
    class file_preprocessor:
        @staticmethod
        def process_file(path):
            return ""

try:
    from modules import oni_vision as vision
except ImportError:
    logger.warning("oni_vision not found, using fallback")
    class vision:
        @staticmethod
        def process_input(*args, **kwargs):
            return torch.zeros(1, 896), 0.0

try:
    from modules import oni_audio as audio
except ImportError:
    logger.warning("oni_audio not found, using fallback")
    class audio:
        @staticmethod
        def forward(*args, **kwargs):
            return torch.zeros(1, 896), 0.0

try:
    from modules import oni_MM_attention as MultiModalAttention
except ImportError:
    logger.warning("oni_MM_attention not found, using fallback")
    class MultiModalAttention:
        def __init__(self, dim):
            self.dim = dim
        def forward(self, x1, x2, x3):
            return x1

try:
    from modules import oni_memory as memory 
except ImportError:
    logger.warning("oni_memory not found, using fallback")
    class memory:
        @staticmethod
        def update_memory(*args, **kwargs):
            pass

try:
    from modules import oni_netmonitor as netmon
except ImportError:
    logger.warning("oni_netmonitor not found, using fallback")
    class netmon:
        @staticmethod
        def start():
            pass

try:
    from modules import oni_portscanner as ps
except ImportError:
    logger.warning("oni_portscanner not found, using fallback")
    class ps:
        @staticmethod
        def start_scan():
            pass

try:
    from modules import oni_executive_function as exec_func
except ImportError:
    logger.warning("oni_executive_function not found, using fallback")
    class exec_func:
        @staticmethod
        def execute(*args, **kwargs):
            pass

try:
    from modules import oni_metacognition as MetaCognition
except ImportError:
    logger.warning("oni_metacognition not found, using fallback")
    class MetaCognition:
        def __init__(self, dim):
            pass
        def forward(self, x):
            return x, 0.5, []

try:
    from modules import oni_homeostasis as HomeostaticController
except ImportError:
    logger.warning("oni_homeostasis not found, using fallback")
    class HomeostaticController:
        def __init__(self, input_dim, hidden_dim):
            pass
        def forward(self, x, state):
            return x, 0.0

try:
    from modules.oni_NLP import OptimizedNLPModule as nlp_module
except ImportError:
    logger.warning("oni_NLP not found, using fallback")
    class nlp_module:
        def __init__(self, config):
            self.embedding = nn.Embedding(1000, 896)
        def forward(self, x, y=None):
            return x, 0.0
        def identify_tasks(self, text):
            return []
        def generate(self, text):
            return "Generated response"

# Import tools with error handling
try:
    from tools.calculator import Calculator
    calculator = Calculator()
except ImportError:
    logger.warning("Calculator not found, using fallback")
    class Calculator:
        def calculate(self, expr):
            return "Calculation not available"
    calculator = Calculator()

try:
    from tools.drawer import pipe
except ImportError:
    logger.warning("Drawer not found, using fallback")
    def pipe(*args, **kwargs):
        return type('obj', (object,), {'images': [None]})()

try:
    from tools.animator import pipeline
except ImportError:
    logger.warning("Animator not found, using fallback")
    def pipeline(*args, **kwargs):
        return type('obj', (object,), {'frames': [[]]})()

# Import external models with error handling
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
    
    # Try to load Qwen model
    try:
        tokenizerQ = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        logger.warning(f"Failed to load Qwen model: {e}")
        tokenizerQ = None
        qwen = None
    
    # Try to load vision model
    try:
        processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct")
        model = AutoModelForImageTextToText.from_pretrained("microsoft/Phi-3.5-vision-instruct", torch_dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        logger.warning(f"Failed to load vision model: {e}")
        processor = None
        model = None
        
except ImportError:
    logger.warning("Transformers not available, using fallbacks")
    tokenizerQ = None
    qwen = None
    processor = None
    model = None

# Import trading module with error handling
try:
    from modules.oni_crypto_trade import TradingBot
    trader = TradingBot
except ImportError:
    logger.warning("Trading module not found, using fallback")
    class TradingBot:
        def __init__(self, *args, **kwargs):
            pass
    trader = TradingBot

# Import tokenizer with error handling
try:
    from modules.oni_Tokenizer import MultitokenBPETokenizer
    tokenizer = MultitokenBPETokenizer()
except ImportError:
    logger.warning("Tokenizer not found, using fallback")
    class MultitokenBPETokenizer:
        def encode(self, text):
            return [1, 2, 3]
        def decode(self, ids):
            return "decoded text"
    tokenizer = MultitokenBPETokenizer()

# Fallback implementations for missing components
class DynamicModuleInjector:
    def inject_module(self, *args, **kwargs):
        return None
    def forward(self, *args, **kwargs):
        return None

class DynamicSynapse(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

class EnergyBasedSynapse(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x), 0.0

class SparseFocusedGroupAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
    def forward(self, x, hidden=None):
        out, _ = self.attention(x, x, x)
        return out, hidden

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
    def run_visible(self, x):
        return x

# Initialize fallback components
vision_system = vision
controller = HomeostaticController

# Define the Experience Replay Buffer
class ExperienceReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Define the Recurrent Q-Network
class RecurrentQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=896, num_layers=20):
        super(RecurrentQNetwork, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_size)
    
    def forward(self, state, hidden=None):
        # state: (batch_size, seq_len, state_size)
        lstm_out, hidden = self.lstm(state, hidden)  # lstm_out: (batch_size, seq_len, hidden_dim)
        q_values = self.fc(lstm_out[:, -1, :])  # Take the output of the last time step
        return q_values, hidden

class OniMicro(nn.Module):
    def __init__(self, tokenizer, input_dim, hidden_dim, output_dim, nhead, num_layers, exec_func, state_size, action_size, learning_rate=0.002, discount_factor=0.98, exploration_rate=1.0, exploration_decay=0.995, target_update=10, device=device):
        super(OniMicro, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.running = False
        self.screen_thread = None
        self.audio_thread = None
        self.audio_stream = None
        self.pyaudio_instance = None
        self.tokenizer = tokenizer
        self.embedding = nlp_module.embedding if hasattr(nlp_module, 'embedding') else nn.Embedding(1000, 896)
        self.attention = SparseFocusedGroupAttention(hidden_dim)
        self.memory = memory 
        self.hm = self.share_memory if hasattr(self, 'share_memory') else lambda: None
        self.memnet = lambda: None  # finder.use_hopfield_network
        self.findpattern = lambda: None  # finder.find_patterns
        self.rbm = RBM(num_visible=256, num_hidden=64)
        self.use_rbm = False
        self.dynamic_layer = EnergyBasedSynapse(input_dim, output_dim)
        self.ed_layer = DynamicSynapse(input_dim, output_dim)
        self.nlp_module = nlp_module
        self.controller = controller 
        self.vision_module = vision_system
        self.visionprocessor = processor
        self.vision_to_text = model
        self.audio_module = audio  # MiniAudioModule(1,896,896,128,1000000)
        self.multi_modal_attention = MultiModalAttention(hidden_dim)
        self.multi_modal_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.homeostatic_controller = HomeostaticController(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.exec_func = exec_func  # Set exec_func during initialization

        self.codeTokenizer = tokenizerQ
        self.coder = qwen
        self.drawer = pipe
        self.animator = pipeline
        self.calculator = calculator
        #self.yt_downloader = yt_downloader
        self.trader = trader 
        self.browser = webdriver.Chrome() if 'webdriver' in globals() else None
        self.networkmonitor = netmon
        self.portscanner = ps
        #self.nmap = nm
        self.injector = DynamicModuleInjector()


        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Q-Networks
        self.q_network = self.safe_init(lambda: RecurrentQNetwork(state_size, action_size, hidden_dim).to(self.device), "Q-Network")
        self.target_q_network = self.safe_init(lambda: RecurrentQNetwork(state_size, action_size, hidden_dim).to(self.device), "target Q-Network")
        if self.q_network is not None:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_q_network.eval()

        # Optimizer and Loss
        self.optimizer = self.safe_init(lambda: optim.Adam(self.q_network.parameters(), lr=learning_rate), "optimizer")
        self.criterion = self.safe_init(lambda: nn.MSELoss(), "loss function")

        # Experience Replay Buffer
        self.memory_buffer = self.safe_init(lambda: ExperienceReplayBuffer(buffer_size=10000, batch_size=64), "memory buffer")

        # RL Parameters
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.target_update = target_update
        self.step_count = 0

        # Hidden State
        self.hidden_state = None

        # Lock for thread safety
        self.lock = self.safe_init(lambda: threading.Lock(), "threading lock")

    def safe_init(self, init_fn, component_name):
        """Safely initialize components to prevent errors."""
        try:
            return init_fn()
        except Exception as e:
            print(f"Error initializing {component_name}: {e}")
            return None

    def forward(self, x_nlp_src, x_nlp_tgt=None, x_vision=None, x_audio=None, system_state=None, hidden=None):
        try:
            if self.use_rbm:
                x = self.rbm.run_visible(x_nlp_src)
            else:
                x = self.output_layer(x_nlp_src)
        except Exception as e:
            print(f"Error in RBM or output layer: {e}")
            x = x_nlp_src  # Default to using the NLP input as the output layer if error

        try:
            # Apply RecurrentSelfAttention
            x, hidden = self.attention(x, hidden)
        except Exception as e:
            print(f"Error in attention mechanism: {e}")
            pass  # Skip attention if not available

        try:
            nlp_output, nlp_energy = self.nlp_module(x, x_nlp_tgt)
        except Exception as e:
            print(f"Error in NLP module: {e}")
            nlp_output, nlp_energy = x, 0.0  # Default to x if NLP fails

        try:
            vision_output, vision_energy = self.vision_module(x_vision)
        except Exception as e:
            print(f"Error in Vision module: {e}")
            vision_output, vision_energy = torch.zeros_like(x), 0.0  # Default if vision fails

        try:
            audio_output, audio_energy = self.audio_module(x_audio)
        except Exception as e:
            print(f"Error in Audio module: {e}")
            audio_output, audio_energy = torch.zeros_like(x), 0.0  # Default if audio fails

        try:
            # Multi-modal attention
            attended_output = self.multi_modal_attention(nlp_output, vision_output, audio_output)
        except Exception as e:
            print(f"Error in Multi-modal attention: {e}")
            attended_output = torch.cat((nlp_output, vision_output, audio_output), dim=1)  # Concatenate raw outputs if attention fails

        try:
            # Fuse the combined outputs from different modalities
            combined_output = torch.cat((attended_output, nlp_output, vision_output, audio_output), dim=1)
            fused_output = self.multi_modal_fusion(combined_output)
        except Exception as e:
            print(f"Error in Multi-modal fusion: {e}")
            fused_output = combined_output  # Default to combined output if fusion fails

        try:
            # Adjust the fused output using the homeostatic controller
            adjusted_output, homeostatic_energy = self.homeostatic_controller(fused_output, system_state)
        except Exception as e:
            print(f"Error in Homeostatic Controller: {e}")
            adjusted_output, homeostatic_energy = fused_output, 0.0  # Default if homeostatic controller fails

        try:
            # Final output layer
            final_output = self.output_layer(adjusted_output)
        except Exception as e:
            print(f"Error in final output layer: {e}")
            final_output = adjusted_output  # Default to adjusted output if output layer fails

        # Calculate total energy
        total_energy = nlp_energy + vision_energy + audio_energy + homeostatic_energy

        try:
            # Make a decision based on the final output and total energy
            decision = self.make_decision_based_on_output(final_output, total_energy)
        except Exception as e:
            print(f"Error in decision-making process: {e}")
            decision = "default decision"  # Default decision if decision-making fails

        return final_output, total_energy, decision, hidden

    # Additional functions like `get_state`, `update_environment`, `train_dqn`, `train_rl`, etc. would remain the same.
    def perform_task(self, input_text):
        """
        Performs the task by inferring it from the input text using the NLP module.
        If a valid task is found, executes it agentically.
        """
        # First, let the nlp_module find potential tasks in the input text
        try:
            task_list = self.nlp_module.identify_tasks(input_text)
        except Exception as e:
            print(f"Error using nlp_module to find tasks: {e}")
            return

        if not task_list:
            print(f"No tasks were found in the input: {input_text}")
            return

        # If tasks are found, attempt to infer and execute them
        for task in task_list:
            print(f"Inferred task: {task}")
            self.execute_task(task)
    
    def scan_files_for_models(self, directory: str):
        """
        Scans a directory for model files.
        """
        model_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".pt", ".pth", ".bin")):
                    model_path = os.path.join(root, file)
                    model_paths.append(model_path)
        return model_paths
    
    def subsume(self, model_file, module_name):
        model_directory = "/PATH" # Update this with your local path
        model_files = self.scan_files_for_models(model_directory)


        for model_file in model_files:
           #  Get the model name without the extension
            module_name = os.path.splitext(os.path.basename(model_file))[0]

           # Load models
            module = self.injector.inject_module(model_file, module_name, source="file")
            if module:
              # Example usage
                input_data = {'input_ids': torch.randint(0, 100, (1, 10)).to(self.injector.device)}
                output = self.injector.forward(module_name, input_data)
                if output is not None:
                  print(f"Successfully injected {module_name}")
                  if isinstance(output, torch.Tensor):
                      print(output.shape)
                  else:
                     print(output)

        # Inject a model from huggingface
 
        module = self.injector.inject_module(model_name, module_name)
        if module:
              # Example usage
             if hasattr(module, 'tokenizer'):
                  encoding = module.tokenizer("Hello world", return_tensors='pt')
                  input_data = encoding
             else:
                   input_data = {'input_ids': torch.randint(0, 100, (1, 10)).to(injector.device)}
             output = self.injector.forward(module_name, input_data)
             if output is not None:
                  print(f"Successfully injected {module_name}")
                  if isinstance(output, torch.Tensor):
                      print(output.shape)
                  else:
                     print(output)


    def execute_task(self, task_description):
        """
        Executes a specific task based on the description inferred from the input text.
        """
        task_map = {
            '/search': lambda: self.perform_search(task_description.split('/search')[-1].strip()),
            'read': lambda: self.read_pdf(self.browser.current_url),
            'select first link': lambda: self.select_link(1),
            'select second link': lambda: self.select_link(2),
            'select third link': lambda: self.select_link(3),
            'draw a picture of' or '/draw': lambda: self.draw_image(task_description.split('draw a picture of' or '/draw')[-1].strip()),
            '/animate': lambda: self.animate(task_description.split('/animate')[-1].strip()),
            'open dashboard': lambda: self.open_dash(),
            '/exit': lambda: self.close_browser(),
            'watch': lambda: self.vision_module(screen_output),
            'listen': lambda: self.process_audio_feed(),
            'help': lambda: self.show_help(),
            'math': lambda: print(self.calculator.calculate(task_description)),
            'code': lambda: self.code_inference(task_description),
            'monitor': lambda: self.start_network_monitor()

            # Add additional task mappings as needed.
        }

        # Infer the task from the task description and execute the corresponding function
        for key, task_fn in task_map.items():
            if key in task_description.lower():
                try:
                    task_fn()
                    print(f"Task '{key}' executed successfully.")
                except Exception as e:
                    print(f"Error performing {key} task: {e}")
                return

        print(f"Unknown or unsupported task: {task_description}")
    def start_network_monitor(self):
        self.networkmonitor.start
        self.networkmonitor.display_network_usage

    def stop_network_monitor(self):
        self.networkmonitor.stop

    def code_inference(self, text):
        prompt = text
        messages = [
            {"role": "system", "content": "You are ONI, an advanced modular AGI system designed for flexible and intelligent interaction. The Oni system is a modular, multi-AI-driven architecture designed to create a versatile, highly adaptive artificial general intelligence (AGI) framework and concatenated model. It integrates multiple autonomous agents, each optimized for specific tasks like natural language understanding, vision processing, and decision-making, while enabling seamless inter-agent collaboration. Built with scalability in mind, Oni supports dynamic task execution, multitasking, and real-time learning. Its control interface allows users to issue high-level commands that translate into structured, multi-step tasks managed by the system. With a strong focus on modularity and extensibility, Oni is designed for robust functionality across diverse domains while maintaining central control for user oversight and adaptability."},
            {"role": "user", "content": prompt}
        ]
        text = self.codeTokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.codeTokenizer([text], return_tensors="pt").to(qwen.device)

        generated_ids = self.coder.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        outputs = self.codeTokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return outputs
    
    def get_state(self, nlp_output=None, vision_output=None, audio_output=None, other_module_states=None):
        # Concatenate outputs from different modules to form the state
        state = np.concatenate([nlp_output, vision_output, audio_output])
        
        # If other module states are provided, include them in the state
        if other_module_states is not None:
            state = np.concatenate([state, other_module_states])
        
        return state
    
    def update_environment(self, new_position, room_data=None):
        """
        Updates Oni's position and loads new room data if a new room is entered.
        
        Args:
            new_position (tuple): New (x, y) coordinates.
            room_data (any, optional): Data for the new room.
        """
        entered_new_room = self.memory.spatial_memory.update_position(new_position)  # Changed to self.memory.spatial_memory
        if entered_new_room and room_data is not None:
            room_key = self.memory.spatial_memory.get_current_room_key()  # Changed to self.memory.spatial_memory
            self.memory.spatial_memory.load_room(room_key, room_data)  # Changed to self.memory.spatial_memory
            # Assuming update_heuristics and explore_next_room are methods in memory or OniMicro
            self.memory.update_heuristics()  # Or self.update_heuristics() if defined in OniMicro
            self.memory.explore_next_room()  # Or self.explore_next_room() if defined in OniMicro

    def get_local_state(self):
        """
        Retrieves the current localized state from the Spatial Memory Module.
        
        Returns:
            any: Current room data.
        """
        return self.memory.spatial_memory.get_current_room_data()  # Changed to self.memory.spatial_memory
    
    def choose_action(self, state: np.ndarray, hidden=None):
        if np.random.rand() < self.exploration_rate:
            # Explore: choose a random action
            return np.random.randint(self.action_size), hidden
        else:
            # Exploit: choose the action with max Q-value
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, state_size)
            with torch.no_grad():
                q_values, hidden = self.q_network(state_tensor, hidden)
            action = torch.argmax(q_values, dim=1).item()
            return action, hidden
    
    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def train_dqn(self):
        if len(self.memory_buffer) < self.memory_buffer.batch_size:
            return  # Not enough samples to train
        
        # Sample a batch of experiences
        experiences = self.memory_buffer.sample()
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Current Q-values
        q_values, _ = self.q_network(states.unsqueeze(1))  # (batch_size, action_size)
        current_q_values = q_values.gather(1, actions)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_q_network(next_states.unsqueeze(1))
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.discount_factor * max_next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
    
    def train_rl(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        self.memory_buffer.add((state, action, reward, next_state, done))
        
        # Perform a training step
        self.train_dqn()
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
    
    def make_decision_based_on_output(self, output, energy):
        if energy < 0.5:
            return "low energy state, maintain current behavior"
        elif output.mean() > 0:
            return "take action based on positive output"
        else:
            return "recalibrate based on negative output"
    
    def generate_response(self, input_text: str, hidden=None):
        # Tokenize the input text
        tokenized_input = self.tokenizer.encode(input_text)
        
        # Convert tokenized input to numpy array and ensure it is float32
        tokenized_input_np = np.array(tokenized_input).astype(np.float32)
        
        # Process input through the RBM if use_rbm is True
        if self.use_rbm:
            processed_input = self.rbm.run_visible(tokenized_input_np)
        else:
            processed_input = tokenized_input_np
        
        # Get the current state (implement your own state extraction)
        current_state = self.get_state(nlp_output=processed_input, vision_output=None, audio_output=None, other_module_states=None)
        
        # Choose an action
        action, hidden = self.choose_action(current_state, hidden)
        
        # Execute the action (implement your own action execution)
        self.execute_task(input_text)
        
        # Observe the outcome and reward (implement your own observation)
        reward, next_state, done = self.q_network(self.nlp_module.process_raw_input(self.vision_module.process_input(input_type='screen')))
        
        # Update the RL agent
        self.train_rl(current_state, action, reward, next_state, done)
        
        # Decode the output to text (implement your own decoding)
        # 'output' was undefined; assuming 'final_output' is the output from the forward method
        # However, 'final_output' is not returned by this method; needs to be captured
        # Here's a possible fix:
        # Generate response via forward method
        # Assuming system_state is available
        # Placeholder system_state
        system_state = torch.zeros(1).to(self.device)  # Adjust based on actual system_state representation
        
        # Call the forward method
        final_output, total_energy, decision, hidden = self.forward(
            x_nlp_src=processed_input, 
            x_nlp_tgt=None, 
            x_vision=None, 
            x_audio=None, 
            system_state=system_state, 
            hidden=hidden
        )
        
        decoded_output = self.tokenizer.decode(torch.argmax(final_output, dim=-1).cpu().numpy())
        
        return decoded_output, hidden

    def call_and_response(self, conversation_history):
        hidden = None
        for turn in conversation_history:
            response, hidden = self.generate_response(turn, hidden)
            # Here you might want to append the response to the conversation history
            # or handle it in some other way depending on your use case
        return response
    
    def make_decision(self, input_text):
        command = self.choose_command(input_text)
        if command:
            getattr(self, command)(input_text)

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.process_audio(audio_data)
        return (in_data, pyaudio.paContinue) 
    
    def process_audio(self, audio_data):
        audio_data, _ = self.audio(audio_data)  # Changed to self.audio and unpacked energy
        return audio_data

    def capture_screen(self):
        # Capture the screen
        screen = ImageGrab.grab(all_screens=True)
        
        # Convert to numpy array
        screen_np = np.array(screen)
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        
        # Convert to PyTorch tensor and normalize
        screen_tensor = torch.from_numpy(screen_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        self.vision_module(screen_tensor)

        return screen_tensor
    def process_image_url(self, input_prompt, url):
        prompt = input_prompt
        url = url
        image = load_image(url)
        model_inputs = self.visionprocessor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.vision_to_text.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded
    
    def screen_to_text(self, input_prompt):
        screen = ImageGrab.grab(all_screens=True)
        prompt = "You are part of Oni.Vision (the vision part of a robust AGI system) if this prompt isn't continued assume the system is seeking a summary of the screen, otherwise respond to this user and what you see on the screen in tandem:" + input_prompt
        model_inputs = self.visionprocessor(text=prompt, images=screen, return_tensors="pt").to(torch.bfloat16).to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.vision_to_text.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.visionprocessor.decode(generation, skip_special_tokens=True) 
        return decoded

    def process_display_feed(self):
        self.running = True
        while self.running:
            # Capture the screen
            screen = ImageGrab.grab(all_screens=True)
            
            # Convert to numpy array
            screen_np = np.array(screen)
            
            # Convert from RGB to BGR (OpenCV uses BGR)
            screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            
            # Convert to PyTorch tensor and normalize
            screen_tensor = torch.from_numpy(screen_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            self.vision_module(screen_tensor)  # Changed to self.vision_module
            
            # Add a delay to control the frame rate if needed
            time.sleep(0.1)  # Adjust the sleep time as necessary

    def start_processing_feed(self):
        if not hasattr(self, 'running') or not self.running:
            self.screen_thread = threading.Thread(target=self.process_display_feed)
            self.screen_thread.start()
    
    def stop_processing_feed(self):
        self.running = False
        if hasattr(self, 'screen_thread') and self.screen_thread.is_alive():
            self.screen_thread.join()

    def process_cameras(self, camera1=0, camera2=1):
        # Initialize cameras
        cap1 = cv2.VideoCapture(camera1)
        cap2 = cv2.VideoCapture(camera2)

        if not (cap1.isOpened() and cap2.isOpened()):
            print("Error: One or both cameras could not be opened.")
            return

        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Error: Could not read frames from both cameras.")
            return

        # Convert frames to grayscale (if needed for processing)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Create stereo vision object (stereoBM is a common choice, but others exist like stereoSGBM)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

        # Compute disparity (depth map)
        disparity = stereo.compute(gray1, gray2)

        # Feed disparity map into the vision module
        self.vision_module(disparity)
    
    def start_audio_stream(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(format=pyaudio.paFloat32,
                                                       channels=1,
                                                       rate=44100,
                                                       input=True,
                                                       stream_callback=self.audio_callback)
        self.audio_stream.start_stream()
    
    def stop_audio_stream(self):
        if hasattr(self, 'audio_stream') and self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance is not None:
            self.pyaudio_instance.terminate()
    
    def process_audio_feed(self):
        self.running_audio = True
        self.start_audio_stream()
        while self.running_audio:
            time.sleep(0.1)  # Keep the thread alive
    
    def start_processing_audio(self):
        if not hasattr(self, 'running_audio') or not self.running_audio:
            self.audio_thread = threading.Thread(target=self.process_audio_feed)
            self.audio_thread.start()
    
    def stop_processing_audio(self):
        self.running_audio = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join()
        self.stop_audio_stream()
    
    def use_mouse_and_keyboard(self, actions):
        # Example actions using the mouse and keyboard
        if isinstance(actions, str):
            pyautogui.typewrite(actions)  # Corrected to type the whole string, instead of character by character
        else:
            for action in actions:
                if action['type'] == 'move':
                    pyautogui.moveTo(action['x'], action['y'])  # Move the mouse to a specific position
                elif action['type'] == 'click':
                    pyautogui.click()  # Click the mouse
    
    def read_pdf(self, pdf_path):
        # Extract text from a PDF file
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            for page_num in range(reader.numPages):
                text += reader.getPage(page_num).extractText()
        return text
    
    def read_code(self, code_path):
        # Read and return the content of a code file
        with open(code_path, "r") as file:
            code = file.read()
            thoughts = self.code_inference(code)
        return code, thoughts
    
    def reward_function(self, audio_signal):
        # Check if the audio signal contains a musical tone
        # Dummy implementation: Check if the mean of the audio signal exceeds a threshold
        if audio_signal.mean() > 0.5:
            return 1  # Positive reinforcement
        else:
            return 0  # No reinforcement
    
    def open_browser(self):
        self.browser = webdriver.Chrome()  # or any other browser driver
        self.browser.get('https://www.google.com')
    
    def initialize_renderer(self):
        self.renderer.start_rendering()
    
    def shutdown_renderer(self):
        self.renderer.stop_rendering()
    
    def render_current_room(self):
        current_room_data = self.memory.spatial_memory.get_current_room_data()  # Changed to self.memory.spatial_memory
        if current_room_data:
            self.renderer.update_room(current_room_data)
    
    def animate(self, prompt):
        # Determine the output format
        format_match = re.search(r'\b(mp4|gif)\b', prompt.lower())
        output_format = format_match.group(1) if format_match else 'gif'
        
        # Generate the animation
        output = pipeline(prompt=prompt, guidance_scale=1.0, num_inference_steps=50)  # Assuming 'step' is 50
        
        # Export to the specified format
        if output_format == 'mp4':
            output_file = "animation.mp4"
            export_to_video(output.frames[0], output_file)
        else:
            output_file = "animation.gif"
            export_to_gif(output.frames[0], output_file)
        
        # Display the output in chat
        if output_format == 'gif':
            display(Image(filename=output_file))
        else:
            print(f"MP4 file saved as {output_file}")
        
        return output_file

    def perform_search(self, search_query):
        try:
            search_box = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.NAME, 'q'))  # Assuming 'q' is the name attribute of the search box
            )
            search_box.clear()
            search_box.send_keys(search_query)
            pyautogui.press('enter')  # Corrected to simulate pressing 'Enter' instead of typing it literally
        except Exception as e:
            print(f"Error performing search: {e}")
            self.fallback_search(search_query)


    def fallback_search(self, search_query):
        pyautogui.typewrite(search_query)
        time.sleep(1)
        pyautogui.press('enter')
        
    def select_link(self, link_number):
        try:
            link_xpath = f"(//h3)[{link_number}]"
            link = WebDriverWait(self.browser, 10).until(
                EC.element_to_be_clickable((By.XPATH, link_xpath))
            )
            link.click()
        except Exception as e:
            print(f"Error selecting the link: {e}")
            # As a fallback, you might consider presenting an error or alternative action


    def draw_image(self, prompt):
        try:
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]

            plt.imshow(image)
            plt.axis('off')
            plt.show()

            save_image = input("Do you want to save this image? (yes/no): ").strip().lower()
            if save_image == 'yes':
                image.save("flux-dev.png")
                print("Image saved as 'flux-dev.png'.")
            else:
                print("Image was not saved.")
        except Exception as e:
            print(f"An error occurred while drawing: {e}")

    def open_dash(self):
        app.run_server(debug=True)  # Assuming 'app' is defined elsewhere

    def show_help(self):
        help_items = [
            'search', 'find', 'draw', 'play', 'trade', 'buy', 'sell', 'connect exchange',
            'open browser', 'exit browser', 'research', 'fact check', 'calculate', 'monitor', 'advise'
        ]
        print(f"The things I can do are {help_items}")
            
    def screenshot_and_read(self):
        screenshot = self.browser.get_screenshot_as_png()
        text = pytesseract.image_to_string(screenshot)
        return text

    def close_browser(self):
        if hasattr(self, 'browser') and self.browser:
            self.browser.quit()
            self.browser = None

    def choose_command(self, input_text):
        # Generate response from the NLP module
        response = self.nlp_module.identify_tasks(input_text)  # Assumed to be defined elsewhere
        
        # Map NLP response to command
        command_map = {
            "capture screen": "capture_screen",
            "use mouse and keyboard": "use_mouse_and_keyboard",
            "read pdf": "read_pdf",
            "read code": "read_code",
            "good job": "reward_function",
            "use browser": "open_browser",
            "search": "perform_web_task",
            "close": "close_browser"
        }
        
        # Extract command from the response
        for command in command_map:
            if command in response:
                return command_map[command]
        return None

    def make_decision(self, input_text):
        command = self.choose_command(input_text)
        if command:
            getattr(self, command)(input_text)

    def safe_process(self, input_text):
        """
        Safely process the input using the primary NLP module, with fallback to the coding agent.
        If any errors arise (e.g., too many indices, expected outputs, etc.), use the coder to generate the response.

        Args:
            input_text (str): The input text to process.

        Returns:
            str: The processed response or fallback response from the coding agent.
        """
        try:
            # Attempt processing with the primary NLP
            response = self.nlp_module.generate(input_text)
            
            # Validate the response (if needed, e.g., check format, expected structure)
            if not self.validate_response(response):
                raise ValueError("Primary NLP response is invalid.")

            return response

        except (IndexError, ValueError, KeyError) as e:
            # Catch specific errors and fallback to the coding agent
            print(f"Error in primary NLP module: {e}")
            print("Falling back to the coding agent...")
            return self.fallback_with_coder(input_text)

        except Exception as e:
            # Generic error handling
            print(f"Unexpected error: {e}")
            print("Falling back to the coding agent...")
            return self.fallback_with_coder(input_text)

    def validate_response(self, response):
        """
        Validates the response from the primary NLP module.
        Placeholder for custom validation logic.

        Args:
            response (any): The response to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Add custom validation logic (e.g., check format, structure)
        if response is None or not isinstance(response, str) or len(response) == 0:
            return False
        return True

    def fallback_with_coder(self, input_text):
        """
        Generate a response using the coder when the primary NLP fails.

        Args:
            input_text (str): The input text for fallback processing.

        Returns:
            str: The response generated by the coder.
        """
        try:
            # Format the input for the coding agent
            formatted_input = f"Process this input: {input_text}"

            # Generate response using the coding agent
            model_inputs = self.codeTokenizer([formatted_input], return_tensors="pt").to(self.coder.device)

            generated_ids = self.coder.generate(
                **model_inputs,
                max_new_tokens=512,
            )

            generated_response = self.codeTokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print("Generated response from coder:", generated_response)
            return generated_response

        except Exception as coder_error:
            print(f"Error in fallback with coder: {coder_error}")
            return "Failed to generate a response. Please try again later."


    def evaluate_step(step):
        """
        Evaluates a generated reasoning step.
        
        Args:
            step (str): The reasoning step to evaluate.
        
        Returns:
            evaluation (dict): Evaluation results, including:
                - "is_valid" (bool): Whether the step is valid.
                - "is_final" (bool): Whether the step is final.
                - "feedback" (str): Feedback to guide the next iteration.
        """
        # Basic evaluation logic (expand with testing, constraints, etc.)
        is_valid = "error" not in step.lower()  # Example: Check if errors are mentioned
        is_final = "solution" in step.lower()  # Example: Check if a solution is reached
        
        feedback = "Refine the approach" if not is_final else None
        return {"is_valid": is_valid, "is_final": is_final, "feedback": feedback}
    def update_prompt(current_prompt, next_step, feedback):
        """
        Updates the prompt for the next iteration.
        
        Args:
            current_prompt (str): The current prompt.
            next_step (str): The newly generated step.
            feedback (str): Feedback to refine reasoning.
        
        Returns:
            updated_prompt (str): The updated prompt.
        """
        return f"{current_prompt}\nNext Step: {next_step}\nFeedback: {feedback}"