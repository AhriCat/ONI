import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytesseract  # For OCR, ensure pytesseract is installed
import numpy as np
import cv2 
import PIL
import PIL.ImageGrab
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
""" original contains locked pretrained vision to text but the model supports it if trained"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MiniVisionTransformer(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, output_dim=256, exec_func=None, num_transformer_layers=6, nhead=8, num_classes=1000):
        super(MiniVisionTransformer, self).__init__()

        self.exec_func = exec_func
        self.pointer = 0  # Pointer for line-by-line reading

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            ConvBlock(input_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()

        # Linear projection for Transformer input
        self.linear_proj = nn.Linear(256, hidden_dim)

        # Transformer layers for temporal/spatial encoding
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderLayer(hidden_dim, nhead) for _ in range(num_transformer_layers)]
        )

        # Adaptation layer for final output
        self.adaption_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

        # OCR layer for text extraction
        self.ocr_layer = nn.Linear(output_dim, output_dim)  # Adjust as needed for OCR

        # Classification head for image/frame classification
        self.classifier = nn.Linear(output_dim, num_classes)

        # LSTM for video feed temporal processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def auto_select_mode(self, x):
        """
        Automatically selects mode based on the input:
        - 'image': Single image or batch of images.
        - 'video': Sequence of frames (video).
        - 'ocr': Text-containing images.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Simple heuristic for mode selection:
        if x.dim() == 4:  # (batch_size, channels, height, width)
            if x.size(1) == 3:  # RGB images, likely image or video
                if x.size(0) > 1:  # Batch of frames, likely video
                    return 'video'
                else:  # Single image
                    return 'image'
            elif x.size(1) == 1:  # Grayscale images, likely for OCR
                return 'ocr'
        elif x.dim() == 5:  # (batch_size, seq_len, channels, height, width)
            return 'video'
        else:
            raise ValueError("Unsupported input dimensions for auto mode selection")

    def forward(self, x):
        """
        Automatically processes input based on auto-selected mode.
        """
        if x is None:
            raise ValueError("Input tensor is None")

        print(f"Input shape: {x.shape}")  # Debugging information

        try:
            # Auto-select mode based on input
            mode = self.auto_select_mode(x)
            print(f"Auto-selected mode: {mode}")

            if self.exec_func is not None:
                x = self.exec_func(x)
                print(f"After exec_func layers shape: {x.shape}")  # Debugging information

            x = self.conv_layers(x)
            print(f"After initial conv layers shape: {x.shape}")  # Debugging information

            x = self.flatten(x)
            print(f"Flattened shape: {x.shape}")  # Debugging information

            x = self.linear_proj(x)
            print(f"Shape after linear projection: {x.shape}")  # Debugging information

            # Transformer processing
            if mode == 'video':
                x = rearrange(x, 'b (seq d) -> b seq d', seq=1)
                x, _ = self.lstm(x)  # LSTM for video feed
                x = x[:, -1, :]  # Use the last LSTM output
            else:
                x = rearrange(x, 'b d -> 1 b d')
                x = self.transformer_encoder(x)
                x = rearrange(x, '1 b d -> b d')

            # Process according to mode
            if mode == 'image':
                x = self.adaption_layer(x)
                classification = self.classifier(x)
                energy = torch.sum(classification ** 2)
                return classification, energy

            elif mode == 'video':
                x = self.adaption_layer(x)
                classification = self.classifier(x)
                energy = torch.sum(classification ** 2)
                return classification, energy

            elif mode == 'ocr':
                # OCR layer for text extraction
                ocr_output = self.ocr_layer(x)
                ocr_text = self.extract_text_from_image(ocr_output)
                return ocr_text

            else:
                raise ValueError("Unsupported mode: Choose between 'image', 'video', or 'ocr'")

        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise

    def extract_text_from_image(self, image_tensor):
        """
        Extract text from an image tensor using pytesseract.
        Assumes image_tensor has been processed into an appropriate format for OCR.
        """
        # Convert tensor to a NumPy array suitable for pytesseract
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.squeeze().detach().cpu().numpy()

            # Assuming image_np is a valid grayscale image for OCR
            if image_np.ndim == 2:  # Grayscale
                text = pytesseract.image_to_string(image_np)
                return text
            else:
                raise ValueError("Input tensor must be a 2D grayscale image for OCR extraction.")
        else:
            raise TypeError("Input must be a torch.Tensor for OCR extraction.")

    def set_pointer(self, pointer_position):
        """
        Set the pointer to a specific line in the image for OCR processing.
        """
        self.pointer = pointer_position

    def focus_line_by_line(self, image_tensor):
        """
        Extract text line by line using the pointer mechanism.
        Handles any color of text by converting it to a high-contrast format.
        """
        # Convert tensor to NumPy for OCR
        image_np = image_tensor.squeeze().detach().cpu().numpy()

        # Convert the image to a suitable format for OCR (e.g., grayscale)
        if image_np.ndim == 3:  # Color image
            # Convert to grayscale using OpenCV
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply adaptive thresholding to enhance text visibility
        image_np = cv2.adaptiveThreshold(
            image_np, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Use pytesseract to get bounding box data for each line
        data = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)

        # Extract lines based on the level (line level is usually 5 in pytesseract)
        lines = [data['text'][i] for i in range(len(data['text'])) if data['level'][i] == 5 and data['text'][i].strip()]

        # Focus on a specific line if the pointer is set correctly
        if 0 <= self.pointer < len(lines):
            return lines[self.pointer]
        else:
            raise IndexError("Pointer position is out of bounds.")

# Example instantiation
num_classes = 1000  # Example for image classification

class MiniVisionTransformerWithIO(MiniVisionTransformer):
    def __init__(self, *args, **kwargs):
        super(MiniVisionTransformerWithIO, self).__init__(*args, **kwargs)
    def get_screen(self):
        try:
            screenshot = PIL.ImageGrab.grab(all_screens=True)
            return screenshot
        except Exception as e:
            print("error grabbing screen")
    def process_screen(self):
        """
        Captures the current screen and processes it for input into the model.
        """
        try:
            # Capture the screen
            screen = np.array(PIL.ImageGrab.grab(all_screens=True))

            # Convert to BGR format for OpenCV
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # Normalize and convert to tensor
            screen_tensor = torch.from_numpy(screen_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            return self.forward(screen_tensor)
        except Exception as e:
            print(f"Error processing screen: {e}")
            return None

    def process_camera(self, camera_index=0):
        """
        Captures a frame from the specified camera and processes it for input into the model.
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError(f"Camera at index {camera_index} could not be opened.")
            
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Failed to capture frame from camera.")

            # Convert to tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            return self.forward(frame_tensor)
        except Exception as e:
            print(f"Error processing camera: {e}")
            return None

    def process_cameras(self, camera_indices=[0, 1]):
        """
        Processes multiple cameras and combines their results.
        """
        try:
            results = []
            for camera_index in camera_indices:
                result = self.process_camera(camera_index)
                if result is not None:
                    results.append(result)
            
            if len(results) == 0:
                raise ValueError("No valid results from cameras.")
            
            # Combine results (e.g., average outputs)
            combined_output = torch.mean(torch.stack([res[0] for res in results]), dim=0)
            combined_energy = sum(res[1] for res in results)

            return combined_output, combined_energy
        except Exception as e:
            print(f"Error processing multiple cameras: {e}")
            return None

    def process_input(self, input_type='screen', camera_indices=[0]):
        """
        Automatically processes input based on the specified input type.
        Args:
            input_type (str): 'screen', 'camera', or 'cameras'.
            camera_indices (list): List of camera indices for processing multiple cameras.
        Returns:
            Processed output based on the input type.
        """
        try:
            if input_type == 'screen':
                return self.process_screen()
            elif input_type == 'camera':
                return self.process_camera(camera_indices[0])  # Default to the first camera index
            elif input_type == 'cameras':
                return self.process_cameras(camera_indices)
            else:
                raise ValueError("Unsupported input type. Choose 'screen', 'camera', or 'cameras'.")
        except Exception as e:
            print(f"Error in process_input: {e}")
            return None

# Example Usage
vision_system = MiniVisionTransformerWithIO(3, 64, 256, exec_func=None, num_transformer_layers=6, nhead=8, num_classes=1000)

# Process screen
screen_output = vision_system.process_input(input_type='screen')

# Process single camera
camera_output = vision_system.process_input(input_type='camera', camera_indices=[0])

# Process multiple cameras
multi_camera_output = vision_system.process_input(input_type='cameras', camera_indices=[0, 1])

print(f"Screen Output: {screen_output}")
print(f"Camera Output: {camera_output}")
print(f"Multi-Camera Output: {multi_camera_output}")

