import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_vision import MiniVisionTransformer, MiniVisionTransformerWithIO

class TestVisionProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create sample vision parameters
        self.input_channels = 3
        self.hidden_dim = 64
        self.output_dim = 256
        self.num_classes = 1000
        
        # Create mock for executive function
        self.mock_exec_func = MagicMock()
        self.mock_exec_func.return_value = torch.randn(2, self.input_channels, 224, 224)
        
    def test_vision_transformer_initialization(self):
        """Test that the vision transformer initializes correctly."""
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Check that the model has the expected attributes
        self.assertEqual(vision_model.exec_func, self.mock_exec_func)
        self.assertEqual(vision_model.pointer, 0)
        
        # Check that the expected layers were created
        self.assertTrue(hasattr(vision_model, 'conv_layers'))
        self.assertTrue(hasattr(vision_model, 'flatten'))
        self.assertTrue(hasattr(vision_model, 'linear_proj'))
        self.assertTrue(hasattr(vision_model, 'transformer_encoder'))
        self.assertTrue(hasattr(vision_model, 'adaption_layer'))
        self.assertTrue(hasattr(vision_model, 'classifier'))
        self.assertTrue(hasattr(vision_model, 'lstm'))
        
    def test_auto_select_mode(self):
        """Test the auto_select_mode method."""
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Test with 2D RGB image
        x = torch.randn(1, 3, 224, 224)
        mode = vision_model.auto_select_mode(x)
        self.assertEqual(mode, 'image')
        
        # Test with batch of RGB images
        x = torch.randn(5, 3, 224, 224)
        mode = vision_model.auto_select_mode(x)
        self.assertEqual(mode, 'video')
        
        # Test with grayscale image
        x = torch.randn(1, 1, 224, 224)
        mode = vision_model.auto_select_mode(x)
        self.assertEqual(mode, 'ocr')
        
        # Test with video
        x = torch.randn(1, 10, 3, 224, 224)
        mode = vision_model.auto_select_mode(x)
        self.assertEqual(mode, 'video')
        
        # Test with invalid input
        with self.assertRaises(ValueError):
            x = torch.randn(3, 224)
            vision_model.auto_select_mode(x)
            
    @patch('torch.nn.Conv2d')
    @patch('torch.nn.BatchNorm2d')
    @patch('torch.nn.ReLU')
    @patch('torch.nn.AdaptiveAvgPool2d')
    @patch('torch.nn.Flatten')
    @patch('torch.nn.Linear')
    @patch('torch.nn.LSTM')
    def test_forward_image_mode(self, mock_lstm, mock_linear, mock_flatten, 
                               mock_pool, mock_relu, mock_bn, mock_conv):
        """Test the forward method in image mode."""
        # Create vision model
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Create dummy input
        x = torch.randn(1, self.input_channels, 224, 224)
        
        # Mock the forward methods of components
        mock_conv_instance = mock_conv.return_value
        mock_conv_instance.return_value = torch.randn(1, 64, 224, 224)
        
        mock_bn_instance = mock_bn.return_value
        mock_bn_instance.return_value = torch.randn(1, 64, 224, 224)
        
        mock_relu_instance = mock_relu.return_value
        mock_relu_instance.return_value = torch.randn(1, 64, 224, 224)
        
        mock_pool_instance = mock_pool.return_value
        mock_pool_instance.return_value = torch.randn(1, 64, 1, 1)
        
        mock_flatten_instance = mock_flatten.return_value
        mock_flatten_instance.return_value = torch.randn(1, 64)
        
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(1, self.output_dim)
        
        # Set up the mocks for the model's components
        vision_model.conv_layers = MagicMock()
        vision_model.conv_layers.return_value = torch.randn(1, 64, 1, 1)
        
        vision_model.flatten = mock_flatten_instance
        vision_model.linear_proj = mock_linear_instance
        
        # Create a mock for the transformer encoder
        vision_model.transformer_encoder = MagicMock()
        vision_model.transformer_encoder.return_value = torch.randn(1, 1, self.hidden_dim)
        
        # Create a mock for the adaptation layer
        vision_model.adaption_layer = mock_linear_instance
        
        # Create a mock for the classifier
        vision_model.classifier = mock_linear_instance
        
        # Call forward
        with patch.object(vision_model, 'auto_select_mode', return_value='image'):
            classification, energy = vision_model.forward(x)
        
        # Check output shapes
        self.assertEqual(classification.shape, (1, self.output_dim))
        self.assertTrue(isinstance(energy, torch.Tensor))
        
        # Check that the expected components were called
        vision_model.conv_layers.assert_called_once()
        mock_flatten_instance.assert_called_once()
        vision_model.transformer_encoder.assert_called_once()
        
    @patch('torch.nn.LSTM')
    def test_forward_video_mode(self, mock_lstm):
        """Test the forward method in video mode."""
        # Create vision model
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Create dummy input
        x = torch.randn(5, self.input_channels, 224, 224)
        
        # Set up the mocks for the model's components
        vision_model.conv_layers = MagicMock()
        vision_model.conv_layers.return_value = torch.randn(5, 64, 1, 1)
        
        vision_model.flatten = MagicMock()
        vision_model.flatten.return_value = torch.randn(5, 64)
        
        vision_model.linear_proj = MagicMock()
        vision_model.linear_proj.return_value = torch.randn(5, self.hidden_dim)
        
        # Create a mock for the LSTM
        mock_lstm_instance = mock_lstm.return_value
        mock_lstm_instance.return_value = (
            torch.randn(5, 1, self.hidden_dim),
            (torch.randn(1, 5, self.hidden_dim), torch.randn(1, 5, self.hidden_dim))
        )
        
        vision_model.lstm = mock_lstm_instance
        
        # Create a mock for the adaptation layer
        vision_model.adaption_layer = MagicMock()
        vision_model.adaption_layer.return_value = torch.randn(5, self.output_dim)
        
        # Create a mock for the classifier
        vision_model.classifier = MagicMock()
        vision_model.classifier.return_value = torch.randn(5, self.num_classes)
        
        # Call forward
        with patch.object(vision_model, 'auto_select_mode', return_value='video'):
            classification, energy = vision_model.forward(x)
        
        # Check output shapes
        self.assertEqual(classification.shape, (5, self.num_classes))
        self.assertTrue(isinstance(energy, torch.Tensor))
        
        # Check that the expected components were called
        vision_model.conv_layers.assert_called_once()
        vision_model.flatten.assert_called_once()
        vision_model.linear_proj.assert_called_once()
        mock_lstm_instance.assert_called_once()
        vision_model.adaption_layer.assert_called_once()
        vision_model.classifier.assert_called_once()
        
    @patch('pytesseract.image_to_string')
    def test_forward_ocr_mode(self, mock_image_to_string):
        """Test the forward method in OCR mode."""
        # Create vision model
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Create dummy input
        x = torch.randn(1, 1, 224, 224)
        
        # Set up the mocks for the model's components
        vision_model.conv_layers = MagicMock()
        vision_model.conv_layers.return_value = torch.randn(1, 64, 1, 1)
        
        vision_model.flatten = MagicMock()
        vision_model.flatten.return_value = torch.randn(1, 64)
        
        vision_model.linear_proj = MagicMock()
        vision_model.linear_proj.return_value = torch.randn(1, self.hidden_dim)
        
        vision_model.ocr_layer = MagicMock()
        vision_model.ocr_layer.return_value = torch.randn(1, self.output_dim)
        
        # Mock image_to_string
        mock_image_to_string.return_value = "Test OCR text"
        
        # Call forward
        with patch.object(vision_model, 'auto_select_mode', return_value='ocr'):
            with patch.object(vision_model, 'extract_text_from_image', return_value="Test OCR text"):
                result = vision_model.forward(x)
        
        # Check output
        self.assertEqual(result, "Test OCR text")
        
        # Check that the expected components were called
        vision_model.conv_layers.assert_called_once()
        vision_model.flatten.assert_called_once()
        vision_model.linear_proj.assert_called_once()
        vision_model.ocr_layer.assert_called_once()
        
    def test_extract_text_from_image(self):
        """Test the extract_text_from_image method."""
        # Create vision model
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Create dummy image tensor
        image_tensor = torch.zeros(1, 224, 224)
        
        # Mock pytesseract.image_to_string
        with patch('pytesseract.image_to_string', return_value="Test OCR text"):
            text = vision_model.extract_text_from_image(image_tensor)
        
        # Check output
        self.assertEqual(text, "Test OCR text")
        
    def test_set_pointer(self):
        """Test the set_pointer method."""
        # Create vision model
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Set pointer
        vision_model.set_pointer(5)
        
        # Check that pointer was set
        self.assertEqual(vision_model.pointer, 5)
        
    @patch('cv2.adaptiveThreshold')
    @patch('cv2.cvtColor')
    @patch('pytesseract.image_to_data')
    def test_focus_line_by_line(self, mock_image_to_data, mock_cvtColor, mock_adaptiveThreshold):
        """Test the focus_line_by_line method."""
        # Create vision model
        vision_model = MiniVisionTransformer(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Create dummy image tensor
        image_tensor = torch.zeros(1, 224, 224)
        
        # Mock cv2 functions
        mock_cvtColor.return_value = np.zeros((224, 224), dtype=np.uint8)
        mock_adaptiveThreshold.return_value = np.zeros((224, 224), dtype=np.uint8)
        
        # Mock pytesseract.image_to_data
        mock_image_to_data.return_value = {
            'text': ['Line 1', 'Line 2', 'Line 3'],
            'level': [5, 5, 5]
        }
        
        # Set pointer
        vision_model.set_pointer(1)
        
        # Call focus_line_by_line
        text = vision_model.focus_line_by_line(image_tensor)
        
        # Check output
        self.assertEqual(text, "Line 2")
        
        # Check that the expected functions were called
        mock_cvtColor.assert_called_once()
        mock_adaptiveThreshold.assert_called_once()
        mock_image_to_data.assert_called_once()
        
    def test_vision_transformer_with_io_initialization(self):
        """Test that the vision transformer with IO initializes correctly."""
        vision_model = MiniVisionTransformerWithIO(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Check that the model inherits from MiniVisionTransformer
        self.assertTrue(isinstance(vision_model, MiniVisionTransformer))
        
        # Check that the model has the expected additional methods
        self.assertTrue(hasattr(vision_model, 'get_screen'))
        self.assertTrue(hasattr(vision_model, 'process_screen'))
        self.assertTrue(hasattr(vision_model, 'process_camera'))
        self.assertTrue(hasattr(vision_model, 'process_cameras'))
        self.assertTrue(hasattr(vision_model, 'process_input'))
        
    @patch('PIL.ImageGrab.grab')
    def test_get_screen(self, mock_grab):
        """Test the get_screen method."""
        # Create vision model
        vision_model = MiniVisionTransformerWithIO(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Mock PIL.ImageGrab.grab
        mock_grab.return_value = MagicMock()
        
        # Call get_screen
        screenshot = vision_model.get_screen()
        
        # Check that the expected function was called
        mock_grab.assert_called_once_with(all_screens=True)
        
        # Check output
        self.assertIsNotNone(screenshot)
        
    @patch('PIL.ImageGrab.grab')
    @patch('cv2.cvtColor')
    @patch('torch.from_numpy')
    def test_process_screen(self, mock_from_numpy, mock_cvtColor, mock_grab):
        """Test the process_screen method."""
        # Create vision model
        vision_model = MiniVisionTransformerWithIO(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Mock PIL.ImageGrab.grab
        mock_grab.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Mock cv2.cvtColor
        mock_cvtColor.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Mock torch.from_numpy
        mock_tensor = MagicMock()
        mock_tensor.permute.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.__truediv__ = MagicMock(return_value=mock_tensor)
        mock_from_numpy.return_value = mock_tensor
        
        # Mock forward method
        vision_model.forward = MagicMock()
        vision_model.forward.return_value = (torch.randn(1, self.num_classes), torch.tensor(0.5))
        
        # Call process_screen
        result = vision_model.process_screen()
        
        # Check that the expected functions were called
        mock_grab.assert_called_once_with(all_screens=True)
        mock_cvtColor.assert_called_once()
        mock_from_numpy.assert_called_once()
        mock_tensor.permute.assert_called_once_with(2, 0, 1)
        mock_tensor.unsqueeze.assert_called_once_with(0)
        vision_model.forward.assert_called_once()
        
        # Check output
        self.assertEqual(result, vision_model.forward.return_value)
        
    @patch('cv2.VideoCapture')
    def test_process_camera(self, mock_VideoCapture):
        """Test the process_camera method."""
        # Create vision model
        vision_model = MiniVisionTransformerWithIO(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Mock cv2.VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_VideoCapture.return_value = mock_cap
        
        # Mock torch.from_numpy
        with patch('torch.from_numpy') as mock_from_numpy:
            mock_tensor = MagicMock()
            mock_tensor.permute.return_value = mock_tensor
            mock_tensor.unsqueeze.return_value = mock_tensor
            mock_tensor.float.return_value = mock_tensor
            mock_tensor.__truediv__ = MagicMock(return_value=mock_tensor)
            mock_from_numpy.return_value = mock_tensor
            
            # Mock forward method
            vision_model.forward = MagicMock()
            vision_model.forward.return_value = (torch.randn(1, self.num_classes), torch.tensor(0.5))
            
            # Call process_camera
            result = vision_model.process_camera(0)
        
        # Check that the expected functions were called
        mock_VideoCapture.assert_called_once_with(0)
        mock_cap.isOpened.assert_called_once()
        mock_cap.read.assert_called_once()
        mock_cap.release.assert_called_once()
        mock_from_numpy.assert_called_once()
        mock_tensor.permute.assert_called_once_with(2, 0, 1)
        mock_tensor.unsqueeze.assert_called_once_with(0)
        vision_model.forward.assert_called_once()
        
        # Check output
        self.assertEqual(result, vision_model.forward.return_value)
        
    def test_process_cameras(self):
        """Test the process_cameras method."""
        # Create vision model
        vision_model = MiniVisionTransformerWithIO(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Mock process_camera
        vision_model.process_camera = MagicMock()
        vision_model.process_camera.side_effect = [
            (torch.randn(1, self.num_classes), torch.tensor(0.5)),
            (torch.randn(1, self.num_classes), torch.tensor(0.6))
        ]
        
        # Call process_cameras
        with patch('torch.mean', return_value=torch.randn(1, self.num_classes)) as mock_mean:
            result = vision_model.process_cameras([0, 1])
        
        # Check that the expected functions were called
        self.assertEqual(vision_model.process_camera.call_count, 2)
        vision_model.process_camera.assert_any_call(0)
        vision_model.process_camera.assert_any_call(1)
        mock_mean.assert_called_once()
        
        # Check output
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result[0], torch.Tensor))
        self.assertTrue(isinstance(result[1], (int, float, torch.Tensor)))
        
    def test_process_input(self):
        """Test the process_input method."""
        # Create vision model
        vision_model = MiniVisionTransformerWithIO(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            exec_func=self.mock_exec_func,
            num_classes=self.num_classes
        )
        
        # Mock the processing methods
        vision_model.process_screen = MagicMock()
        vision_model.process_screen.return_value = "screen_result"
        
        vision_model.process_camera = MagicMock()
        vision_model.process_camera.return_value = "camera_result"
        
        vision_model.process_cameras = MagicMock()
        vision_model.process_cameras.return_value = "cameras_result"
        
        # Test screen input
        result = vision_model.process_input('screen')
        self.assertEqual(result, "screen_result")
        vision_model.process_screen.assert_called_once()
        
        # Test camera input
        result = vision_model.process_input('camera', [2])
        self.assertEqual(result, "camera_result")
        vision_model.process_camera.assert_called_once_with(2)
        
        # Test cameras input
        result = vision_model.process_input('cameras', [0, 1])
        self.assertEqual(result, "cameras_result")
        vision_model.process_cameras.assert_called_once_with([0, 1])
        
        # Test invalid input
        with self.assertRaises(ValueError):
            vision_model.process_input('invalid')

if __name__ == '__main__':
    unittest.main()