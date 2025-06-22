import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_audio import MiniAudioModule

class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create mock dependencies
        self.mock_energy_based_synapse = MagicMock()
        self.mock_dynamic_synapse = MagicMock()
        self.mock_efficient_attention = MagicMock()
        
        # Patch the dependencies
        self.patches = [
            patch('modules.oni_audio.EnergyBasedSynapse', self.mock_energy_based_synapse),
            patch('modules.oni_audio.DynamicSynapse', self.mock_dynamic_synapse),
            patch('modules.oni_audio.EfficientAttention', self.mock_efficient_attention)
        ]
        
        for p in self.patches:
            p.start()
            
        # Create sample audio parameters
        self.input_channels = 3
        self.hidden_dim = 896
        self.num_freq_bins = 256
        self.num_time_steps = 128
        self.vocab_size = 1000
        
    def tearDown(self):
        """Clean up after each test method."""
        # Stop all patches
        for p in self.patches:
            p.stop()
    
    def test_audio_module_initialization(self):
        """Test that the audio module initializes correctly."""
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Check that the module has the expected attributes
        self.assertEqual(audio_module.input_channels, self.input_channels)
        self.assertEqual(audio_module.hidden_dim, self.hidden_dim)
        self.assertEqual(audio_module.num_freq_bins, self.num_freq_bins)
        self.assertEqual(audio_module.num_time_steps, self.num_time_steps)
        self.assertEqual(audio_module.vocab_size, self.vocab_size)
        
        # Check that the expected layers were created
        self.assertTrue(hasattr(audio_module, 'text_embedding'))
        self.assertTrue(hasattr(audio_module, 'dynamic_layer'))
        self.assertTrue(hasattr(audio_module, 'adaption_layer'))
        self.assertTrue(hasattr(audio_module, 'conv_layers'))
        self.assertTrue(hasattr(audio_module, 'attention'))
        self.assertTrue(hasattr(audio_module, 'bi_lstm'))
        self.assertTrue(hasattr(audio_module, 'text_to_audio_fc'))
        self.assertTrue(hasattr(audio_module, 'fc_output'))
        self.assertTrue(hasattr(audio_module, 'phase_predictor'))
        self.assertTrue(hasattr(audio_module, 'decoder'))
        self.assertTrue(hasattr(audio_module, 'music_lstm'))
        self.assertTrue(hasattr(audio_module, 'music_fc'))
        
    @patch('torch.nn.LSTM')
    @patch('torch.nn.Linear')
    def test_audio_forward_mode(self, mock_linear, mock_lstm):
        """Test the audio forward mode of the audio module."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Create dummy input
        x = torch.randn(2, self.input_channels, self.num_freq_bins, self.num_time_steps)
        
        # Mock the forward methods of components
        audio_module.dynamic_layer = MagicMock()
        audio_module.dynamic_layer.return_value = (
            torch.randn(2, self.input_channels, self.num_freq_bins * self.num_time_steps),
            torch.tensor([0.5, 0.6])
        )
        
        audio_module.adaption_layer = MagicMock()
        audio_module.adaption_layer.return_value = torch.randn(2, 1, self.num_freq_bins, self.num_time_steps)
        
        audio_module.conv_layers = MagicMock()
        audio_module.conv_layers.return_value = torch.randn(2, 128, self.num_freq_bins // 4, self.num_time_steps // 4)
        
        audio_module.attention = MagicMock()
        audio_module.attention.return_value = torch.randn(2, 128, self.num_freq_bins // 4, self.num_time_steps // 4)
        
        # Mock LSTM
        mock_lstm_instance = mock_lstm.return_value
        mock_lstm_instance.return_value = (
            torch.randn(2, self.num_time_steps // 4, self.hidden_dim * 2),
            (torch.randn(1, 2, self.hidden_dim), torch.randn(1, 2, self.hidden_dim))
        )
        
        # Mock Linear layers
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, self.num_freq_bins)
        
        audio_module.fc_output = mock_linear_instance
        audio_module.phase_predictor = mock_linear_instance
        
        audio_module.decoder = MagicMock()
        audio_module.decoder.return_value = torch.randn(2, self.num_freq_bins * self.num_time_steps)
        
        # Call forward in audio mode
        output, energy = audio_module.forward(x, mode='audio')
        
        # Check output shape
        self.assertEqual(output.shape, (2, self.num_freq_bins * self.num_time_steps))
        self.assertTrue(isinstance(energy, torch.Tensor))
        
        # Check that the expected components were called
        audio_module.dynamic_layer.assert_called_once()
        audio_module.adaption_layer.assert_called_once()
        audio_module.conv_layers.assert_called_once()
        audio_module.attention.assert_called_once()
        audio_module.decoder.assert_called_once()
        
    @patch('torch.nn.Embedding')
    @patch('torch.nn.Linear')
    def test_text_to_audio_forward_mode(self, mock_linear, mock_embedding):
        """Test the text-to-audio forward mode of the audio module."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Create dummy input
        text_input = torch.randint(0, self.vocab_size, (2, 10))
        
        # Mock the forward methods of components
        mock_embedding_instance = mock_embedding.return_value
        mock_embedding_instance.return_value = torch.randn(2, 10, self.hidden_dim)
        
        audio_module.text_embedding = mock_embedding_instance
        
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, self.num_freq_bins * self.num_time_steps)
        
        audio_module.text_to_audio_fc = mock_linear_instance
        
        audio_module.decoder = MagicMock()
        audio_module.decoder.return_value = torch.randn(2, self.num_freq_bins * self.num_time_steps)
        
        # Call forward in text-to-audio mode
        output = audio_module.forward(text_input=text_input, mode='text')
        
        # Check output shape
        self.assertEqual(output.shape, (2, self.num_freq_bins * self.num_time_steps))
        
        # Check that the expected components were called
        mock_embedding_instance.assert_called_once()
        mock_linear_instance.assert_called_once()
        audio_module.decoder.assert_called_once()
        
    @patch('torch.nn.LSTM')
    @patch('torch.nn.Linear')
    def test_music_forward_mode(self, mock_linear, mock_lstm):
        """Test the music forward mode of the audio module."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Create dummy input
        x = torch.randn(2, 10, self.hidden_dim)
        
        # Mock the forward methods of components
        mock_lstm_instance = mock_lstm.return_value
        mock_lstm_instance.return_value = (
            torch.randn(2, 10, self.hidden_dim),
            (torch.randn(2, 2, self.hidden_dim), torch.randn(2, 2, self.hidden_dim))
        )
        
        audio_module.music_lstm = mock_lstm_instance
        
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, self.num_freq_bins * self.num_time_steps)
        
        audio_module.music_fc = mock_linear_instance
        
        audio_module.decoder = MagicMock()
        audio_module.decoder.return_value = torch.randn(2, self.num_freq_bins * self.num_time_steps)
        
        # Call forward in music mode
        output = audio_module.forward(x, mode='music')
        
        # Check output shape
        self.assertEqual(output.shape, (2, self.num_freq_bins * self.num_time_steps))
        
        # Check that the expected components were called
        mock_lstm_instance.assert_called_once()
        mock_linear_instance.assert_called_once()
        audio_module.decoder.assert_called_once()
        
    @patch('torch.nn.functional.mel_spectrogram')
    def test_preprocess_audio(self, mock_mel_spectrogram):
        """Test audio preprocessing."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Create dummy waveform
        waveform = torch.randn(1, 16000)
        
        # Mock mel_spectrogram
        mock_mel_spectrogram.return_value = torch.randn(1, self.num_freq_bins, 100)
        
        # Set up the mock for the module's mel_spectrogram
        audio_module.mel_spectrogram = MagicMock()
        audio_module.mel_spectrogram.return_value = torch.randn(1, self.num_freq_bins, 100)
        
        # Call preprocess_audio
        output = audio_module.preprocess_audio(waveform)
        
        # Check that the expected function was called
        audio_module.mel_spectrogram.assert_called_once_with(waveform)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.num_freq_bins, 100))
        
    @patch('pyttsx3.init')
    def test_tts_generate(self, mock_init):
        """Test text-to-speech generation."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Mock TTS engine
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        # Set up the mock for the module's tts_engine
        audio_module.tts_engine = mock_engine
        
        # Call tts_generate
        text = "Hello, this is a test."
        audio_path = audio_module.tts_generate(text, play_audio=False)
        
        # Check that the expected methods were called
        mock_engine.save_to_file.assert_called_once()
        mock_engine.runAndWait.assert_called_once()
        
        # Check that the function returns a path
        self.assertTrue(isinstance(audio_path, str))
        self.assertTrue("generated_speech_" in audio_path)
        self.assertTrue(audio_path.endswith(".wav"))
        
    @patch('wave.open')
    @patch('pyaudio.PyAudio')
    def test_play_audio(self, mock_pyaudio, mock_wave_open):
        """Test audio playback."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Mock wave file
        mock_wave_file = MagicMock()
        mock_wave_file.getsampwidth.return_value = 2
        mock_wave_file.getnchannels.return_value = 1
        mock_wave_file.getframerate.return_value = 16000
        mock_wave_file.readframes.side_effect = [b'data', b'']
        mock_wave_open.return_value = mock_wave_file
        
        # Mock PyAudio
        mock_pa = MagicMock()
        mock_stream = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pa
        
        # Call play_audio
        audio_path = "test.wav"
        audio_module.play_audio(audio_path)
        
        # Check that the expected methods were called
        mock_wave_open.assert_called_once_with(audio_path, 'rb')
        mock_pa.open.assert_called_once()
        mock_stream.write.assert_called_once_with(b'data')
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pa.terminate.assert_called_once()
        mock_wave_file.close.assert_called_once()
        
    @patch('transformers.Wav2Vec2Tokenizer')
    @patch('transformers.Wav2Vec2ForCTC')
    def test_audio_to_text(self, mock_model, mock_tokenizer):
        """Test audio-to-text conversion."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {"input_values": torch.randn(1, 1000)}
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_logits = MagicMock()
        mock_logits.logits = torch.randn(1, 1000, 32)
        mock_model_instance.return_value = mock_logits
        mock_model.return_value = mock_model_instance
        
        # Set up the mocks for the module's components
        audio_module.audio_tokenizer = mock_tokenizer_instance
        audio_module.audio_to_text_model = mock_model_instance
        
        # Create dummy audio tensor
        audio_tensor = torch.randn(16000)
        
        # Call audio_to_text
        with patch('torch.argmax', return_value=torch.tensor([1, 2, 3])):
            with patch.object(mock_tokenizer_instance, 'decode', return_value="test transcription"):
                text = audio_module.audio_to_text(audio_tensor)
        
        # Check that the expected methods were called
        mock_tokenizer_instance.assert_called_once()
        mock_model_instance.assert_called_once()
        
        # Check output
        self.assertEqual(text, "test transcription")
        
    @patch('transformers.pipeline')
    def test_text_to_music(self, mock_pipeline):
        """Test text-to-music generation."""
        # Create audio module
        audio_module = MiniAudioModule(
            self.input_channels,
            self.hidden_dim,
            self.num_freq_bins,
            self.num_time_steps,
            self.vocab_size
        )
        
        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = {
            "audio": np.zeros(1000),
            "sampling_rate": 16000
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Set up the mock for the module's text_to_music
        audio_module.text_to_music = mock_pipeline_instance
        
        # Mock scipy.io.wavfile.write
        with patch('scipy.io.wavfile.write') as mock_write:
            # Mock pygame
            with patch('pygame.mixer') as mock_mixer:
                # Call TTM
                audio_module.TTM("Generate a happy tune")
        
        # Check that the expected methods were called
        mock_pipeline_instance.assert_called_once()
        mock_write.assert_called_once()
        mock_mixer.init.assert_called_once()
        mock_mixer.music.load.assert_called_once()
        mock_mixer.music.play.assert_called_once()

if __name__ == '__main__':
    unittest.main()