 # Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import torchaudio
from einops import rearrange
import pytesseract
import cv2  # For image preprocessing
import pyttsx3  # Import for Text-to-Speech processing
import pyaudio  # Import for audio playback
import wave  # For reading wav files
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer  # Import for Audio-to-Text processing
import pygame
import scipy
from elevenlabs import ElevenLabs, play, Voice, VoiceSettings
# from custom_layers import EnergyBasedSynapse, DynamicSynapse, EfficientAttention
import uuid
import os

text_to_music = pipeline("text-to-audio", model="facebook/musicgen-small", device=torch.device('cuda'))


class MiniAudioModule(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_freq_bins=869, num_time_steps=128, vocab_size=1000):
        super(MiniAudioModule, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_freq_bins = num_freq_bins
        self.num_time_steps = num_time_steps
        self.vocab_size = vocab_size

        # Embedding for text input
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Dynamic and adaptation layers
        self.dynamic_layer = EnergyBasedSynapse(input_channels, hidden_dim)
        self.adaption_layer = DynamicSynapse(hidden_dim, 896)

        # Convolutional layers for feature extraction
        self.conv_layers = self._build_conv_layers()

        # Attention mechanism
        self.attention = EfficientAttention(128, 128)

        # Bidirectional LSTM for temporal modeling
        self.bi_lstm = nn.LSTM(128 * (num_freq_bins // 4), hidden_dim, bidirectional=True, batch_first=True)

        # Text-to-Audio processing
        self.text_to_audio_fc = nn.Sequential(
            nn.Linear(hidden_dim, num_freq_bins * num_time_steps),
            nn.Tanh()
        )

        # Audio output layers
        self.fc_output = nn.Linear(hidden_dim * 2, num_freq_bins)
        self.phase_predictor = nn.Linear(hidden_dim * 2, num_freq_bins)

        # Audio decoder for generating waveforms
        self.decoder = self._build_audio_decoder(hidden_dim, num_freq_bins, num_time_steps)

        # Audio processing tools
        self.mel_spectrogram, self.inverse_mel_scale = self._init_audio_processing_tools(num_freq_bins)

        # Music production layers
        self.music_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.music_fc = nn.Linear(hidden_dim, num_freq_bins * num_time_steps)

        # Text-to-Speech model
        self.tts_engine = pyttsx3.init()

        # Audio-to-Text model
        self.audio_tokenizer, self.audio_to_text_model = self._init_audio_to_text_model()

        # Text-to-Music generation function
        self.text_to_music = text_to_music  # Placeholder for text-to-music synthesis
        self.client = ElevenLabs(
            api_key="sk_d91e2e99f68999dc1952db4ff5c4179c4e83a585cef7e8b9",
        )

    def _build_conv_layers(self):
        """Build convolutional layers for feature extraction."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.num_freq_bins // 4, self.num_time_steps // 4))
        )

    def _build_audio_decoder(self, hidden_dim, num_freq_bins, num_time_steps):
        """Build the audio decoder network."""
        return nn.Sequential(
            nn.Linear(num_freq_bins * num_time_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_freq_bins * num_time_steps),
            nn.Tanh()
        )

    def TTM(self, text):
        # Convert text to music
        music = self.text_to_music(text, forward_params={"do_sample": True})
        
        # Write the audio data to a file
        output_path = "musicgen_out.wav"
        scipy.io.wavfile.write(output_path, rate=music["sampling_rate"], data=music["audio"])
        
        # Initialize pygame mixer and load the file
        pygame.mixer.init()
        pygame.mixer.music.load(output_path)
        
        # Play the music
        pygame.mixer.music.play()

    # Don't forget to handle quitting or stopping the mixer


    def _init_audio_processing_tools(self, num_freq_bins):
        """Initialize audio processing tools."""
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=num_freq_bins
        )
        inverse_mel_scale = T.InverseMelScale(n_stft=num_freq_bins, n_mels=num_freq_bins)
        return mel_spectrogram, inverse_mel_scale

    def _init_audio_to_text_model(self):
        """Initialize the Wav2Vec2 model for audio-to-text processing."""
        audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        audio_to_text_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        return audio_tokenizer, audio_to_text_model

    def forward(self, x=None, text_input=None, mode='audio'):
        if mode == 'text':
            return self._text_to_audio_forward(text_input)
        elif mode == 'audio':
            return self._audio_forward(x)
        elif mode == 'music':
            return self._music_forward(x)
        else:
            raise ValueError("Invalid mode. Choose 'audio', 'text', or 'music'.")

    def _text_to_audio_forward(self, text_input):
        text_embedded = self.text_embedding(text_input)
        text_embedded = text_embedded.mean(dim=1)  # Average over the sequence length
        audio_features = self.text_to_audio_fc(text_embedded)
        audio_features = audio_features.view(-1, 1, self.num_freq_bins, self.num_time_steps)
        audio_waveform = self.decoder(audio_features.view(-1, self.num_freq_bins * self.num_time_steps))
        return audio_waveform

    def _audio_forward(self, x):
        # Assume x is of shape (batch_size, input_channels, num_freq_bins, num_time_steps)
        batch_size = x.size(0)

        # Dynamic processing
        x, energy = self.dynamic_layer(x.view(batch_size, self.input_channels, -1))
        x = x.view(batch_size, 1, self.num_freq_bins, self.num_time_steps)

        # Adaptation
        x = self.adaption_layer(x)

        # Feature extraction
        x = self.conv_layers(x)

        # Attention mechanism
        x = self.attention(x)

        # Temporal modeling
        x = x.reshape(batch_size, self.num_time_steps // 4, -1)
        x, _ = self.bi_lstm(x)

        # Output generation
        magnitude = self.fc_output(x)
        phase = self.phase_predictor(x)

        # Combine magnitude and phase to reconstruct the complex spectrogram
        audio_features = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

        # Audio output
        audio_waveform = self.decoder(audio_features.view(-1, self.num_freq_bins * self.num_time_steps))
        return audio_waveform, energy

    def _music_forward(self, x):
        batch_size = x.size(0)
        x, _ = self.music_lstm(x)
        audio_features = self.music_fc(x[:, -1, :])  # Use only the last output
        audio_features = audio_features.view(batch_size, 1, self.num_freq_bins, self.num_time_steps)
        audio_waveform = self.decoder(audio_features.view(-1, self.num_freq_bins * self.num_time_steps))
        return audio_waveform

    def preprocess_audio(self, waveform, sample_rate=16000):
        """
        Preprocesses audio waveform by converting it to a Mel spectrogram.
        """
        return self.mel_spectrogram(waveform)

    def tts_generate(self, text, play_audio=False):
        """ 
        Generate speech from text using the pyttsx3 TTS engine.
        Uses a unique filename for each generation to prevent file conflicts.
        """
        # Generate a unique filename using UUID to ensure no file overwrites
        audio_path = f'generated_speech_{uuid.uuid4()}.wav'
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(audio_path) or '.', exist_ok=True)
            
            # Generate speech and save to file
            self.tts_engine.save_to_file(text, audio_path)
            self.tts_engine.runAndWait()
            
            if play_audio:
                # Play the generated audio using PyAudio
                self.play_audio(audio_path)
            
            return audio_path
        
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None

    def better_tts(self, text):
        voice = self.client.generate(
            text=text,
            voice=Voice(
                voice_id='nPczCjzI2devNBz1zQrb',
                settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
            )
            )

        play(voice)
        
    def play_audio(self, audio_path):
        """ 
        Play the generated audio file using PyAudio.
        Includes error handling and resource management.
        """
        try:
            chunk = 1024
            wf = wave.open(audio_path, 'rb')
            pa = pyaudio.PyAudio()
            
            # Open stream
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Read and play audio in chunks
            data = wf.readframes(chunk)
            while data:
                stream.write(data)
                data = wf.readframes(chunk)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            pa.terminate()
            wf.close()
            
            # Optionally, remove the audio file after playing
            # Uncomment the next line if you want to delete the file after playing
            # os.remove(audio_path)
        
        except Exception as e:
            print(f"Error playing audio: {e}")

    def audio_to_text(self, audio_tensor):
        """
        Convert audio input to text using a pretrained Wav2Vec2 model.
        """
        # Resample audio if needed
        input_values = self.audio_tokenizer(audio_tensor, return_tensors="pt", sampling_rate=16000).input_values
        logits = self.audio_to_text_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.audio_tokenizer.decode(predicted_ids[0])

# Example instantiation
num_classes = 1000  # Example for image classification
audio = MiniAudioModule(3, 896, 256)
