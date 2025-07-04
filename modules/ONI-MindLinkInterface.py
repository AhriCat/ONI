#!/usr/bin/env python3
"""
ONI-MindLink Interface
Integration between Mind-link EEG BCI and ONI AGI System

This interface enables direct brain-to-AGI communication by processing EEG signals
from the Mind-link device and feeding them into ONI's multimodal sensory cortex.

Author: ONI-MindLink Integration Team
License: Pantheum License (matching ONI)
"""

import numpy as np
import serial
import time
import threading
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainWaveType(Enum):
    """EEG brainwave frequency bands"""
    DELTA = (0.5, 4.0)      # Deep sleep, unconscious
    THETA = (4.0, 8.0)      # Drowsiness, meditation, creativity
    ALPHA = (8.0, 13.0)     # Relaxed awareness, focus
    BETA = (13.0, 30.0)     # Active thinking, concentration
    GAMMA = (30.0, 100.0)   # High-level cognitive processing

class MentalState(Enum):
    """Recognized mental states"""
    FOCUSED = "focused"
    RELAXED = "relaxed"
    MEDITATIVE = "meditative"
    CREATIVE = "creative"
    STRESSED = "stressed"
    DROWSY = "drowsy"
    ALERT = "alert"
    CONFUSED = "confused"
    EMOTIONAL = "emotional"
    NEUTRAL = "neutral"

@dataclass
class EEGReading:
    """Single EEG data point"""
    timestamp: float
    raw_signal: float
    filtered_signal: float
    frequency_bands: Dict[BrainWaveType, float]
    mental_state: MentalState
    confidence: float
    attention_level: float
    meditation_level: float

@dataclass
class BrainCommand:
    """Brain-derived command for ONI"""
    command_type: str
    parameters: Dict
    confidence: float
    timestamp: float
    mental_context: MentalState

class MindLinkInterface:
    """Interface for Mind-link EEG device using LUFA USB framework"""
    
    def __init__(self, usb_vendor_id: int = 0x03EB, usb_product_id: int = 0x2044):
        # Mind-link uses LUFA framework, likely presents as USB HID or CDC device
        self.vendor_id = usb_vendor_id  # LUFA default VID
        self.product_id = usb_product_id  # LUFA default PID
        self.usb_device = None
        self.is_connected = False
        self.is_streaming = False
        self.data_buffer = []
        self.callbacks = []
        self.sample_rate = 256  # Common EEG sampling rate
        
        # Mind-link specific configuration
        self.config = {
            'firmware_version': None,
            'electrode_channels': 1,  # Single channel EEG
            'adc_resolution': 10,     # 10-bit ADC typical for AVR
            'reference_voltage': 3.3,
            'gain': 1000,             # Typical EEG amplifier gain
            'highpass_filter': 0.5,   # Hz
            'lowpass_filter': 50.0,   # Hz
            'notch_filter': 60.0      # Hz (US power line)
        }
        
    def connect(self) -> bool:
        """Connect to Mind-link USB device using LUFA framework"""
        try:
            import usb.core
            import usb.util
            
            # Find Mind-link device
            self.usb_device = usb.core.find(idVendor=self.vendor_id, idProduct=self.product_id)
            
            if self.usb_device is None:
                logger.error("Mind-link device not found. Check USB connection and firmware.")
                return False
            
            # Configure the device
            self.usb_device.set_configuration()
            
            # Get device info
            self.config['firmware_version'] = self.usb_device.bcdDevice
            
            self.is_connected = True
            logger.info(f"Connected to Mind-link device (FW: {self.config['firmware_version']:#x})")
            
            # Send initialization command to firmware
            self._send_command(b'INIT')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Mind-link USB device: {e}")
            logger.info("Make sure Mind-link.hex is flashed to the device with:")
            logger.info("avrdude -p m32u2 -P usb -c usbasp -u -U flash:w:Mind-link.hex")
            return False
    
    def disconnect(self):
        """Disconnect from Mind-link USB device"""
        if self.usb_device:
            try:
                # Send shutdown command
                self._send_command(b'STOP')
                # Release USB interface
                usb.util.dispose_resources(self.usb_device)
                self.usb_device = None
                self.is_connected = False
                logger.info("Disconnected from Mind-link device")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
    
    def _send_command(self, command: bytes):
        """Send command to Mind-link firmware"""
        if not self.is_connected or not self.usb_device:
            return
        
        try:
            # Send command via USB control transfer
            self.usb_device.ctrl_transfer(
                bmRequestType=0x40,  # Host to device, vendor specific
                bRequest=0x01,       # Custom command
                wValue=0,
                wIndex=0,
                data_or_wLength=command
            )
        except Exception as e:
            logger.error(f"Failed to send command {command}: {e}")
    
    def start_streaming(self):
        """Start streaming EEG data from Mind-link firmware"""
        if not self.is_connected:
            logger.error("Not connected to Mind-link device")
            return
        
        # Send start streaming command to firmware
        self._send_command(b'START')
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_usb_data)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info(f"Started EEG streaming at {self.sample_rate}Hz")
    
    def stop_streaming(self):
        """Stop streaming EEG data"""
        self.is_streaming = False
        if self.is_connected:
            self._send_command(b'STOP')
        logger.info("Stopped EEG data streaming")
    
    def _stream_usb_data(self):
        """Stream data from Mind-link via USB bulk transfer"""
        while self.is_streaming:
            try:
                # Read data from USB endpoint (typically endpoint 1 for LUFA)
                data = self.usb_device.read(0x81, 64, timeout=100)  # 64 bytes max, 100ms timeout
                
                if data:
                    # Parse the data packet from firmware
                    for i in range(0, len(data), 2):  # 2 bytes per sample (16-bit)
                        if i + 1 < len(data):
                            # Combine bytes into 16-bit sample (little-endian)
                            raw_sample = data[i] | (data[i+1] << 8)
                            
                            # Convert to voltage (assuming 10-bit ADC, 3.3V reference)
                            voltage = (raw_sample / 1024.0) * self.config['reference_voltage']
                            
                            # Convert to microvolts (typical EEG range)
                            microvolt_sample = (voltage - 1.65) * 1000000 / self.config['gain']
                            
                            eeg_reading = self._process_raw_sample(microvolt_sample)
                            if eeg_reading:
                                self._notify_callbacks(eeg_reading)
                
                time.sleep(1.0 / self.sample_rate)  # Maintain sample rate
                
            except Exception as e:
                if self.is_streaming:  # Only log if we're supposed to be streaming
                    logger.error(f"Error streaming USB data: {e}")
                break
    
    def _process_raw_data(self, raw_data: str) -> EEGReading:
        """Process raw EEG data from Mind-link"""
        try:
            # Parse raw data (format depends on Mind-link implementation)
            signal_value = float(raw_data)
            
            # Apply filtering and feature extraction
            filtered_signal = self._apply_filters(signal_value)
            frequency_bands = self._extract_frequency_bands(filtered_signal)
            mental_state = self._classify_mental_state(frequency_bands)
            attention = self._calculate_attention(frequency_bands)
            meditation = self._calculate_meditation(frequency_bands)
            
            return EEGReading(
                timestamp=time.time(),
                raw_signal=signal_value,
                filtered_signal=filtered_signal,
                frequency_bands=frequency_bands,
                mental_state=mental_state,
                confidence=0.8,  # Placeholder
                attention_level=attention,
                meditation_level=meditation
            )
        except Exception as e:
            logger.error(f"Error processing raw data: {e}")
            return None
    
    def _apply_filters(self, signal: float) -> float:
        """Apply signal filtering (bandpass, notch, etc.)"""
        # Implement digital filtering here
        return signal * 0.95  # Placeholder
    
    def _extract_frequency_bands(self, signal: float) -> Dict[BrainWaveType, float]:
        """Extract frequency band amplitudes using FFT"""
        # Implement FFT-based frequency analysis
        return {
            BrainWaveType.DELTA: abs(signal * 0.1),
            BrainWaveType.THETA: abs(signal * 0.2),
            BrainWaveType.ALPHA: abs(signal * 0.4),
            BrainWaveType.BETA: abs(signal * 0.3),
            BrainWaveType.GAMMA: abs(signal * 0.1)
        }
    
    def _classify_mental_state(self, bands: Dict[BrainWaveType, float]) -> MentalState:
        """Classify mental state based on frequency bands"""
        if bands[BrainWaveType.ALPHA] > 0.6:
            return MentalState.FOCUSED
        elif bands[BrainWaveType.THETA] > 0.5:
            return MentalState.CREATIVE
        elif bands[BrainWaveType.BETA] > 0.7:
            return MentalState.STRESSED
        else:
            return MentalState.NEUTRAL
    
    def _calculate_attention(self, bands: Dict[BrainWaveType, float]) -> float:
        """Calculate attention level (0-1)"""
        return min(1.0, (bands[BrainWaveType.BETA] + bands[BrainWaveType.GAMMA]) / 2)
    
    def _calculate_meditation(self, bands: Dict[BrainWaveType, float]) -> float:
        """Calculate meditation level (0-1)"""
        return min(1.0, (bands[BrainWaveType.ALPHA] + bands[BrainWaveType.THETA]) / 2)
    
    def add_callback(self, callback):
        """Add callback for EEG data"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, eeg_reading: EEGReading):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(eeg_reading)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

class ONIMindLinkBridge:
    """Bridge between Mind-link and ONI AGI System"""
    
    def __init__(self, oni_websocket_url: str = "ws://localhost:8080/oni"):
        self.mindlink = MindLinkInterface()
        self.oni_url = oni_websocket_url
        self.oni_websocket = None
        self.is_connected_to_oni = False
        
        # Use case processors
        self.use_case_processors = {
            "thought_control": self._process_thought_control,
            "emotion_detection": self._process_emotion_detection,
            "focus_training": self._process_focus_training,
            "creative_collaboration": self._process_creative_collaboration,
            "vr_navigation": self._process_vr_navigation,
            "meditation_guidance": self._process_meditation_guidance,
            "learning_optimization": self._process_learning_optimization,
            "stress_monitoring": self._process_stress_monitoring,
            "cognitive_enhancement": self._process_cognitive_enhancement,
            "dream_analysis": self._process_dream_analysis,
            "memory_enhancement": self._process_memory_enhancement,
            "social_interaction": self._process_social_interaction,
            "health_monitoring": self._process_health_monitoring,
            "accessibility_control": self._process_accessibility_control,
            "research_assistance": self._process_research_assistance,
            "therapeutic_intervention": self._process_therapeutic_intervention,
            "neural_feedback": self._process_neural_feedback,
            "consciousness_study": self._process_consciousness_study,
            "brain_training": self._process_brain_training,
            "predictive_modeling": self._process_predictive_modeling
        }
        
        # Initialize Mind-link callbacks
        self.mindlink.add_callback(self._on_eeg_data)
    
    async def connect_to_oni(self):
        """Connect to ONI via WebSocket"""
        try:
            self.oni_websocket = await websockets.connect(self.oni_url)
            self.is_connected_to_oni = True
            logger.info("Connected to ONI AGI System")
            
            # Send initial handshake
            await self._send_to_oni({
                "type": "mindlink_connection",
                "message": "Mind-link BCI connected to ONI",
                "capabilities": list(self.use_case_processors.keys())
            })
        except Exception as e:
            logger.error(f"Failed to connect to ONI: {e}")
    
    async def _send_to_oni(self, data: Dict):
        """Send data to ONI"""
        if self.oni_websocket and self.is_connected_to_oni:
            try:
                await self.oni_websocket.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending to ONI: {e}")
    
    def _on_eeg_data(self, eeg_reading: EEGReading):
        """Process incoming EEG data"""
        if not eeg_reading:
            return
        
        # Process through all active use cases
        for use_case, processor in self.use_case_processors.items():
            try:
                brain_command = processor(eeg_reading)
                if brain_command:
                    asyncio.create_task(self._send_brain_command(brain_command))
            except Exception as e:
                logger.error(f"Error in use case {use_case}: {e}")
    
    async def _send_brain_command(self, command: BrainCommand):
        """Send brain command to ONI"""
        await self._send_to_oni({
            "type": "brain_command",
            "command": command.command_type,
            "parameters": command.parameters,
            "confidence": command.confidence,
            "timestamp": command.timestamp,
            "mental_context": command.mental_context.value
        })
    
    # ========================================
    # USE CASE PROCESSORS (15+ implementations)
    # ========================================
    
    def _process_thought_control(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 1: Direct thought control of ONI systems"""
        if eeg.mental_state == MentalState.FOCUSED and eeg.attention_level > 0.8:
            return BrainCommand(
                command_type="execute_thought_command",
                parameters={
                    "attention_level": eeg.attention_level,
                    "focus_intensity": eeg.frequency_bands[BrainWaveType.BETA]
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_emotion_detection(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 2: Real-time emotion detection and response"""
        emotional_intensity = eeg.frequency_bands[BrainWaveType.GAMMA]
        if emotional_intensity > 0.3:
            return BrainCommand(
                command_type="emotion_response",
                parameters={
                    "emotional_state": eeg.mental_state.value,
                    "intensity": emotional_intensity,
                    "suggested_response": "empathetic_engagement"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_focus_training(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 3: Attention and focus training with ONI feedback"""
        if eeg.attention_level < 0.5:
            return BrainCommand(
                command_type="focus_training_feedback",
                parameters={
                    "current_focus": eeg.attention_level,
                    "target_focus": 0.8,
                    "training_suggestion": "increase_beta_waves"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_creative_collaboration(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 4: Creative collaboration between brain and ONI"""
        if eeg.mental_state == MentalState.CREATIVE:
            return BrainCommand(
                command_type="creative_collaboration",
                parameters={
                    "creativity_level": eeg.frequency_bands[BrainWaveType.THETA],
                    "inspiration_mode": "theta_dominant",
                    "collaboration_type": "artistic_generation"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_vr_navigation(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 5: Navigate ONI's VR environments with thought"""
        if eeg.attention_level > 0.6:
            return BrainCommand(
                command_type="vr_navigation",
                parameters={
                    "navigation_intent": "forward" if eeg.frequency_bands[BrainWaveType.BETA] > 0.5 else "pause",
                    "attention_focus": eeg.attention_level,
                    "movement_confidence": eeg.confidence
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_meditation_guidance(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 6: AI-guided meditation based on brain states"""
        if eeg.meditation_level > 0.3:
            return BrainCommand(
                command_type="meditation_guidance",
                parameters={
                    "meditation_depth": eeg.meditation_level,
                    "alpha_coherence": eeg.frequency_bands[BrainWaveType.ALPHA],
                    "guidance_type": "breathing_rhythm"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_learning_optimization(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 7: Optimize learning based on cognitive load"""
        cognitive_load = eeg.frequency_bands[BrainWaveType.BETA] + eeg.frequency_bands[BrainWaveType.GAMMA]
        if cognitive_load > 0.8:
            return BrainCommand(
                command_type="learning_optimization",
                parameters={
                    "cognitive_load": cognitive_load,
                    "learning_adjustment": "reduce_complexity",
                    "break_suggestion": True
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_stress_monitoring(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 8: Monitor and manage stress levels"""
        if eeg.mental_state == MentalState.STRESSED:
            return BrainCommand(
                command_type="stress_management",
                parameters={
                    "stress_level": eeg.frequency_bands[BrainWaveType.BETA],
                    "intervention_type": "breathing_exercise",
                    "urgency": "moderate"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_cognitive_enhancement(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 9: Enhance cognitive performance with neurofeedback"""
        if eeg.attention_level < 0.7:
            return BrainCommand(
                command_type="cognitive_enhancement",
                parameters={
                    "enhancement_target": "attention",
                    "current_performance": eeg.attention_level,
                    "neurofeedback_type": "beta_training"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_dream_analysis(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 10: Analyze and interpret dream patterns"""
        if eeg.mental_state == MentalState.DROWSY:
            return BrainCommand(
                command_type="dream_analysis",
                parameters={
                    "sleep_stage": "rem" if eeg.frequency_bands[BrainWaveType.THETA] > 0.5 else "nrem",
                    "dream_intensity": eeg.frequency_bands[BrainWaveType.GAMMA],
                    "analysis_type": "pattern_recognition"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_memory_enhancement(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 11: Enhance memory formation and recall"""
        if eeg.frequency_bands[BrainWaveType.THETA] > 0.4:
            return BrainCommand(
                command_type="memory_enhancement",
                parameters={
                    "memory_type": "encoding" if eeg.attention_level > 0.6 else "consolidation",
                    "theta_power": eeg.frequency_bands[BrainWaveType.THETA],
                    "enhancement_protocol": "theta_burst_stimulation"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_social_interaction(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 12: Enhance social interactions with brain-state awareness"""
        if eeg.mental_state in [MentalState.EMOTIONAL, MentalState.STRESSED]:
            return BrainCommand(
                command_type="social_interaction",
                parameters={
                    "social_readiness": eeg.attention_level,
                    "emotional_state": eeg.mental_state.value,
                    "interaction_suggestion": "empathetic_response"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_health_monitoring(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 13: Monitor neurological health indicators"""
        anomaly_score = abs(eeg.raw_signal - eeg.filtered_signal)
        if anomaly_score > 0.5:
            return BrainCommand(
                command_type="health_monitoring",
                parameters={
                    "anomaly_detected": True,
                    "anomaly_score": anomaly_score,
                    "health_recommendation": "consult_specialist"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_accessibility_control(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 14: Assistive technology control for disabilities"""
        if eeg.attention_level > 0.7:
            return BrainCommand(
                command_type="accessibility_control",
                parameters={
                    "control_type": "cursor_movement",
                    "control_strength": eeg.attention_level,
                    "accessibility_mode": "motor_assistance"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_research_assistance(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 15: Assist scientific research with brain data"""
        return BrainCommand(
            command_type="research_assistance",
            parameters={
                "data_contribution": {
                    "subject_id": "anonymous",
                    "mental_state": eeg.mental_state.value,
                    "frequency_bands": {k.name: v for k, v in eeg.frequency_bands.items()},
                    "attention": eeg.attention_level,
                    "meditation": eeg.meditation_level
                },
                "research_type": "consciousness_study"
            },
            confidence=eeg.confidence,
            timestamp=eeg.timestamp,
            mental_context=eeg.mental_state
        )
    
    def _process_therapeutic_intervention(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 16: Therapeutic interventions based on brain states"""
        if eeg.mental_state == MentalState.STRESSED:
            return BrainCommand(
                command_type="therapeutic_intervention",
                parameters={
                    "intervention_type": "guided_relaxation",
                    "severity": "moderate",
                    "therapeutic_protocol": "progressive_muscle_relaxation"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_neural_feedback(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 17: Provide real-time neural feedback"""
        return BrainCommand(
            command_type="neural_feedback",
            parameters={
                "feedback_type": "visual_audio",
                "brain_state": eeg.mental_state.value,
                "feedback_intensity": eeg.confidence,
                "target_state": "optimal_performance"
            },
            confidence=eeg.confidence,
            timestamp=eeg.timestamp,
            mental_context=eeg.mental_state
        )
    
    def _process_consciousness_study(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 18: Study consciousness and self-awareness"""
        consciousness_metric = (eeg.frequency_bands[BrainWaveType.GAMMA] + 
                              eeg.frequency_bands[BrainWaveType.BETA]) / 2
        if consciousness_metric > 0.3:
            return BrainCommand(
                command_type="consciousness_study",
                parameters={
                    "consciousness_level": consciousness_metric,
                    "awareness_indicators": {
                        "gamma_activity": eeg.frequency_bands[BrainWaveType.GAMMA],
                        "beta_activity": eeg.frequency_bands[BrainWaveType.BETA]
                    },
                    "study_protocol": "neural_correlates_of_consciousness"
                },
                confidence=eeg.confidence,
                timestamp=eeg.timestamp,
                mental_context=eeg.mental_state
            )
        return None
    
    def _process_brain_training(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 19: Personalized brain training programs"""
        return BrainCommand(
            command_type="brain_training",
            parameters={
                "training_type": "attention_training" if eeg.attention_level < 0.6 else "meditation_training",
                "current_level": eeg.attention_level,
                "training_difficulty": "adaptive",
                "session_progress": eeg.confidence
            },
            confidence=eeg.confidence,
            timestamp=eeg.timestamp,
            mental_context=eeg.mental_state
        )
    
    def _process_predictive_modeling(self, eeg: EEGReading) -> Optional[BrainCommand]:
        """Use Case 20: Predictive modeling of mental states"""
        return BrainCommand(
            command_type="predictive_modeling",
            parameters={
                "prediction_target": "mental_state_transition",
                "current_state": eeg.mental_state.value,
                "predicted_state": "focused" if eeg.attention_level > 0.5 else "relaxed",
                "prediction_confidence": eeg.confidence,
                "time_horizon": "30_seconds"
            },
            confidence=eeg.confidence,
            timestamp=eeg.timestamp,
            mental_context=eeg.mental_state
        )
    
    async def start_bridge(self):
        """Start the Mind-link to ONI bridge"""
        logger.info("Starting ONI-MindLink Bridge...")
        
        # Connect to Mind-link
        if not self.mindlink.connect():
            logger.error("Failed to connect to Mind-link device")
            return False
        
        # Connect to ONI
        await self.connect_to_oni()
        
        # Start streaming
        self.mindlink.start_streaming()
        
        logger.info("ONI-MindLink Bridge active - All 20 use cases enabled")
        return True
    
    async def stop_bridge(self):
        """Stop the bridge"""
        logger.info("Stopping ONI-MindLink Bridge...")
        self.mindlink.stop_streaming()
        self.mindlink.disconnect()
        
        if self.oni_websocket:
            await self.oni_websocket.close()
        
        logger.info("ONI-MindLink Bridge stopped")

# ========================================
# MAIN EXECUTION
# ========================================

async def main():
    """Main execution function"""
    bridge = ONIMindLinkBridge()
    
    try:
        await bridge.start_bridge()
        
        # Keep the bridge running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await bridge.stop_bridge()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        await bridge.stop_bridge()

if __name__ == "__main__":
    print("="*60)
    print("ONI-MINDLINK BRAIN-COMPUTER INTERFACE")
    print("="*60)
    print("🧠 Connecting human consciousness to AGI...")
    print("🔗 20 revolutionary use cases enabled")
    print("🚀 The future of human-AI collaboration starts now")
    print("="*60)
    
    asyncio.run(main())
