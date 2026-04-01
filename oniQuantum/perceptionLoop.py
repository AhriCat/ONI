# perception_loop.py

import time
import threading
from core import QuantumONICore
import numpy as np
import random

class OniQuantumPerceptionLoop:
    def __init__(self, n_qubits=4, mode='autonomous'):
        self.qoni = QuantumONICore(n_qubits=n_qubits)
        self.running = False
        self.mode = mode  # "autonomous" or "interactive"
        self.idle_threshold = 5
        self.idle_counter = 0
        self.engagement_score = 0.0
        self.lock = threading.Lock()

    def perceive_environment(self):
        # Placeholder: simulate perception (random noise, placeholder for webcam/audio feed)
        sensory_event = random.random()
        self.engagement_score = sensory_event  # Real logic: actual model-based signal detection
        if sensory_event > 0.85:
            dummy_text = "the singularity is near"
            print(f"[perception] Text detected: {dummy_text}")
            embedded = self.qoni.embed_text(dummy_text)
            self.qoni.add_to_memory(embedded, modality="text")
            self.idle_counter = 0
        else:
            self.idle_counter += 1

    def internal_thought(self):
        # Placeholder: "think" by running time series memory scans or self-query
        sequence = np.random.rand(10)
        sim = self.qoni.detect_timeseries_patterns(sequence)
        print(f"[internal] Thought pattern similarity: {sim}")

    def user_interaction_check(self):
        if self.mode == "interactive":
            try:
                text = input("[user] Type something for ONI to process (or press Enter to skip): ")
                if text.strip():
                    vec = self.qoni.embed_text(text)
                    label = self.qoni.classify_text(text) if hasattr(self.qoni.classifier, 'support_') else "Unknown"
                    print(f"[ONI-Q] Predicted label: {label}")
                    self.qoni.add_to_memory(vec, modality="text")
                    self.idle_counter = 0
            except Exception as e:
                print(f"[error] user input failed: {e}")

    def main_loop(self, sleep_time=2):
        self.running = True
        print("[ONI-Q] Starting continuous perception loop...")
        while self.running:
            with self.lock:
                self.perceive_environment()
                if self.idle_counter >= self.idle_threshold:
                    print("[ONI-Q] Entering idle state, invoking user or introspection...")
                    self.user_interaction_check()
                    self.internal_thought()
            time.sleep(sleep_time)

    def stop(self):
        with self.lock:
            self.running = False
            print("[ONI-Q] Perception loop halted.")

