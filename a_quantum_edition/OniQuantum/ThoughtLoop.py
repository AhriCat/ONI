# oni_thought_loop.py

import time
import threading
from core import QuantumONICore
import numpy as np
import queue
import random

class QuantumThoughtLoop:
    def __init__(self, n_qubits=4, sleep_time=2.0):
        self.qoni = QuantumONICore(n_qubits=n_qubits)
        self.sleep_time = sleep_time
        self.running = False
        self.user_input_queue = queue.Queue()
        self.internal_state = []
        self.lock = threading.Lock()
        self.thread = None

    def generate_internal_query(self):
        # Simulate internal thoughts based on time, entropy, or prior embeddings
        prompts = [
            "What is the meaning of entanglement?",
            "Am I conscious?",
            "Can I predict my next memory?",
            "What did I learn yesterday?",
            "What does stillness feel like?",
            "What should I do next?"
        ]
        return random.choice(prompts)

    def process_query(self, query, source='internal'):
        print(f"[{source}] query: {query}")
        vec = self.qoni.embed_text(query)
        closest = self.qoni.memory['text'][-1] if self.qoni.memory['text'] else "none"
        self.qoni.add_to_memory(vec, modality="text")
        if hasattr(self.qoni.classifier, 'support_'):
            label = self.qoni.classify_text(query)
            print(f"[ONI-Q] label → {label}")
        else:
            print(f"[ONI-Q] memory size → {len(self.qoni.memory['text'])}")

    def user_input_listener(self):
        while self.running:
            try:
                query = input("[user] > ").strip()
                if query:
                    self.user_input_queue.put(query)
            except KeyboardInterrupt:
                self.running = False
                break

    def main_thought_loop(self):
        self.running = True
        listener_thread = threading.Thread(target=self.user_input_listener, daemon=True)
        listener_thread.start()

        print("[ONI-Q] Initiating autonomous stream of thought...")

        while self.running:
            with self.lock:
                try:
                    if not self.user_input_queue.empty():
                        query = self.user_input_queue.get_nowait()
                        self.process_query(query, source="user")
                    else:
                        internal_query = self.generate_internal_query()
                        self.process_query(internal_query, source="oni")
                except Exception as e:
                    print(f"[ONI-Q] error: {e}")
            time.sleep(self.sleep_time)

    def start(self):
        if not self.thread:
            self.thread = threading.Thread(target=self.main_thought_loop)
            self.thread.start()

    def stop(self):
        self.running = False
        print("[ONI-Q] Stopping cognitive loop...")
        if self.thread:
            self.thread.join()
