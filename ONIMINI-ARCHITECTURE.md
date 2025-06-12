# 🧠 OniMini Architecture Overview

**OniMini** is a lightweight, modular cognitive architecture designed to function as a general-purpose agent on a single GPU. It integrates emotional regulation, reasoning, memory, and multimodal perception into a unified inference loop. Though compact, it simulates key aspects of human-like cognition including meta-reflection, emotional feedback, and persistent memory.

---

## 🔩 System Modules

### 1. Multimodal Preprocessing
- Converts raw input (text, image, audio) into a shared latent space.
- Uses:
  - A small language model for text
  - Vision transformer for image understanding
  - Spectrogram-based ResNet for audio
- All modalities are projected to a 1536-dimensional common space.

### 2. Persistent Memory
- Stores vectorized memories tied to linguistic or perceptual experiences.
- Includes:
  - Quality and novelty filtering
  - Embedding-based retrieval via nearest neighbors
  - Compression, archiving, and restoration capabilities
- Supports long-term learning and experience accumulation.

### 3. Emotional State and Energy
- Tracks emotion as a function of valence and arousal.
- Modulates model outputs based on:
  - Detected emotional tone
  - Fatigue and energy levels
  - Recent emotional history
- Feeds emotional signals back into learning, modifying the internal state.

### 4. Meta-Cognition
- Monitors reasoning integrity and ethical alignment.
- Tracks a set of internal “principles” and evaluates conflict in context.
- Returns confidence scores and conflict summaries.
- Enables internal checks and course corrections during inference.

### 5. Chain of Thought Engine
- Recurrently iterates through short, coherent thoughts.
- Uses:
  - Recurrent LSTM backbone
  - Sparse attention for long-term focus
  - Context gating with memory and perception
- Produces interpretable, modular reasoning steps.

### 6. Reasoning-Orchestration (`ONI`)
- Central controller that:
  - Processes input via the preprocessor
  - Modulates through emotional and metacognitive layers
  - Calls the Chain of Thought for recursive inference
  - Uses memory and multimodal context to inform decisions
- Provides tool-use capabilities (typing, clicking, browsing).

---

## 🔁 Data Flow Summary

```
[Input: Text/Image/Audio]
         ↓
[Multimodal Preprocessor]
         ↓
[Memory Retrieval] ↔ [Persistent Memory]
         ↓
[Emotion + Energy Modulation]
         ↓
[Chain of Thought Reasoning]
         ↓
[Meta-Cognition Check]
         ↓
[Final Output Generation]
```

---

## 🧪 Design Priorities

- **Compactness**: Runs on a single GPU, optimized for speed and memory.
- **Grounding**: Multimodal input ensures sensory context.
- **Emotion-aware**: Modulates cognition and behavior.
- **Conflict-conscious**: Internal principles checked during inference.
- **Extensibility**: Easy to swap or extend components.

---

## 🔮 Future Directions

- Add speech synthesis and voice input
- Connect to VR / physical environments
- Implement more abstract planning and simulation capabilities
- Introduce memory-guided goal persistence
- Enable local multi-agent interactions

---

OniMini is a step toward building a thoughtful, internally coherent, emotionally responsive artificial general intelligence scaffolded on a modular neuro-symbolic stack.
