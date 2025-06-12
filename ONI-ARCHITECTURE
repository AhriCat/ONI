# 🧠 ONI Architecture Overview

**ONI** (Omni-Neural Intelligence) is a large-scale, modular AGI framework designed for reasoning, perception, agency, and self-regulation. Unlike the compact OniMini, the full ONI system incorporates persistent learning, dynamic tool injection, multi-agent scheduling, and executive control to serve as a lifelong generalist intelligence.

---

## 🧩 Major Architectural Subsystems

### 1. Multimodal Sensory Cortex
- Handles text, audio, vision, and external device inputs.
- Modules:
  - `oni_audio`: Stream/audio event processing
  - `oni_vision`: Image/video screen analysis, OCR, and camera input
  - `oni_NLP`: Tokenization, parsing, and language reasoning
  - `oni_MM_attention`: Integrates modalities via attention + fusion

---

### 2. Memory and Internal World
- Modules:
  - `oni_memory`: Long-term storage, associative memory, spatial map, and replay buffer
  - `ExperienceReplayBuffer`: RL-specific buffer for interaction data
  - Spatial room memory system for environment modeling and navigable context

---

### 3. Decision & Reasoning Engine
- Central controller: `OniMicro`
- Powered by:
  - Recurrent Q-Network (`LSTM`) + attention
  - Dynamic energy-modulated synaptic layers
  - Modular forward pass over all perception modules
- Outputs include:
  - Action selection
  - Command generation
  - Context-aware reasoning traces

---

### 4. Executive Function & Task Management
- `oni_executive_function`: Maps inferred tasks to subsystem invocations
- `perform_task()` and `execute_task()`:
  - Parse natural instructions
  - Map to command lambdas (e.g., `draw`, `search`, `read`, `animate`)
  - Executes in multithreaded or prioritized queues

---

### 5. Self-Regulation and Homeostasis
- `oni_homeostasis`: Tracks system state (fatigue, reward, novelty, etc.)
- Energy and emotional signals modulate:
  - Memory consolidation
  - Output intensity
  - Curiosity and exploration in RL agents

---

### 6. Tool Use and External Control
- Built-in control tools:
  - `pyautogui`: Mouse, keyboard, automation
  - `selenium`: Browser navigation and interaction
  - `pytesseract`: OCR from screen or documents
  - `cv2`, `PIL`, `PyPDF2`: Vision and document processing
- Autonomous environment interfacing via:
  - Audio I/O
  - Vision screen capture
  - PDF/code reading
  - Image generation (`pipe`)
  - Animation rendering (`pipeline`)

---

### 7. Dynamic Agent Injection
- `DynamicModuleInjector`:
  - Injects pretrained HuggingFace or local models at runtime
  - Enables plugin-like upgrades (LLMs, vision models, coders)
  - Modular forward logic based on model structure

---

### 8. Reinforcement Learning Subsystem
- `RecurrentQNetwork`: LSTM-based Q-learning
- Supports:
  - DQN training
  - Target network updates
  - Reward modeling from perceptual/emotional feedback
- RL tightly integrates with executive logic and experience buffer

---

### 9. Meta-Cognition and Error Recovery
- `oni_metacognition`: Reflects on reasoning quality and internal consistency
- `safe_process()` + `fallback_with_coder()`:
  - Primary NLP failure fallback to code generation
  - Stepwise planning and response refinement
- Supports self-evaluation and retry logic

---

## 🔁 High-Level Dataflow

```
[User Input / Environment Data]
            ↓
[Multimodal Cortex: NLP + Vision + Audio]
            ↓
[Memory Query + State Construction]
            ↓
[Attention + Fusion → Decision Engine]
            ↓
[Executive Function → Subsystem Invocation]
            ↓
[Output, Action, Rendering, Learning]
```

---

## 🛠 Design Principles

- **Modularity**: All agents and subsystems are pluggable and replaceable.
- **Multi-agent readiness**: Each module can act independently and coordinate.
- **Feedback loops**: Emotional and energy regulation feed into decision systems.
- **Dynamic scalability**: Models can be injected/swapped without reboot.
- **Interactivity**: System can browse, draw, click, animate, code, or speak.
- **Persistence**: Memories evolve via replay, compression, and environmental mapping.

---

## 🔮 Planned / Hidden Features

- Spatial memory expansion into 3D or VR
- Knowledge distillation across agent modules
- AGI-to-human interface abstraction
- Real-time behavioral cloning or human imitation
- Economic reasoning and trading agents

---

**ONI is designed not just as an AI model, but as a scaffold for future general intelligence**: able to reflect, plan, sense, act, and grow in complexity without central failure.
