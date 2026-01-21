# ONI — Operant Neural Intelligence

ONI is a **modular repository of neural network components** designed to support research and development toward agentic AI systems. It provides a collection of PyTorch modules covering memory, reasoning, attention, emotion modeling, vision, audio, and tool integration—intended as building blocks that researchers and developers can study, adapt, and compose.

**Important:** ONI is an experimental research project. The modules vary in maturity—some are well-developed implementations, others are prototypes or conceptual sketches. This is not a production-ready AGI system; it's a toolkit of components that may be useful for building toward more capable AI architectures.

---

## What ONI Actually Is

ONI provides:

- **Reusable neural modules** for memory systems, attention mechanisms, reasoning components, and multimodal processing
- **Reference implementations** of concepts from cognitive architecture research
- **Integration scaffolding** for combining modules into larger systems
- **Training utilities** and example workflows

ONI is *not* a complete AI system you can deploy. It's a component library and research codebase.

---

## Module Overview

### Memory Systems (`modules/memory/`)

Memory is one of ONI's more developed areas, offering multiple complementary approaches:

| Module | Description |
|--------|-------------|
| `oni_memory.py` | Core memory manager integrating working, episodic, and semantic memory |
| `oni_memoryv2.py` | Extended memory system with additional consolidation mechanisms |
| `episodic_memory.py` | Stores sequential, autobiographical experiences with embedding layers |
| `semantic_memory.py` | Abstract knowledge storage with pattern finding |
| `working_memory_module.py` | Short-term context buffer |
| `hopfield.py` | Sparse and continuous Hopfield networks for pattern completion |
| `memory_consolidator.py` | Sleep-inspired memory consolidation |
| `fading_memory.py` | Temporal decay mechanisms |
| `snapshot_memory.py` | State preservation |
| `spatial_memory.py` | Location-aware memory representations |

### Reasoning & NLP (`modules/NLP/`)

Reasoning modules explore different approaches to structured inference:

| Module | Description |
|--------|-------------|
| `oni_metacognition.py` | Self-monitoring, abductive reasoning, hypothesis generation |
| `causal_reasoning.py` | Causal graph construction and intervention modeling |
| `analogical_reasoning.py` | Cross-domain analogical mapping |
| `counterfactual_reasoning.py` | "What if" scenario evaluation |
| `multi_step_planning.py` | Goal decomposition and plan generation |
| `oni_chain_of_thought.py` | Step-by-step reasoning traces |
| `conflict_graph.py` | Principle conflict detection and resolution |

### Attention Mechanisms (`modules/attention/`)

Several attention variants for different use cases:

| Module | Description |
|--------|-------------|
| `temporal_tri_attention.py` | Hierarchical attention at global, local, and temporal scales with sparsity |
| `multi_modal_attention.py` | Cross-modal attention fusion |
| `efficient_attention.py` | Computational optimizations |
| `concept_similarity_memory_attention.py` | Memory-augmented attention |
| `reformer_attention.py` | LSH-based efficient attention |

### Emotion & Values (`modules/emotion/`)

Modules for modeling affective states and ethical reasoning:

| Module | Description |
|--------|-------------|
| `oni_emotions.py` | Valence-arousal emotional modeling with state tracking and decay |
| `oni_compassion.py` | Multi-agent ethical framework with Agency, Capability, Suffering (ACS) metrics; Pareto optimization for multi-stakeholder decisions |

### Vision (`modules/vision/`)

| Module | Description |
|--------|-------------|
| `oni_vision.py` | Vision transformer with OCR integration |
| `vision_core.py` | Extended vision processing utilities |

### Audio (`modules/audio/`)

| Module | Description |
|--------|-------------|
| `oni_audio.py` | Audio processing with TTS/STT hooks |

### Dynamics & Homeostasis

| Module | Description |
|--------|-------------|
| `oni_homeostasis.py` | System stability regulation with anomaly detection, Hebbian-inspired adaptation |
| `dynamics/energy_based_synapse.py` | Energy-modulated synaptic layers |
| `dynamics/oni_dynLayer.py` | Dynamic neural layer implementations |

### World Model (`modules/WorldModel/`)

| Module | Description |
|--------|-------------|
| `latent_space_operations.py` | Latent diffusion operations, memory-augmented generation |
| `world_modeler.py` | Environment state representation |

### Additional Components

| Directory | Contents |
|-----------|----------|
| `modules/haptics/` | Tactile system modeling |
| `modules/robotics/` | IoT/robot controller interfaces |
| `modules/skills/` | Dynamic module injection, specialized skills |
| `modules/feedforward/` | FFN variants including hyper-networks |

---

## Prototype Architectures (`prototypellms/`)

Experimental transformer variants and alternative architectures:

| File | Description |
|------|-------------|
| `HypergraphNLP.py` | Hypergraph-based language model using hyperedge convolutions |
| `GatedRecurrentTransformer.py` | Transformer with gated recurrence |
| `KroneckerTransformer.py` | Kronecker-factorized attention |
| `SudoQuantumMicroTransformer.py` | Quantum-inspired micro-transformer (classical simulation) |

---

## Tools (`tools/`)

Integration utilities for external systems:

| Tool | Purpose |
|------|---------|
| `RAG.py` | Retrieval-augmented generation with PDF processing |
| `git_integration.py` | Git repository operations |
| `playwright_automation.py` | Browser automation |
| `ros_integration.py` | ROS robotics middleware |
| `docker_manager.py` | Container orchestration |
| `blender_controller.py`, `unity_controller.py`, `unreal_controller.py` | 3D engine integration |
| `navigation.py` | Spatial navigation |
| `search.py` | Web search utilities |
| `security_sandboxing.py` | Execution sandboxing |
| `tool_chaining.py` | Multi-tool workflow orchestration |

---

## Training (`trainingLoops/`)

Example training scripts:

- `train_causal_reasoning.py` — Causal reasoning module training with intervention queries
- `train_analog_reasoning.py` — Analogical reasoning training
- `train_tactile_system.py` — Haptic system training

---

## Agent Workflows (`agentWorkflows/`)

Minimal agent loop examples:

- `minimal_recursive_self_directed_loop.py` — Basic autonomous task loop
- `deep_research.py` — Multi-step research workflow
- `code_review.py` — Automated code review
- `github_code_integration.py` — GitHub operations

---

## Domain Applications (`oniapps/`)

Specialized application modules (varying completeness):

- `oni_science_lab.py` — Scientific computation utilities
- `oni_bioinformatics.py` — Bioinformatics tools
- `oni_molecular_dynamics.py` — Molecular simulation
- `oni_quantum_simulator.py` — Quantum circuit simulation
- `oni_trading_dash.py` — Trading dashboard prototype

---

## Blockchain Components (`chain/`)

Experimental distributed training infrastructure:

- Proof-of-Compute consensus for training contribution tracking
- Model update versioning
- Smart contract for contribution rewards (`contracts/ONIToken.sol`)

**Note:** The blockchain components are in early prototype stage.

---

## Installation

```bash
git clone https://github.com/ahricat/oni.git
cd oni

# Install dependencies
pip install -r requirements.txt

# Or use the install script
chmod +x scripts/install.sh
./scripts/install.sh
```

### Requirements

- Python 3.12+
- PyTorch with CUDA support (recommended)
- 16GB+ RAM (48GB+ for full system)
- See `requirements.txt` for package dependencies

---

## Usage Examples

The `usage/` directory contains integration examples:

- `TkinterChat.ipynb` — Simple chat interface
- `roboticsController.py` — Robot control example
- `oni_VRCNPC.cs` — Unity VR NPC integration
- `unrealNPC.cpp` — Unreal Engine NPC

---

## Project Structure

```
ONI/
├── modules/           # Core neural modules
│   ├── memory/        # Memory systems
│   ├── NLP/           # Reasoning & language
│   ├── attention/     # Attention mechanisms
│   ├── emotion/       # Affective modeling
│   ├── vision/        # Visual processing
│   ├── audio/         # Audio processing
│   ├── dynamics/      # Dynamic layers
│   ├── WorldModel/    # World modeling
│   ├── haptics/       # Tactile systems
│   ├── robotics/      # Robot interfaces
│   └── skills/        # Skill modules
├── prototypellms/     # Experimental architectures
├── tools/             # External integrations
├── oniapps/           # Domain applications
├── chain/             # Blockchain components
├── trainingLoops/     # Training scripts
├── agentWorkflows/    # Agent examples
├── tests/             # Test suite
├── usage/             # Usage examples
├── frontend/          # Web interface
└── scripts/           # Utility scripts
```

---

## Current Status & Limitations

ONI is research software with the following caveats:

- **Varying maturity:** Some modules are well-tested, others are stubs or prototypes
- **Integration gaps:** Not all modules compose seamlessly
- **Documentation:** Inline documentation varies; some modules need more explanation
- **Performance:** Not optimized for production deployment
- **Dependencies:** Some modules have specific hardware requirements
- **examples of composition** see oni.py or all in one models folder to see how you may compose your own onis. 

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

Pantheum License — see `LICENSE` for terms. Research and educational use; users are responsible for ethical application.

---

## Acknowledgments

ONI draws inspiration from cognitive architecture research, including work on memory consolidation, metacognition, and multi-agent coordination. The compassion framework builds on ideas from AI safety research around value alignment and multi-stakeholder optimization.
