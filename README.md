# ONI — Operant Neural Intelligence

ONI is a **modular repository of neural network components** designed to support research and development toward agentic AI systems. It provides a collection of PyTorch modules covering memory, reasoning, attention, emotion modeling, vision, audio, and tool integration—intended as building blocks that researchers and developers can study, adapt, and compose.

**Important:** ONI is an experimental research project. The modules vary in maturity—some are well-developed implementations, others are prototypes or conceptual sketches but they're being updated continuously. This could be a production-ready AGI system depending on how you configure and train it; it's basically just a toolkit of components that may be useful for building toward more capable AI architectures. Example of how this may be configured can be found in oniQuantum/, oniMini/ and ONI.py. Think legos or bionical or gundam or megazoid. They're made up of interconnected parts that you would consider building blocks. ONI is designed to have atomic modules that can be used in a wide variety of situations depending on how you configure them and their paths.  

---

### ONI provides:

- **Reusable neural modules** for memory systems, attention mechanisms, reasoning components, and multimodal processing
- **Reference implementations** of concepts from cognitive architecture research
- **Integration scaffolding** for combining modules into larger systems
- **Training utilities** and example workflows



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
| `oni_metacognition.py` | Self-reflection, abductive/analogical/causal reasoning, hypothesis generation, confidence estimation, `diagnose_self()` for evolution |
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
| `latentresidualblock.py` | LatentCompressed Attention and Residual attention with MHC implementation blk|
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
| `GatedRecurrentTransformer.py` | Transformer with gated rnn |
| `KroneckerTransformer.py` | to be used in donut -> kronecker transform for cycloidal positional bias model with ternary tokenizer |
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

## Evolution System (`evolution/`) — ONI-DGM

ONI includes a **Darwin-Gödel Machine**-style self-improvement loop that uses ONI's own metacognition to identify weaknesses and apply targeted GRPO training via the [Superintelligence Oven](https://github.com/AhriCat/super_intelligence_oven).

**Key differentiator from vanilla DGM:** instead of an external model diagnosing failures, ONI diagnoses itself. As ONI improves, its diagnosis improves — a virtuous cycle.

### Architecture

```
evaluate → self-diagnose (metacognition) → propose → train (Oven GRPO) → re-evaluate → archive
```

Parent selection follows DGM's sigmoid-child-proportional algorithm:
```
P(parent_i) = sigmoid(score_i) × 1/(1 + children_i) / Z
```

### Components

| File | Purpose |
|------|---------|
| `oni_dgm_outer.py` | Main `ONIDarwinGodelMachine` orchestrator |
| `oni_self_diagnosis.py` | Self-referential diagnosis via `MetaCognitionModule.diagnose_self()` |
| `oni_archive.py` | DGM-style variant archive with open-ended exploration |
| `oni_oven_integration.py` | Wraps `superintelligence_oven.bake()` for GRPO; mock SGD fallback |
| `oni_evaluation.py` | Multi-modal weighted evaluator (NLP 30%, Vision 20%, Audio 15%, …) |
| `oni_robotics_trainer.py` | Diffusion trajectory generation with smoothness/haptic metrics |
| `improvement_proposal.py` | `ImprovementProposal`, `EvaluationLog`, `ONIVariant` data structures |
| `config.py` | All hyperparameters in one place |
| `monitor.py` | CLI progress reporter |
| `utils/` | Parent selection, weight patch, benchmark helpers |
| `tests/` | 11 tests covering diagnosis, archive, and parent selection |

### Quick start

```bash
# Test mode — no GPU or model weights required
python -m trainingLoops.train_evolution \
    --archive_dir ./oni_archive \
    --max_generations 5 \
    --verbose

# Full evolution with real model and Oven
pip install -e ../super_intelligence_oven
python -m trainingLoops.train_evolution \
    --archive_dir ./oni_archive \
    --initial_variant ./ \
    --model_path ./weights/oni_v1.pt \
    --max_generations 80

# Monitor progress
python -m evolution.monitor ./oni_archive
```

### Oven integration

The Superintelligence Oven provides the GRPO training back-end with:
- Hot-swappable local teachers (Qwen3-4B, Qwen3-8B-Q4, diffusion motion)
- Remote agent swarm (critic / adversary / specialist / style / curriculum)
- QwenEmbedVerifier semantic reward (text modalities)
- Per-module teacher routing (`ModuleType.ROBOTICS` → diffusion teacher, `ModuleType.METACOGNITION` → big teacher, etc.)

When the oven is not installed, the integration falls back to a mock 10-step SGD pass so the full evolution loop can be validated without external dependencies.

---

## Training (`trainingLoops/`)

| Script | Purpose |
|--------|---------|
| `train_evolution.py` | **ONI-DGM evolution entrypoint** — full self-improvement loop |
| `train_causal_reasoning.py` | Causal reasoning module training with intervention queries |
| `train_analog_reasoning.py` | Analogical reasoning training |
| `train_tactile_system.py` | Haptic system training |

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

> **Important:** The blockchain components are a **research prototype** simulating proof-of-compute and ledger concepts for AI training coordination. They do **not** deploy real smart contracts to any live network and do **not** handle real cryptocurrency or tokens of monetary value. The Solidity contract and Python simulation classes are reference implementations only. Do not use in production financial or governance systems without a full security audit.

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
├── ONI.py                 # Main model (composes all modules)
├── modules/               # Core neural modules
│   ├── oni_metacognition.py  # Re-export shim (fixes import path)
│   ├── memory/            # Memory systems
│   ├── NLP/               # Reasoning & language (metacognition lives here)
│   ├── attention/         # Attention mechanisms
│   ├── emotion/           # Affective modeling
│   ├── vision/            # Visual processing
│   ├── audio/             # Audio processing
│   ├── dynamics/          # Dynamic layers
│   ├── WorldModel/        # World modeling
│   ├── haptics/           # Tactile systems
│   ├── robotics/          # Robot interfaces
│   └── skills/            # Skill modules
├── evolution/             # ONI-DGM self-improvement system
│   ├── oni_dgm_outer.py   # Main evolution orchestrator
│   ├── oni_self_diagnosis.py
│   ├── oni_archive.py
│   ├── oni_oven_integration.py
│   ├── oni_evaluation.py
│   ├── oni_robotics_trainer.py
│   ├── improvement_proposal.py
│   ├── config.py
│   ├── monitor.py
│   ├── utils/
│   └── tests/
├── prototypellms/         # Experimental architectures
├── tools/                 # External integrations
├── oniapps/               # Domain applications
├── chain/                 # Blockchain prototype (research only)
├── trainingLoops/         # Training scripts incl. train_evolution.py
├── agentWorkflows/        # Agent examples
├── evaluation/            # Module fitness evaluation
├── tests/                 # Test suite
├── usage/                 # Usage examples
├── frontend/              # Web interface
└── scripts/               # Utility scripts
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
PS sorry for any past posts being disengenuous seeming and the website. Some of it was very enthusiastic thinking others are still in the works as later plans or unreleased models. 
---

## Acknowledgments

ONI draws inspiration from cognitive architecture research, including work on memory consolidation, metacognition, and multi-agent coordination. The compassion framework builds on ideas from AI safety research around value alignment and multi-stakeholder optimization.
