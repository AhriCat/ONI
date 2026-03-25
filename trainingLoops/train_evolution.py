# File: trainingLoops/train_evolution.py
"""
Evolution training entrypoint for ONI-DGM.

Usage:
    # Test mode (minimal mock model, no GPU required)
    python -m trainingLoops.train_evolution --archive_dir ./oni_archive \\
        --max_generations 5 --verbose

    # Full evolution with real ONI model
    python -m trainingLoops.train_evolution --archive_dir ./oni_archive \\
        --initial_variant ./ --model_path ./weights/oni_v1.pt \\
        --max_generations 80

Monitor progress:
    python -m evolution.monitor ./oni_archive
"""
import argparse
import json
import logging
import os
import sys
import torch

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution.oni_dgm_outer import ONIDarwinGodelMachine
from evolution.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, verbose: bool = False):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evolution.log")
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_oni_model(model_path: str = None):
    """
    Load the ONI model.

    Attempts to import the real ONI class. Falls back to a MinimalONI
    test model so the evolution pipeline can be validated without a full
    ONI installation.
    """
    logger = logging.getLogger(__name__)

    if model_path and os.path.exists(model_path):
        try:
            from ONI import ONI
            model = ONI()
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded ONI model from {model_path}")
            return model, None
        except Exception as e:
            logger.warning(f"Failed to load ONI model from {model_path}: {e}")

    logger.info("Using MinimalONI test model (real weights not loaded)")

    class MinimalONI(torch.nn.Module):
        def __init__(self, hidden_dim=512):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
            self._init_metacognition(hidden_dim)

        def _init_metacognition(self, hidden_dim):
            try:
                # Try importing the real metacognition module
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from modules.NLP.oni_metacognition import MetaCognitionModule
                self.metacognition_module = MetaCognitionModule(hidden_dim)
                logging.getLogger(__name__).info(
                    "MinimalONI: real MetaCognitionModule loaded"
                )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"MetaCognitionModule not importable ({e}), using mock"
                )
                self.metacognition_module = self._make_mock_meta(hidden_dim)

        @staticmethod
        def _make_mock_meta(hidden_dim):
            class _MockMeta:
                def __init__(self):
                    self.hidden_dim = hidden_dim

                def diagnose_self(self, task_input, task_output,
                                  expected_output=None, task_id="unknown"):
                    return {
                        'explanation': {
                            'reasoning_strategy': {'name': 'Mock'},
                            'confidence': {'score': 0.5},
                            'uncertainty': {'epistemic': 0.5, 'aleatoric': 0.5},
                            'conflicts': {'detected': []}
                        },
                        'error_signal': {},
                        'weakness_indicators': [('low_confidence', 0.5)],
                        'task_id': task_id
                    }

                class abductive_reasoning:
                    @staticmethod
                    def __call__(x):
                        return x, {'best_score': torch.tensor(0.3)}

            return _MockMeta()

        def forward(self, x):
            return torch.tanh(self.fc(x))

    return MinimalONI(), None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ONI-DGM Evolution — evolve ONI using Darwin-Godel Machine"
    )
    parser.add_argument(
        "--archive_dir", type=str, default="./oni_archive",
        help="Directory for the variant archive"
    )
    parser.add_argument(
        "--initial_variant", type=str, default="./",
        help="Path to the initial ONI variant (repo root)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to ONI model weights (.pt)"
    )
    parser.add_argument(
        "--max_generations", type=int, default=None,
        help="Override max_generations from config"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging"
    )
    parser.add_argument(
        "--config_override", type=str, default=None,
        help="Path to a JSON file with config overrides"
    )
    args = parser.parse_args()

    setup_logging(args.archive_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config_override and os.path.exists(args.config_override):
        with open(args.config_override) as f:
            overrides = json.load(f)
        # Deep merge top-level keys
        for k, v in overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
        logger.info(f"Applied config overrides from {args.config_override}")

    # Load model
    oni_model, tokenizer = load_oni_model(args.model_path)

    # Run evolution
    dgm = ONIDarwinGodelMachine(
        oni_model=oni_model,
        tokenizer=tokenizer,
        archive_dir=args.archive_dir,
        initial_variant_path=args.initial_variant,
        config=config
    )

    best_variant = dgm.run(max_generations=args.max_generations)

    if best_variant:
        logger.info(
            f"Evolution complete. Best variant: {best_variant.variant_id} "
            f"(score={best_variant.overall_score:.4f})"
        )
    else:
        logger.info("Evolution complete. No variants produced.")


if __name__ == "__main__":
    main()
