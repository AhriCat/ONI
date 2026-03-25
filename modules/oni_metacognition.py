# File: modules/oni_metacognition.py
# [EDITOR] Re-export shim. The real module lives at modules/NLP/oni_metacognition.py
# but ONI.py imports from modules.oni_metacognition. This bridge fixes the broken import.

import sys
import os

# Ensure the NLP subdirectory is on the path
_nlp_dir = os.path.join(os.path.dirname(__file__), 'NLP')
if _nlp_dir not in sys.path:
    sys.path.insert(0, _nlp_dir)

from modules.NLP.oni_metacognition import (
    MetaCognitionModule,
    AbductiveReasoning,
    AnalogicalReasoning,
    CausalInferenceEngine
)

__all__ = [
    'MetaCognitionModule',
    'AbductiveReasoning',
    'AnalogicalReasoning',
    'CausalInferenceEngine'
]
