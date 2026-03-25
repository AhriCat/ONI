# File: evolution/config.py
"""
ONI-DGM Evolution Configuration.
Adjust these parameters based on available compute and desired evolution speed.
"""

DEFAULT_CONFIG = {
    # Archive settings
    'archive': {
        'lambda_param': 10.0,   # Sigmoid sharpness for parent selection
        'alpha_0': 0.5,         # Sigmoid midpoint
    },

    # Evolution settings
    'evolution': {
        'max_generations': 80,
        'parents_per_generation': 2,
        'parallel_workers': 2,
        'parent_selection_method': 'score_child_prop',
    },

    # Training settings
    'training': {
        'steps_per_proposal': 500,
        'group_size': 8,
        'kl_beta': 0.1,
        'learning_rate': 1e-6,
        # Oven teacher to use (must match a TeacherConfig name)
        'initial_teacher': 'small_qwen3_4b',
    },

    # Diagnosis settings
    'diagnosis': {
        'epistemic_threshold': 0.7,
        'conflict_threshold': 0.5,
        'min_proposal_priority': 0.3,
    },

    # Evaluation settings
    'evaluation': {
        'nlp_tasks': 50,
        'vision_tasks': 30,
        'audio_tasks': 20,
        'robotics_tasks': 20,
        'integration_tasks': 10,
    },

    # Oven agent swarm weights
    'agent_weights': {
        'critic': 0.30,
        'adversary': 0.20,
        'specialist': 0.20,
        'style': 0.15,
        'curriculum': 0.15,
    },
}
