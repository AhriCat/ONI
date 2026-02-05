# --------------------------------------------------------------
# oni_wrapper.py
# --------------------------------------------------------------
"""
ONI Emotion / Compassion Wrapper

Usage
-----
>>> from my_model import MyModel               # any torch.nn.Module
>>> from oni_wrapper import patch_model

>>> # Wrap the model – you can pass the hidden dimension used by
>>> # your model (required for the EmotionalLayer embedding) and
>>> # an optional initial energy budget.
>>> wrapped = patch_model(MyModel(),
...                      hidden_dim=896,
...                      init_energy=120)

>>> # Tensor input (standard forward)
>>> out = wrapped(torch.randn(1, 10))

>>> # Text input (the wrapper will call the underlying classifier
>>> # pipeline, run the emotion layer, apply the energy module,
>>> # and return a dict with classification, emotional state &
>>> # feedback stats.)
>>> result = wrapped("I am thrilled about the new project!")
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

# ------------------------------------------------------------------
# 1️⃣ Import the ONI modules (will raise a clear error if the package
#    is not installed).
# ------------------------------------------------------------------
try:
    # Emotion‑related sub‑modules
    from oni.modules.emotion.oni_emotions import (
        EmotionalLayer,
        EnergyModule,
        EmotionalFeedbackModule,
    )
    # High‑level compassion system (used for multi‑agent planning, etc.)
    from oni.modules.emotion.oni_compassion import ONICompassionSystem
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import ONI emotion/compassion modules. "
        "Make sure the `oni` package (with the sub‑modules "
        "`oni.modules.emotion.oni_emotions` and "
        "`oni.modules.emotion.oni_compassion`) is installed."
    ) from exc


# ------------------------------------------------------------------
# 2️⃣ Mixin that adds the three emotion modules to any nn.Module.
# ------------------------------------------------------------------
class ONIEmotionMixin(nn.Module):
    """
    Mixin that equips a model with:

    * ``self.emotion_layer`` – an ``EmotionalLayer`` that modulates the
      hidden representation with valence / arousal information.
    * ``self.energy_module`` – an ``EnergyModule`` that attenuates the
      output when the agent’s energy is low.
    * ``self.feedback_module`` – an ``EmotionalFeedbackModule`` that
      can apply gradient‑level feedback based on the current emotional
      state.

    The mixin expects the *base* model to return a **tensor** when its
    ``forward`` is called with a tensor, and a **list of dicts** when
    called with a string (the format returned by a Hugging‑Face
    text‑classification pipeline).  The wrapper normalises both cases
    into the same internal flow.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int,
        init_energy: float = 100.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.device = device or torch.device("cpu")

        # ------------------------------------------------------------------
        # Initialise the three ONI sub‑modules.
        # ------------------------------------------------------------------
        self.emotion_layer = EmotionalLayer(hidden_dim).to(self.device)
        self.energy_module = EnergyModule(init_energy=init_energy).to(self.device)
        self.feedback_module = EmotionalFeedbackModule().to(self.device)

        # If the base model already has a `.to(...)` method we forward the
        # call so that the whole wrapper lives on the same device.
        self.to(self.device)

    # ------------------------------------------------------------------
    # Helper – move *all* sub‑modules (including the wrapped base model) to
    # the same device when ``.to(...)`` is invoked on the wrapper.
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs) -> "ONIEmotionMixin":  # type: ignore[override]
        self.base_model = self.base_model.to(*args, **kwargs)
        self.emotion_layer = self.emotion_layer.to(*args, **kwargs)
        self.energy_module = self.energy_module.to(*args, **kwargs)
        self.feedback_module = self.feedback_module.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ------------------------------------------------------------------
    # Core forward – works for tensors **and** plain strings.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Union[torch.Tensor, str, list[str]],
        *args,
        **kwargs,
    ) -> Any:
        """
        Parameters
        ----------
        x : torch.Tensor | str | list[str]
            *Tensor* → processed directly by the wrapped model.  
            *String / list of strings* → passed through the text‑classification
            pipeline that lives inside ``self.base_model`` (the wrapper
            expects the base model to behave like a Hugging‑Face pipeline).

        Returns
        -------
        Any
            • For tensor input: the energy‑modulated hidden representation.  
            • For text input: a dict containing the classifier output,
              the current emotional state and feedback statistics.
        """
        # --------------------------------------------------------------
        # 1️⃣ Tensor path – classic nn.Module forward
        # --------------------------------------------------------------
        if isinstance(x, torch.Tensor):
            # Base model produces a hidden representation (any shape)
            hidden = self.base_model(x, *args, **kwargs)

            # Emotional modulation (valence / arousal) + energy attenuation
            hidden = self.emotion_layer(hidden, emotion_input=None)
            hidden = self.energy_module(
                hidden,
                self.emotion_layer.arousal.item(),
                self.emotion_layer.valence.item(),
                self.emotion_layer.emotional_state,
            )
            return hidden

        # --------------------------------------------------------------
        # 2️⃣ Text path – we assume the wrapped model is a Hugging‑Face
        #    pipeline that returns a list of dicts: [{'label': ..., 'score': ...}, ...]
        # --------------------------------------------------------------
        if isinstance(x, (str, list)):
            # Normalise to list of strings for the pipeline
            texts = [x] if isinstance(x, str) else list(x)

            # Run the underlying classifier (the pipeline is stored in
            # ``self.base_model`` – if the object is not a pipeline we try
            # to call ``.__call__`` which works for most Hugging‑Face objects)
            try:
                classifier_outputs = self.base_model(texts)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Wrapped base model does not behave like a Hugging‑Face "
                    "pipeline. Ensure it can be called with a list of strings."
                ) from exc

            # For simplicity we process **one** string at a time and
            # return a list of result dicts (kept compatible with the
            # original demo code you posted).
            results = []
            for txt, clf_out in zip(texts, classifier_outputs):
                # Dummy placeholder hidden tensor – the real model would
                # produce something meaningful; we keep the shape compatible
                # with the emotion layer embedding dimension.
                placeholder_hidden = torch.randn(1, self.emotion_layer.emotion_embedding.num_embeddings).to(self.device)

                # Run the emotional layer *with* the classifier output
                mod_hidden = self.emotion_layer(placeholder_hidden, emotion_input=clf_out)

                # Apply the energy module
                mod_hidden = self.energy_module(
                    mod_hidden,
                    self.emotion_layer.arousal.item(),
                    self.emotion_layer.valence.item(),
                    self.emotion_layer.emotional_state,
                )

                # Apply (optional) feedback – this mutates the emotion_layer
                # parameters according to the current emotional state.
                emotional_state = self.emotion_layer.emotional_state
                self.feedback_module.apply_emotional_feedback(
                    self.emotion_layer,  # module to receive feedback
                    {
                        "valence": self.emotion_layer.valence.item(),
                        "intensity": emotional_state.emotion_intensity,
                        "current_emotion": emotional_state.current_emotion,
                    },
                )

                # Assemble the public result dict
                results.append(
                    {
                        "text": txt,
                        "classification": clf_out,
                        "emotional_state": {
                            "current_emotion": (
                                list(self.emotion_layer.EMOTION_VA_MAP.keys())[emotional_state.current_emotion]
                                if emotional_state.current_emotion is not None
                                else "none"
                            ),
                            "intensity": emotional_state.emotion_intensity,
                            "valence": self.emotion_layer.valence.item(),
                            "arousal": self.emotion_layer.arousal.item(),
                        },
                        "feedback_stats": self.feedback_module.get_feedback_stats(),
                    }
                )
            return results

        raise TypeError(
            f"Unsupported input type `{type(x)}`. Expected torch.Tensor, str or list of str."
        )


# ------------------------------------------------------------------
# 3️⃣ Convenience function – patch any existing nn.Module (or pipeline)
# ------------------------------------------------------------------
def patch_model(
    base_model: nn.Module,
    hidden_dim: int,
    init_energy: float = 100.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Return a **new object** that behaves like ``base_model`` but also
    exposes the ONI emotion/compassion machinery.

    Parameters
    ----------
    base_model : nn.Module
        Your original model (e.g. a `torch.nn.Linear` stack, a
        `transformers` pipeline, etc.).
    hidden_dim : int
        Dimensionality of the hidden representation that will be fed
        into ``EmotionalLayer``.  Must match the size of the tensor that
        ``base_model`` returns when called with a tensor.
    init_energy : float, optional
        Starting energy budget for the ``EnergyModule`` (default = 100).
    device : torch.device, optional
        If omitted the wrapper will use ``torch.device('cpu')`` or the
        device of ``base_model`` if it already has parameters.

    Returns
    -------
    nn.Module
        An instance of ``ONIEmotionMixin`` that forwards all
        arguments to ``base_model`` and then runs the ONI pipeline.
    """
    # If the user passed a pipeline that lives on CPU but wants CUDA,
    # move it now (otherwise the mix‑in will raise a device‑mismatch error).
    if device is None:
        # Try to infer device from the first parameter of the base model,
        # otherwise fall back to CPU.
        try:
            first_param = next(base_model.parameters())
            device = first_param.device
        except Exception:
            device = torch.device("cpu")

    # Build the wrapper – we use multiple inheritance so that the
    # resulting object is an ``nn.Module`` (required for most PyTorch
    # utilities) *and* carries the mix‑in behaviour.
    class WrappedModel(ONIEmotionMixin, type(base_model)):
        pass

    # Initialise the combined class.
    wrapped = WrappedModel(base_model, hidden_dim, init_energy=init_energy, device=device)

    # Preserve the original ``__repr__`` for nicer debugging.
    wrapped.__repr__ = lambda self=wrapped: (
        f"{self.__class__.__name__}(base={base_model!r}, "
        f"hidden_dim={hidden_dim}, init_energy={init_energy})"
    )
    return wrapped


# ------------------------------------------------------------------
# 4️⃣ (Optional) Small helper to expose the high‑level ONI system.
# ------------------------------------------------------------------
def create_oni_system(
    reality_state: Optional[Dict[str, Any]] = None,
) -> ONICompassionSystem:
    """
    Convenience shortcut for users that also need the full
    ``ONICompassionSystem`` (multi‑agent planning, self‑modification,
    etc.).

    Parameters
    ----------
    reality_state : dict, optional
        Initial world state; if omitted an empty dict is used.

    Returns
    -------
    ONICompassionSystem
    """
    return ONICompassionSystem(reality_state=reality_state)


# ------------------------------------------------------------------
# Example (run only when executed as a script)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Simple demo – a dummy linear model that outputs a 896‑dim vector
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 896)

        def forward(self, x):
            return self.fc(x)

    # Wrap the dummy model
    wrapped = patch_model(DummyModel(), hidden_dim=896, init_energy=120)

    # Tensor path
    tensor_out = wrapped(torch.randn(1, 10))
    print("Tensor output shape:", tensor_out.shape)

    # Text path (requires a Hugging‑Face pipeline – we mock one here)
    class MockPipeline:
        def __call__(self, texts):
            # Return a *list* of classification dicts for each input string
            return [
                [{"label": "joy", "score": 0.92}, {"label": "surprise", "score": 0.08}]
                for _ in texts
            ]

    # Replace the underlying model with a mock pipeline to see the text flow
    wrapped.base_model = MockPipeline()
    txt_res = wrapped("I am absolutely thrilled!")
    print("\nText result:")
    print(txt_res)
