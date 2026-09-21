"""synth-curator: multi-axiom curation of synthetic medical images."""
from .model_interface import Model, model_from_function
from .pipeline import SynthCurator

__version__ = "0.0.1.dev0"
__all__ = ["SynthCurator", "Model", "model_from_function"]
