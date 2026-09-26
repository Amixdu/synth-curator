"""synth-curator: utility-weighted curation of synthetic image pools."""
from .model_interface import Model, ProductionModel, model_from_function
from .pipeline import SynthCurator

__version__ = "0.0.1.dev0"
__all__ = ["SynthCurator", "Model", "ProductionModel", "model_from_function"]
