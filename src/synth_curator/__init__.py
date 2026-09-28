"""synth-curator: utility-weighted curation of synthetic image pools."""
__version__ = "0.0.1.dev0"   # set before the imports below - weights_bundle.py reads it during import

from .model_interface import Model, ProductionModel, model_from_function
from .pipeline import SynthCurator

__all__ = ["SynthCurator", "Model", "ProductionModel", "model_from_function"]
