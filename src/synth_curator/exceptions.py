"""Library errors. Library code raises these instead of calling sys.exit."""


class SynthCuratorError(Exception):
    """Base class for all library errors."""


class MissingMetricsError(SynthCuratorError):
    """Pool lacks metric columns that the weights require."""


class MissingFeaturesError(SynthCuratorError):
    """Feature tensors absent, or not covering the pool's filenames."""


class WeightsError(SynthCuratorError):
    """Weights or tournament predictions malformed / insufficient."""
