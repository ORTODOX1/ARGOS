"""
ARGOS -- Autonomous Robot for General Onboard Surveillance.

Autonomous ship inspection system that detects hull and machinery defects
using edge-deployed ONNX vision models, integrates with POSEIDON-DIAG for
sensor fusion, and escalates unknown anomalies to the SYNIZ TRIZ engine.
"""

__version__ = "0.4.1"
__author__ = "Herman Doronin"
__license__ = "MIT"

from argos.inspector import InspectionEngine

__all__ = [
    "InspectionEngine",
    "__version__",
]
