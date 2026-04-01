"""
argos.vision -- Computer-vision pipeline for defect detection and classification.

Provides edge-optimised ONNX inference for real-time hull and machinery
inspection aboard autonomous inspection robots.
"""

from argos.vision.classifier import ClassifiedDefect, DefectClassifier, Severity
from argos.vision.detector import DefectDetector, DefectType, Detection

__all__ = [
    "DefectDetector",
    "DefectClassifier",
    "Detection",
    "DefectType",
    "Severity",
    "ClassifiedDefect",
]
