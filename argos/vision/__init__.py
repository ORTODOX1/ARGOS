"""
argos.vision -- Computer-vision pipeline for defect detection and classification.

Provides edge-optimised ONNX inference for real-time hull and machinery
inspection aboard autonomous inspection robots.
"""

from argos.vision.classifier import DefectClassifier
from argos.vision.detector import DefectDetector

__all__ = [
    "DefectDetector",
    "DefectClassifier",
]
