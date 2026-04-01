"""
argos.vision.classifier -- Defect severity classification.

Accepts cropped regions produced by the detector and assigns a severity
grade using a lightweight ONNX classification head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from argos.config import EdgeProcessorConfig
from argos.vision.detector import Detection


class Severity(IntEnum):
    """Severity levels aligned with DNV-GL classification rules."""

    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass(frozen=True)
class ClassifiedDefect:
    """Detection enriched with a severity grade and explanation score."""

    detection: Detection
    severity: Severity
    severity_confidence: float
    shap_top_feature: str


class DefectClassifier:
    """Classifies detected defect regions by severity.

    Parameters
    ----------
    config:
        Edge processor configuration pointing to the classifier ONNX model.
    """

    _SEVERITY_MAP: dict[int, Severity] = {
        0: Severity.MINOR,
        1: Severity.MODERATE,
        2: Severity.SEVERE,
        3: Severity.CRITICAL,
    }

    _SHAP_FEATURES: list[str] = [
        "texture_entropy",
        "edge_density",
        "colour_deviation",
        "area_ratio",
    ]

    def __init__(self, config: EdgeProcessorConfig | None = None) -> None:
        self._cfg = config or EdgeProcessorConfig()
        self._session = ort.InferenceSession(
            self._cfg.classifier_model_path,
            providers=[self._cfg.execution_provider],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._input_size: tuple[int, int] = (224, 224)

    def _crop_and_preprocess(
        self, frame: np.ndarray, det: Detection
    ) -> np.ndarray:
        """Extract the detected region, resize, and normalise."""
        x1 = max(int(det.x_min), 0)
        y1 = max(int(det.y_min), 0)
        x2 = min(int(det.x_max), frame.shape[1])
        y2 = min(int(det.y_max), frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, self._input_size)
        blob = resized.astype(np.float32) / 255.0
        return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    def classify(
        self, frame: np.ndarray, detections: Sequence[Detection]
    ) -> list[ClassifiedDefect]:
        """Assign severity to each detection.

        Parameters
        ----------
        frame:
            Original BGR image from which detections were produced.
        detections:
            Bounding-box detections from ``DefectDetector.detect``.

        Returns
        -------
        list[ClassifiedDefect]
            Each detection paired with its severity grade.
        """
        results: list[ClassifiedDefect] = []
        for det in detections:
            blob = self._crop_and_preprocess(frame, det)
            logits = self._session.run(None, {self._input_name: blob})[0][0]
            probs = _softmax(logits)
            cls_idx = int(np.argmax(probs))
            severity = self._SEVERITY_MAP.get(cls_idx, Severity.MINOR)
            # Stub: top SHAP feature based on argmax of auxiliary head
            shap_idx = int(np.argmax(logits[:len(self._SHAP_FEATURES)]))
            results.append(
                ClassifiedDefect(
                    detection=det,
                    severity=severity,
                    severity_confidence=float(probs[cls_idx]),
                    shap_top_feature=self._SHAP_FEATURES[shap_idx],
                )
            )
        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
