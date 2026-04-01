"""Tests for argos.vision.detector module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argos.vision.detector import (
    DefectDetector,
    DefectType,
    Detection,
    NautilusQuantLUT,
)


class TestDefectType:
    """Verify defect type enum values match model label map."""

    def test_enum_values(self):
        assert DefectType.CORROSION.value == "corrosion"
        assert DefectType.CRACK.value == "crack"
        assert DefectType.FOULING.value == "fouling"
        assert DefectType.LEAK.value == "leak"
        assert DefectType.DEFORMATION.value == "deformation"

    def test_enum_count(self):
        assert len(DefectType) == 5


class TestDetection:
    """Verify Detection dataclass creation and attributes."""

    def test_creation(self):
        det = Detection(
            x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0,
            defect_type=DefectType.CRACK, confidence=0.87,
        )
        assert det.defect_type is DefectType.CRACK
        assert det.confidence == pytest.approx(0.87)
        assert det.x_max - det.x_min == pytest.approx(90.0)

    def test_frozen(self):
        det = Detection(0, 0, 1, 1, DefectType.LEAK, 0.5)
        with pytest.raises(AttributeError):
            det.confidence = 0.9  # type: ignore[misc]


class TestNautilusQuantLUT:
    """Verify INT8 dequantisation look-up table logic."""

    def test_identity_passthrough(self):
        lut = NautilusQuantLUT()
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = lut.dequantize(tensor)
        np.testing.assert_array_almost_equal(result, tensor)

    def test_scale_and_offset(self):
        lut = NautilusQuantLUT(
            scale=np.array([0.5], dtype=np.float32),
            zero_point=np.array([10], dtype=np.int8),
        )
        tensor = np.array([20, 30], dtype=np.int8)
        result = lut.dequantize(tensor)
        expected = (np.array([20, 30], dtype=np.float32) - 10.0) * 0.5
        np.testing.assert_array_almost_equal(result, expected)


class TestNMS:
    """Verify non-maximum suppression with overlapping boxes."""

    @pytest.fixture()
    def detector(self):
        with patch("onnxruntime.InferenceSession") as mock_sess:
            mock_instance = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "images"
            mock_instance.get_inputs.return_value = [mock_input]
            mock_sess.return_value = mock_instance
            return DefectDetector()

    def test_suppresses_overlapping(self, detector):
        boxes = np.array([
            [0, 0, 100, 100],
            [5, 5, 105, 105],
            [200, 200, 300, 300],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
        kept = detector._apply_nms(boxes, scores)
        assert len(kept) == 2
        assert 0 in kept
        assert 2 in kept

    def test_keeps_all_non_overlapping(self, detector):
        boxes = np.array([
            [0, 0, 10, 10],
            [100, 100, 110, 110],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        kept = detector._apply_nms(boxes, scores)
        assert len(kept) == 2
