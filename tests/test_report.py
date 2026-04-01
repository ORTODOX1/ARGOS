"""Tests for argos.report module."""

import datetime as dt
import json

import pytest

from argos.report import DefectEntry, InspectionReport, ReportGenerator
from argos.vision.classifier import ClassifiedDefect, Severity
from argos.vision.detector import DefectType, Detection


@pytest.fixture()
def sample_defects():
    """Create a set of classified defects for testing."""
    det_crack = Detection(10, 20, 110, 120, DefectType.CRACK, 0.92)
    det_corrosion = Detection(200, 50, 350, 180, DefectType.CORROSION, 0.78)
    return [
        ClassifiedDefect(det_crack, Severity.CRITICAL, 0.88, "edge_density"),
        ClassifiedDefect(det_corrosion, Severity.MODERATE, 0.75, "texture_entropy"),
    ]


@pytest.fixture()
def sample_report(sample_defects):
    """Build a report from sample defects using ReportGenerator."""
    gen = ReportGenerator()
    return gen.build(
        timestamp=dt.datetime(2025, 6, 15, 10, 30, 0, tzinfo=dt.timezone.utc),
        gps_lat=45.4315,
        gps_lon=12.3456,
        classified_defects=sample_defects,
        unknown_analyses=[{"detection": "stub", "hypotheses": []}],
        sensor_context={"engine_rpm": 1200.0, "exhaust_temp_c": 350.0},
    )


class TestDefectEntry:
    """Verify DefectEntry creation from ClassifiedDefect."""

    def test_from_classified(self, sample_defects):
        entry = DefectEntry.from_classified(sample_defects[0])
        assert entry.defect_type == "crack"
        assert entry.severity == "critical"
        assert entry.confidence == pytest.approx(0.92)
        assert len(entry.bbox) == 4


class TestInspectionReport:
    """Verify report assembly, computed fields, and serialisation."""

    def test_total_defects(self, sample_report):
        assert sample_report.total_defects == 3  # 2 classified + 1 unknown

    def test_max_severity(self, sample_report):
        assert sample_report.max_severity == "critical"

    def test_max_severity_none(self):
        report = InspectionReport(
            report_id="abc123",
            timestamp="2025-06-15T10:30:00+00:00",
            gps_lat=0.0, gps_lon=0.0,
            defects=[], unknown_analyses=[],
            sensor_context={},
        )
        assert report.max_severity == "none"

    def test_to_dict_structure(self, sample_report):
        data = sample_report.to_dict()
        assert "report_id" in data
        assert "gps" in data
        assert data["gps"]["lat"] == pytest.approx(45.4315)
        assert len(data["defects"]) == 2

    def test_to_json_roundtrip(self, sample_report):
        json_str = sample_report.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_defects"] == 3
        assert parsed["max_severity"] == "critical"

    def test_timestamp_format(self, sample_report):
        assert "2025-06-15" in sample_report.timestamp
        assert "+00:00" in sample_report.timestamp

    def test_report_id_is_hex(self, sample_report):
        assert len(sample_report.report_id) == 12
        int(sample_report.report_id, 16)  # must not raise
