"""
argos.report -- Inspection report generation (JSON + PDF stub).

Produces structured reports that include defect locations, severity
grades, SHAP-based explanations, sensor context, GPS coordinates,
and reference images.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, NoReturn

from argos.vision.classifier import ClassifiedDefect


@dataclass
class DefectEntry:
    """Serialisable representation of a single defect finding."""

    defect_type: str
    severity: str
    confidence: float
    severity_confidence: float
    bbox: tuple[float, float, float, float]
    shap_top_feature: str

    @classmethod
    def from_classified(cls, cd: ClassifiedDefect) -> DefectEntry:
        d = cd.detection
        return cls(
            defect_type=d.defect_type.value,
            severity=cd.severity.name.lower(),
            confidence=d.confidence,
            severity_confidence=cd.severity_confidence,
            bbox=(d.x_min, d.y_min, d.x_max, d.y_max),
            shap_top_feature=cd.shap_top_feature,
        )


@dataclass
class InspectionReport:
    """Complete inspection report with defect findings and metadata."""

    report_id: str
    timestamp: str
    gps_lat: float
    gps_lon: float
    defects: list[DefectEntry]
    unknown_analyses: list[dict[str, Any]]
    sensor_context: dict[str, Any]
    total_defects: int = field(init=False)
    max_severity: str = field(init=False)

    def __post_init__(self) -> None:
        self.total_defects = len(self.defects) + len(self.unknown_analyses)
        severities = [d.severity for d in self.defects]
        severity_order = ["critical", "severe", "moderate", "minor"]
        self.max_severity = next(
            (s for s in severity_order if s in severities), "none"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-compatible dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "gps": {"lat": self.gps_lat, "lon": self.gps_lon},
            "total_defects": self.total_defects,
            "max_severity": self.max_severity,
            "defects": [asdict(d) for d in self.defects],
            "unknown_analyses": self.unknown_analyses,
            "sensor_context": self.sensor_context,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ReportGenerator:
    """Factory that assembles ``InspectionReport`` instances.

    The ``build`` method gathers all pipeline outputs into a single
    immutable report.  A PDF rendering stub is provided for future
    integration with a LaTeX or ReportLab backend.
    """

    def build(
        self,
        timestamp: dt.datetime,
        gps_lat: float,
        gps_lon: float,
        classified_defects: list[ClassifiedDefect],
        unknown_analyses: list[dict[str, Any]],
        sensor_context: dict[str, Any],
    ) -> InspectionReport:
        """Assemble an inspection report from pipeline outputs."""
        defect_entries = [
            DefectEntry.from_classified(cd) for cd in classified_defects
        ]
        return InspectionReport(
            report_id=uuid.uuid4().hex[:12],
            timestamp=timestamp.isoformat(),
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            defects=defect_entries,
            unknown_analyses=unknown_analyses,
            sensor_context=sensor_context,
        )

    def render_pdf(self, report: InspectionReport, output_path: str) -> NoReturn:
        """Stub: render the report as a PDF document.

        A production implementation would use ReportLab or LaTeX to
        produce a print-ready inspection certificate.

        Always raises ``NotImplementedError`` until a backend is integrated.
        """
        # TODO: integrate ReportLab for PDF generation
        raise NotImplementedError(
            "PDF rendering is not yet implemented -- see roadmap issue #47"
        )
