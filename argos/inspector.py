"""
argos.inspector -- Main inspection orchestration engine.

Coordinates the full pipeline: frame capture, defect detection,
known/unknown routing, severity classification or SYNIZ escalation,
Neo4j persistence, and AEGIS dashboard push.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import httpx
import numpy as np

from argos.config import ArgosSettings
from argos.poseidon_bridge import PoseidonBridge
from argos.report import InspectionReport, ReportGenerator
from argos.syniz_client import SynizClient
from argos.vision.classifier import DefectClassifier, Severity
from argos.vision.detector import DefectDetector, Detection

logger = logging.getLogger(__name__)

_UNKNOWN_CONFIDENCE_CEILING = 0.30


@dataclass
class InspectionEngine:
    """Top-level orchestrator for autonomous ship inspection.

    Owns the camera capture loop and delegates to vision, SYNIZ, and
    POSEIDON sub-systems.  Results are persisted to Neo4j and forwarded
    to the AEGIS monitoring dashboard.
    """

    settings: ArgosSettings = field(default_factory=ArgosSettings)
    _detector: DefectDetector = field(init=False, repr=False)
    _classifier: DefectClassifier = field(init=False, repr=False)
    _syniz: SynizClient = field(init=False, repr=False)
    _poseidon: PoseidonBridge = field(init=False, repr=False)
    _report_gen: ReportGenerator = field(init=False, repr=False)
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._detector = DefectDetector(self.settings.edge)
        self._classifier = DefectClassifier(self.settings.edge)
        self._syniz = SynizClient(self.settings.syniz)
        self._poseidon = PoseidonBridge(self.settings.poseidon)
        self._report_gen = ReportGenerator()
        self._http = httpx.AsyncClient(
            base_url=self.settings.aegis_dashboard_url,
            timeout=10.0,
        )

    async def startup(self) -> None:
        """Initialise hardware and network connections."""
        self._poseidon.open()
        await self._syniz.connect()
        logger.info("InspectionEngine started")

    async def shutdown(self) -> None:
        """Release all resources."""
        self._poseidon.close()
        await self._syniz.close()
        await self._http.aclose()
        logger.info("InspectionEngine stopped")

    def _capture_frame(self) -> np.ndarray:
        """Grab a single frame from the inspection camera."""
        cap = cv2.VideoCapture(self.settings.camera.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera.height)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Camera capture failed")
        return frame

    def _is_unknown(self, det: Detection) -> bool:
        """Heuristic: if detector confidence is below ceiling, treat as unknown."""
        return det.confidence < _UNKNOWN_CONFIDENCE_CEILING

    async def inspect_once(self, gps_lat: float, gps_lon: float) -> InspectionReport:
        """Execute one full inspection cycle.

        Parameters
        ----------
        gps_lat, gps_lon:
            Current GPS coordinates of the inspection robot.

        Returns
        -------
        InspectionReport
            Completed report with all defects, severities, and any
            SYNIZ hypotheses for unknown anomalies.
        """
        frame = self._capture_frame()
        timestamp = dt.datetime.now(dt.timezone.utc)
        sensor_ctx = self._poseidon.to_context_dict(
            self._poseidon.read_snapshot()
        )

        detections = self._detector.detect(frame)
        known = [d for d in detections if not self._is_unknown(d)]
        unknown = [d for d in detections if self._is_unknown(d)]

        classified = self._classifier.classify(frame, known)

        syniz_results: list[dict[str, Any]] = []
        for det in unknown:
            x1, y1 = int(det.x_min), int(det.y_min)
            x2, y2 = int(det.x_max), int(det.y_max)
            crop = frame[y1:y2, x1:x2]
            hypotheses = await self._syniz.analyse_unknown_defect(
                crop, {**sensor_ctx, "gps": [gps_lat, gps_lon]}
            )
            syniz_results.append({
                "detection": det,
                "hypotheses": hypotheses,
            })

        report = self._report_gen.build(
            timestamp=timestamp,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            classified_defects=classified,
            unknown_analyses=syniz_results,
            sensor_context=sensor_ctx,
        )

        await self._push_to_aegis(report)
        return report

    async def _push_to_aegis(self, report: InspectionReport) -> None:
        """Forward the inspection report to the AEGIS dashboard API."""
        try:
            resp = await self._http.post(
                "/inspections",
                json=report.to_dict(),
            )
            resp.raise_for_status()
            logger.info("Report %s pushed to AEGIS", report.report_id)
        except httpx.HTTPError as exc:
            logger.error("AEGIS push failed: %s", exc)

    async def run_continuous(
        self, gps_lat: float, gps_lon: float, interval_s: float = 5.0
    ) -> None:
        """Run inspection in a continuous loop."""
        await self.startup()
        try:
            while True:
                try:
                    report = await self.inspect_once(gps_lat, gps_lon)
                    logger.info("Inspection %s: %d defects found",
                                report.report_id, report.total_defects)
                except RuntimeError as exc:
                    logger.error("Inspection cycle error: %s", exc)
                await asyncio.sleep(interval_s)
        finally:
            await self.shutdown()
