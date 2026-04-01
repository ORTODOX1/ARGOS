"""
argos.syniz_client -- Async WebSocket client for the SYNIZ TRIZ engine.

When the ARGOS detector encounters an anomaly that does not match any
known defect class, the image and inspection context are forwarded to
SYNIZ for contradiction-based root-cause hypothesising.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import websockets
from websockets.legacy.client import WebSocketClientProtocol

from argos.config import SynizConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrizHypothesis:
    """A single TRIZ-generated root-cause hypothesis returned by SYNIZ."""

    principle_id: int
    principle_name: str
    contradiction: str
    suggested_action: str
    confidence: float


@dataclass
class SynizClient:
    """Persistent WebSocket client that communicates with SYNIZ.

    Supports automatic reconnection with exponential back-off up to
    ``max_reconnect_attempts`` as configured in ``SynizConfig``.
    """

    config: SynizConfig = field(default_factory=SynizConfig)
    _ws: WebSocketClientProtocol | None = field(default=None, init=False, repr=False)
    _attempt: int = field(default=0, init=False, repr=False)

    async def connect(self) -> None:
        """Establish or re-establish the WebSocket connection."""
        extra_headers = {"Authorization": f"Bearer {self.config.api_key}"}
        self._ws = await websockets.connect(
            self.config.ws_endpoint,
            extra_headers=extra_headers,
            open_timeout=self.config.timeout_s,
        )
        self._attempt = 0
        logger.info("Connected to SYNIZ at %s", self.config.ws_endpoint)

    async def _ensure_connected(self) -> None:
        if self._ws is None or self._ws.closed:
            while self._attempt < self.config.max_reconnect_attempts:
                try:
                    await self.connect()
                    return
                except (OSError, websockets.WebSocketException) as exc:
                    self._attempt += 1
                    wait = min(2 ** self._attempt, 30)
                    logger.warning("SYNIZ reconnect %d/%d in %ds: %s",
                                   self._attempt, self.config.max_reconnect_attempts, wait, exc)
                    await asyncio.sleep(wait)
            raise ConnectionError("Exhausted SYNIZ reconnection attempts")

    async def analyse_unknown_defect(
        self,
        image: np.ndarray,
        context: dict[str, Any],
    ) -> list[TrizHypothesis]:
        """Send an unrecognised defect image to SYNIZ for TRIZ analysis.

        Parameters
        ----------
        image:
            BGR crop of the anomalous region.
        context:
            Metadata dict (GPS, sensor readings, timestamp, etc.).

        Returns
        -------
        list[TrizHypothesis]
            Ranked hypotheses from the SYNIZ engine.
        """
        await self._ensure_connected()
        assert self._ws is not None

        _, buf = __import__("cv2").imencode(".png", image)
        payload = json.dumps({
            "type": "analyse",
            "image_b64": base64.b64encode(buf.tobytes()).decode(),
            "context": context,
        })
        await self._ws.send(payload)
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.config.timeout_s)
        response = json.loads(raw)

        return [
            TrizHypothesis(
                principle_id=h["principle_id"],
                principle_name=h["principle_name"],
                contradiction=h["contradiction"],
                suggested_action=h["suggested_action"],
                confidence=h["confidence"],
            )
            for h in response.get("hypotheses", [])
        ]

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
            logger.info("SYNIZ connection closed")
