"""
argos.poseidon_bridge -- CAN bus bridge to POSEIDON-DIAG.

Reads SAE J1939 parameter group numbers (PGNs) from the vessel's engine
and auxiliary systems via python-can.  Provides real-time sensor context
for defect-severity fusion (e.g. high exhaust temp near a crack raises
severity).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Any

import can

from argos.config import PoseidonConfig

logger = logging.getLogger(__name__)


# --- J1939 PGN constants ---------------------------------------------------
PGN_ENGINE_RPM = 0xF004          # Electronic Engine Controller 1
PGN_EXHAUST_TEMP = 0xFEF5        # Exhaust gas temperature
PGN_OIL_PRESSURE = 0xFEEF        # Engine oil pressure
PGN_COOLANT_TEMP = 0xFEEE        # Engine coolant temperature


@dataclass(frozen=True)
class SensorSnapshot:
    """Decoded snapshot of vessel machinery telemetry."""

    engine_rpm: float
    exhaust_temp_c: float
    oil_pressure_kpa: float
    coolant_temp_c: float
    raw_pgns: dict[int, bytes] = field(default_factory=dict)


class PoseidonBridge:
    """CAN bus reader that decodes J1939 PGNs from POSEIDON-DIAG.

    Parameters
    ----------
    config:
        CAN interface and J1939 addressing configuration.
    """

    def __init__(self, config: PoseidonConfig | None = None) -> None:
        self._cfg = config or PoseidonConfig()
        self._bus: can.Bus | None = None

    def open(self) -> None:
        """Initialise the CAN bus interface."""
        self._bus = can.Bus(
            interface=self._cfg.can_bustype,
            channel=self._cfg.can_interface,
            bitrate=self._cfg.can_bitrate,
        )
        logger.info("CAN bus opened on %s (%s)",
                     self._cfg.can_interface, self._cfg.can_bustype)

    def close(self) -> None:
        """Shut down the CAN bus interface."""
        if self._bus:
            self._bus.shutdown()
            logger.info("CAN bus closed")

    def _pgn_from_id(self, arbitration_id: int) -> int:
        """Extract the PGN from a 29-bit CAN arbitration ID."""
        return (arbitration_id >> 8) & 0xFFFF

    def read_snapshot(self, timeout_s: float = 2.0) -> SensorSnapshot:
        """Collect one snapshot of engine telemetry from the CAN bus.

        Reads messages until all target PGNs are captured or timeout
        expires, then decodes and returns a ``SensorSnapshot``.
        """
        if self._bus is None:
            raise RuntimeError("CAN bus not opened -- call open() first")

        collected: dict[int, bytes] = {}
        target_pgns = {PGN_ENGINE_RPM, PGN_EXHAUST_TEMP, PGN_OIL_PRESSURE, PGN_COOLANT_TEMP}

        while target_pgns - collected.keys():
            msg = self._bus.recv(timeout=timeout_s)
            if msg is None:
                break
            pgn = self._pgn_from_id(msg.arbitration_id)
            if pgn in target_pgns:
                collected[pgn] = msg.data

        return SensorSnapshot(
            engine_rpm=self._decode_rpm(collected.get(PGN_ENGINE_RPM, b"\x00" * 8)),
            exhaust_temp_c=self._decode_temp(collected.get(PGN_EXHAUST_TEMP, b"\x00" * 8)),
            oil_pressure_kpa=self._decode_pressure(collected.get(PGN_OIL_PRESSURE, b"\x00" * 8)),
            coolant_temp_c=self._decode_temp(collected.get(PGN_COOLANT_TEMP, b"\x00" * 8)),
            raw_pgns=collected,
        )

    @staticmethod
    def _decode_rpm(data: bytes) -> float:
        """Decode engine RPM from EEC1 (SPN 190): bytes 3-4, 0.125 RPM/bit."""
        raw = struct.unpack_from("<H", data, 3)[0]
        return raw * 0.125

    @staticmethod
    def _decode_temp(data: bytes) -> float:
        """Decode temperature SPN: first 2 bytes, 0.03125 deg/bit, offset -273."""
        raw = struct.unpack_from("<H", data, 0)[0]
        return raw * 0.03125 - 273.0

    @staticmethod
    def _decode_pressure(data: bytes) -> float:
        """Decode oil pressure SPN: byte 3, 4 kPa/bit."""
        return float(data[3]) * 4.0

    def to_context_dict(self, snapshot: SensorSnapshot) -> dict[str, Any]:
        """Serialise a snapshot to a plain dict for upstream consumers."""
        return {
            "engine_rpm": snapshot.engine_rpm,
            "exhaust_temp_c": snapshot.exhaust_temp_c,
            "oil_pressure_kpa": snapshot.oil_pressure_kpa,
            "coolant_temp_c": snapshot.coolant_temp_c,
        }
