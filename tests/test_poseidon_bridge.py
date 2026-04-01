"""Tests for argos.poseidon_bridge module."""

import struct
from unittest.mock import MagicMock, patch

import pytest

from argos.poseidon_bridge import (
    PGN_COOLANT_TEMP,
    PGN_ENGINE_RPM,
    PGN_EXHAUST_TEMP,
    PGN_OIL_PRESSURE,
    PoseidonBridge,
    SensorSnapshot,
)


class TestSensorSnapshot:
    """Verify SensorSnapshot creation and immutability."""

    def test_creation(self):
        snap = SensorSnapshot(
            engine_rpm=1200.0,
            exhaust_temp_c=350.0,
            oil_pressure_kpa=400.0,
            coolant_temp_c=85.0,
        )
        assert snap.engine_rpm == pytest.approx(1200.0)
        assert snap.coolant_temp_c == pytest.approx(85.0)

    def test_frozen(self):
        snap = SensorSnapshot(0, 0, 0, 0)
        with pytest.raises(AttributeError):
            snap.engine_rpm = 999.0  # type: ignore[misc]


class TestJ1939Decoding:
    """Verify PGN decoding for engine telemetry values."""

    @pytest.fixture()
    def bridge(self):
        return PoseidonBridge()

    def test_decode_rpm(self, bridge):
        """PGN 61444 (EEC1): RPM at bytes 3-4, resolution 0.125 RPM/bit."""
        rpm_value = 1600.0
        raw = int(rpm_value / 0.125)
        data = bytearray(8)
        struct.pack_into("<H", data, 3, raw)
        result = bridge._decode_rpm(bytes(data))
        assert result == pytest.approx(rpm_value)

    def test_decode_temperature(self, bridge):
        """Temperature SPNs: bytes 0-1, 0.03125 deg/bit, offset -273."""
        target_temp = 85.0
        raw = int((target_temp + 273.0) / 0.03125)
        data = bytearray(8)
        struct.pack_into("<H", data, 0, raw)
        result = bridge._decode_temp(bytes(data))
        assert result == pytest.approx(target_temp, abs=0.04)

    def test_decode_oil_pressure(self, bridge):
        """Oil pressure SPN: byte 3, 4 kPa/bit."""
        data = bytearray(8)
        data[3] = 100
        result = bridge._decode_pressure(bytes(data))
        assert result == pytest.approx(400.0)

    def test_pgn_extraction(self, bridge):
        """Extract PGN from a 29-bit CAN arbitration ID."""
        arb_id = (0x18F004FE)  # Priority=6, PGN=0xF004, SA=0xFE
        pgn = bridge._pgn_from_id(arb_id)
        assert pgn == PGN_ENGINE_RPM


class TestReadSnapshot:
    """Verify full snapshot reading with mocked CAN bus."""

    @pytest.fixture()
    def bridge_with_mock_bus(self):
        bridge = PoseidonBridge()
        mock_bus = MagicMock()

        rpm_data = bytearray(8)
        struct.pack_into("<H", rpm_data, 3, int(1200 / 0.125))

        temp_data = bytearray(8)
        struct.pack_into("<H", temp_data, 0, int((350 + 273) / 0.03125))

        pressure_data = bytearray(8)
        pressure_data[3] = 100

        coolant_data = bytearray(8)
        struct.pack_into("<H", coolant_data, 0, int((85 + 273) / 0.03125))

        messages = []
        for pgn, data in [
            (PGN_ENGINE_RPM, rpm_data),
            (PGN_EXHAUST_TEMP, temp_data),
            (PGN_OIL_PRESSURE, pressure_data),
            (PGN_COOLANT_TEMP, coolant_data),
        ]:
            msg = MagicMock()
            msg.arbitration_id = (6 << 26) | (pgn << 8) | 0xFE
            msg.data = bytes(data)
            messages.append(msg)

        mock_bus.recv = MagicMock(side_effect=messages)
        bridge._bus = mock_bus
        return bridge

    def test_snapshot_values(self, bridge_with_mock_bus):
        snap = bridge_with_mock_bus.read_snapshot()
        assert snap.engine_rpm == pytest.approx(1200.0)
        assert snap.exhaust_temp_c == pytest.approx(350.0, abs=0.04)
        assert snap.oil_pressure_kpa == pytest.approx(400.0)
        assert snap.coolant_temp_c == pytest.approx(85.0, abs=0.04)
