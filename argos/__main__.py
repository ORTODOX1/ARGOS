"""
argos.__main__ -- CLI entry point for the ARGOS inspection system.

Usage:
    python -m argos --config config.yaml

Only live camera capture is implemented; there is no simulation or replay
mode.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from argos.config import ArgosSettings
from argos.inspector import InspectionEngine

logger = logging.getLogger("argos")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="argos",
        description="ARGOS -- Autonomous Robot for General Onboard Surveillance",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"),
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between inspection cycles (default: 5.0)",
    )
    parser.add_argument(
        "--lat", type=float, default=45.4315,
        help="GPS latitude of the inspection robot (default: 45.4315)",
    )
    parser.add_argument(
        "--lon", type=float, default=12.3456,
        help="GPS longitude of the inspection robot (default: 12.3456)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> ArgosSettings:
    """Load settings from a YAML file, falling back to env/defaults."""
    if path.exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Loaded config from %s", path)
        return ArgosSettings(**raw)
    logger.warning("Config file %s not found, using defaults", path)
    return ArgosSettings()


async def run(args: argparse.Namespace) -> None:
    """Initialise and run the inspection engine."""
    settings = load_config(args.config)
    engine = InspectionEngine(settings=settings)

    logger.info("Starting live capture (interval=%.1fs)", args.interval)

    await engine.run_continuous(
        gps_lat=args.lat, gps_lon=args.lon,
        interval_s=args.interval,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m argos``."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
