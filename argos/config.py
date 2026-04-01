"""
argos.config -- Centralised configuration via pydantic-settings.

All tunables for edge processor, camera pipeline, external service
endpoints, CAN bus interface, and ROS integration are declared here.
Environment variables override defaults (prefix ``ARGOS_``).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class CameraConfig(BaseSettings):
    """Industrial camera capture parameters."""

    model_config = {"env_prefix": "ARGOS_CAM_"}

    device_id: int = Field(0, description="V4L2 device index")
    width: int = 1920
    height: int = 1080
    fps: int = 30
    exposure_auto: bool = True
    white_balance_k: int = Field(5500, description="Colour temperature in Kelvin")


class EdgeProcessorConfig(BaseSettings):
    """ONNX Runtime execution provider settings."""

    model_config = {"env_prefix": "ARGOS_EDGE_"}

    detector_model_path: str = "models/defect_detector_v3.onnx"
    classifier_model_path: str = "models/severity_classifier_v2.onnx"
    execution_provider: str = Field("CUDAExecutionProvider", description="ONNX EP")
    confidence_threshold: float = 0.45
    nms_iou_threshold: float = 0.50
    input_size: tuple[int, int] = (640, 640)


class SynizConfig(BaseSettings):
    """SYNIZ TRIZ-analysis service connection."""

    model_config = {"env_prefix": "ARGOS_SYNIZ_"}

    ws_endpoint: str = "wss://syniz.fincantieri.internal/v1/ws"
    api_key: str = Field("", description="Bearer token for SYNIZ gateway")
    timeout_s: float = 30.0
    max_reconnect_attempts: int = 5


class PoseidonConfig(BaseSettings):
    """POSEIDON-DIAG CAN bus bridge settings."""

    model_config = {"env_prefix": "ARGOS_POSEIDON_"}

    can_interface: str = "can0"
    can_bustype: str = "socketcan"
    can_bitrate: int = 250_000
    j1939_source_address: int = 0xFE


class RosConfig(BaseSettings):
    """ROS 2 topic names used by ARGOS."""

    model_config = {"env_prefix": "ARGOS_ROS_"}

    image_topic: str = "/argos/camera/image_raw"
    defect_topic: str = "/argos/defects"
    odometry_topic: str = "/argos/odom"
    health_topic: str = "/argos/health"


class ArgosSettings(BaseSettings):
    """Root configuration aggregating all sub-configs."""

    model_config = {"env_prefix": "ARGOS_"}

    camera: CameraConfig = CameraConfig()
    edge: EdgeProcessorConfig = EdgeProcessorConfig()
    syniz: SynizConfig = SynizConfig()
    poseidon: PoseidonConfig = PoseidonConfig()
    ros: RosConfig = RosConfig()
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    aegis_dashboard_url: str = "https://aegis.fincantieri.internal/api/v2"
