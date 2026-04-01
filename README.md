<p align="center">
  <img src="https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ROS_2-22314E?style=for-the-badge&logo=ros&logoColor=white" alt="ROS 2">
  <img src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
</p>

<h1 align="center">ARGOS</h1>
<h3 align="center">Autonomous Ship Inspection Robot</h3>

<p align="center">
  <em>Edge AI + machine vision + TRIZ problem-solving for maritime inspection tasks<br>where humans should not go</em>
</p>

---

## The Problem

Ship inspection is dangerous, expensive, and incomplete. Ballast tanks, void spaces, cargo holds, and underwater hull surfaces require human inspectors to work in confined spaces, at height, or underwater. Inspections are time-limited by diver bottom time, tank ventilation, and dry-dock schedules. Critical defects in hard-to-reach areas are routinely missed.

Classification societies (DNV, Lloyd's, Bureau Veritas) are increasingly accepting remote inspection technologies, but existing ROV and drone solutions lack the intelligence to identify novel defect patterns or make autonomous decisions when encountering unknown conditions.

ARGOS addresses this by combining edge-deployed machine vision with an inventive problem-solving backend. When the robot sees something it has been trained on, it classifies and reports. When it encounters something genuinely new, it escalates to a multi-agent TRIZ reasoning system that generates hypotheses and inspection strategies in real time.

---

## System Architecture

```
                    ┌─────────────────────────────────────┐
                    │          AEGIS-MONITOR               │
                    │     Operator Dashboard (React)       │
                    │  Live video + 3D map + alarm panel   │
                    └──────────────┬──────────────────────┘
                                   │ WebSocket
                    ┌──────────────v──────────────────────┐
                    │         Shore / Ship Server          │
                    │                                      │
                    │  ┌──────────┐    ┌───────────────┐  │
                    │  │ SYNIZ    │    │  TRITON-ML     │  │
                    │  │ TRIZ     │    │  Predictive    │  │
                    │  │ Swarm    │    │  Maintenance   │  │
                    │  │ Engine   │    │  Models        │  │
                    │  └────┬─────┘    └───────┬───────┘  │
                    │       │                  │          │
                    │  ┌────v──────────────────v───────┐  │
                    │  │     Knowledge Graph (Neo4j)    │  │
                    │  │  Defect DB + Inspection History │  │
                    │  └───────────────────────────────┘  │
                    └──────────────┬──────────────────────┘
                                   │ 4G/5G / Ship LAN / Acoustic
                    ┌──────────────v──────────────────────┐
                    │         ARGOS Robot Core              │
                    │                                      │
                    │  ┌──────────────────────────────┐   │
                    │  │   Edge Processor              │   │
                    │  │   NautilusQuant-compressed    │   │
                    │  │   vision models (3-bit ONNX)  │   │
                    │  └──────────┬───────────────────┘   │
                    │             │                        │
                    │  ┌──────────v───────────────────┐   │
                    │  │   Machine Vision Pipeline     │   │
                    │  │                               │   │
                    │  │  Camera → Detect → Classify   │   │
                    │  │     │                         │   │
                    │  │     ├─ Known defect → Report  │   │
                    │  │     └─ Unknown → SYNIZ query  │   │
                    │  └──────────────────────────────┘   │
                    │                                      │
                    │  ┌──────────────────────────────┐   │
                    │  │   POSEIDON-DIAG Interface     │   │
                    │  │   CAN/NMEA → sensor fusion    │   │
                    │  │   Engine data + robot telemetry│   │
                    │  └──────────────────────────────┘   │
                    │                                      │
                    │  Navigation │ Sensors │ Actuators    │
                    └──────────────────────────────────────┘
```

---

## How It Works

### 1. Edge Vision (NautilusQuant)

The robot runs machine vision models compressed to 3-bit precision using NautilusQuant's deterministic golden ratio quantization. This enables real-time inference on low-power edge processors without GPU:

- **Defect detection**: corrosion, cracks, coating breakdown, weld defects, pitting
- **Biofouling classification**: barnacles, algae, tubeworms, slime (severity grading)
- **Structural assessment**: plate deformation, bracket failure, stiffener buckling
- **Leak detection**: oil sheen, water ingress, condensation patterns

The 512-byte lookup table and zero-overhead quantization make NautilusQuant ideal for deterministic inference on safety-critical embedded hardware where reproducibility is mandatory.

### 2. Known Defect Path (TRITON-ML)

When the vision system detects a recognized defect pattern:

1. Classify defect type and severity using TRITON-ML models
2. Cross-reference with ship's maintenance history (Neo4j knowledge graph)
3. Estimate remaining useful life of the affected component
4. Generate inspection report with SHAP explainability
5. Push alert to AEGIS-MONITOR operator dashboard

### 3. Unknown Situation Path (SYNIZ)

When the robot encounters something outside its training distribution:

1. Anomaly detector flags the observation as novel
2. Image, sensor context, and location are packaged as a SYNIZ task
3. 50 TRIZ agents debate the observation across multiple hypotheses:
   - What physical process could cause this pattern?
   - Which TRIZ contradiction does it represent?
   - What is the Ideal Final Result for this inspection scenario?
4. The SuperAgent synthesizes a recommended action:
   - Additional sensor readings to collect
   - Alternative inspection angles
   - Hypothesis for shore-side expert review
5. Robot executes the recommendation and logs results back to the knowledge graph

### 4. Sensor Fusion (POSEIDON-DIAG)

The robot integrates with the ship's existing instrumentation via CAN bus and NMEA 2000:

- Engine parameters (RPM, temperatures, pressures) provide operational context
- Vibration data from ship's accelerometers correlates with visual findings
- Navigation data (GPS, heading, speed) enables defect geolocation
- The robot's own sensors (IMU, depth, sonar) feed into the unified data stream

---

## Inspection Modes

| Mode | Environment | Platform | Key Challenges |
|---|---|---|---|
| **Hull Survey** | Underwater | ROV / magnetic crawler | Visibility, currents, biofouling removal |
| **Ballast Tank** | Confined space | Tracked crawler / drone | Humidity, coating condition, limited access |
| **Cargo Hold** | Large open space | Aerial drone | Scale, lighting, structural complexity |
| **Engine Room** | Indoor, hot | Tracked crawler | Temperature, vibration, oil/grease |
| **Void Spaces** | Confined, dark | Mini crawler | Access, communication, orientation |

---

## Edge Hardware Targets

| Platform | Use Case | Inference |
|---|---|---|
| NVIDIA Jetson Orin Nano | Primary vision processor | NautilusQuant 3-bit ONNX, 30+ FPS |
| Intel Movidius Myriad X | Low-power secondary | OpenVINO INT8, 15 FPS |
| Coral Edge TPU | Ultra-low power | TFLite quantized, 10 FPS |
| Hailo-8 | High-throughput | 26 TOPS at 2.5W |

All models are exported via ONNX from TRITON-ML training pipeline, then compressed with NautilusQuant for deployment.

---

## Ecosystem Integration

ARGOS is not a standalone system. It is the physical embodiment of a complete maritime intelligence stack:

| Component | Role in ARGOS |
|---|---|
| [**NautilusQuant**](https://github.com/ORTODOX1/NautilusQuant) | Model compression for edge inference (3-bit, deterministic, 512-byte LUT) |
| [**SYNIZ**](https://github.com/ORTODOX1/SYNIZ) | TRIZ-based reasoning when encountering unknown defects or novel situations |
| [**TRITON-ML**](https://github.com/ORTODOX1/TRITON-ML) | Predictive maintenance models (defect classification, RUL estimation) |
| [**POSEIDON-DIAG**](https://github.com/ORTODOX1/POSEIDON-DIAG) | Ship systems interface (CAN/J1939/NMEA 2000 sensor data fusion) |
| [**AEGIS-MONITOR**](https://github.com/ORTODOX1/AEGIS-MONITOR) | Operator dashboard (live video feed, 3D inspection map, alarm management) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Robot Framework | ROS 2 Humble (navigation, sensor drivers, SLAM) |
| Vision | OpenCV, ONNX Runtime, NautilusQuant compression |
| Edge Processor | NVIDIA Jetson Orin Nano / Hailo-8 |
| Communication | CAN bus (J1939/NMEA 2000), 4G/5G, acoustic modem (underwater) |
| Backend | Python 3.11+, FastAPI, Neo4j |
| Problem Solving | SYNIZ (50 TRIZ agents, Graphiti memory) |
| ML Pipeline | PyTorch, XGBoost, ONNX export, SHAP |
| Dashboard | React, TypeScript, Three.js, WebSocket |
| Systems Interface | Rust (POSEIDON-DIAG CAN/NMEA stack) |
| Deployment | Docker Compose, ROS 2 launch files |

---

## Classification Society Context

Remote inspection technologies are increasingly accepted by major classification societies:

- **DNV**: Rules for Classification, Pt.7 Ch.1 — remote inspection techniques (RIT)
- **Lloyd's Register**: ShipRight procedure for approval of service suppliers for remote inspection
- **Bureau Veritas**: NI 668 — Guidelines for remote surveys and inspections
- **IACS**: Recommendation 42 — Guidelines for use of remote inspection techniques

ARGOS is designed to generate inspection reports compatible with these frameworks. All detections include confidence scores, SHAP explainability, and full sensor context for surveyor review.

---

## Quick Start

```bash
# Clone
git clone https://github.com/ORTODOX1/ARGOS.git
cd ARGOS

# Simulation mode (no robot hardware required)
docker-compose -f docker-compose.sim.yml up -d

# This starts:
#   - Gazebo simulation with ship hull model
#   - ARGOS vision pipeline with sample defect images
#   - SYNIZ backend for unknown defect reasoning
#   - AEGIS-MONITOR dashboard at http://localhost:5173
```

---

## Roadmap

- [x] System architecture and ecosystem integration design
- [x] NautilusQuant edge inference pipeline (ONNX 3-bit export)
- [ ] ROS 2 robot framework with simulated hull inspection
- [ ] Defect detection model training (corrosion, cracks, fouling)
- [ ] SYNIZ integration for unknown defect reasoning
- [ ] Gazebo simulation environment (ship hull, ballast tank)
- [ ] POSEIDON-DIAG CAN bridge for sensor fusion
- [ ] AEGIS-MONITOR live inspection view
- [ ] Field trials with magnetic crawler prototype

---

## Author

Marine engineer with 3+ years of hands-on ship power plant maintenance. I have crawled through ballast tanks and inspected hull plates in dry dock. ARGOS automates the inspection tasks I used to do manually --- in places where no one should have to go.

---

## License

MIT
