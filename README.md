<p align="center">
  <img src="https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime">
</p>

<h1 align="center">ARGOS</h1>
<h3 align="center">Autonomous Ship Inspection Robot</h3>

<p align="center">
  <em>Edge AI + machine vision + TRIZ problem-solving for maritime inspection tasks<br>where humans should not go</em>
</p>

---

> ### Status: design document + Python skeleton
>
> The repository contains **ten Python modules and 23 unit tests**. There is no
> robot, no trained model, no simulation and no field data. Concretely:
>
> | Described below | In the repository |
> |---|---|
> | ROS 2 robot framework | **absent** — four unused topic-name strings in `argos/config.py` |
> | Rust CAN/NMEA stack | **absent** — the CAN bridge is Python + `python-can` |
> | Gazebo simulation, `docker-compose.sim.yml` | **absent** |
> | AEGIS-MONITOR dashboard | **absent** — no frontend code |
> | NautilusQuant 3-bit quantization | **absent** — `argos/vision/detector.py` has an INT8 identity-passthrough stub |
> | Neo4j knowledge graph queries | **absent** — connection settings only, no query code |
> | SHAP explainability | **absent** — the field is filled from `argmax` of an auxiliary head |
> | TRITON-ML RUL estimation | **absent** |
> | PDF inspection reports | **absent** — `argos/report.py` raises `NotImplementedError` |
> | Trained defect-detection models | **absent** — no ONNX weights are distributed |
>
> What does work: the detector/classifier wrappers around ONNX Runtime, the
> J1939 CAN decoder, the SYNIZ WebSocket client, the JSON report builder and
> the inspection loop that ties them together — given a model file you supply.

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

**Planned.** The intent is to run vision models quantized with
[NautilusQuant](https://github.com/hermandoronin/NautilusQuant) so inference fits
on a low-power edge processor without a GPU. Today `argos/vision/detector.py`
loads a plain ONNX model and `NautilusQuantLUT` is an identity-passthrough stub;
no quantized model and no export path exist yet. Target defect classes:

- **Defect detection**: corrosion, cracks, coating breakdown, weld defects, pitting
- **Biofouling classification**: barnacles, algae, tubeworms, slime (severity grading)
- **Structural assessment**: plate deformation, bracket failure, stiffener buckling
- **Leak detection**: oil sheen, water ingress, condensation patterns

The reason for choosing NautilusQuant over a random-rotation scheme is
determinism — a fixed rotation ROM instead of a PRNG-derived matrix — which
matters when a class surveyor has to reproduce a result. That integration is not
implemented here.

### 2. Known Defect Path (TRITON-ML)

When the vision system detects a recognized defect pattern:

1. Classify defect type and severity with a local ONNX classification head — **implemented** (`argos/vision/classifier.py`)
2. Build a JSON inspection report with detections, confidence and GPS position — **implemented** (`argos/report.py`)
3. POST the report to a dashboard endpoint over HTTP — **implemented** (`argos/inspector.py`)
4. Cross-reference the ship's maintenance history in a Neo4j graph — **planned**
5. Estimate remaining useful life via TRITON-ML — **planned**
6. Attach real SHAP attributions instead of the current argmax placeholder — **planned**
7. Render the report as a PDF — **planned**, `ReportGenerator.render_pdf` raises `NotImplementedError`

### 3. Unknown Situation Path (SYNIZ)

When the robot encounters something outside its training distribution:

1. Anomaly detector flags the observation as novel
2. Image, sensor context, and location are packaged as a SYNIZ task
3. The SYNIZ swarm debates the observation across multiple hypotheses:
   - What physical process could cause this pattern?
   - Which TRIZ contradiction does it represent?
   - What is the Ideal Final Result for this inspection scenario?
4. The SuperAgent synthesizes a recommended action:
   - Additional sensor readings to collect
   - Alternative inspection angles
   - Hypothesis for shore-side expert review
5. The recommendation is returned to the inspection loop (the knowledge-graph
   write-back is planned, not implemented)

Implemented: the WebSocket client, the request encoding and the hypothesis
parsing (`argos/syniz_client.py`). It needs a running SYNIZ instance.

### 4. Sensor Fusion (POSEIDON-DIAG)

`argos/poseidon_bridge.py` decodes four J1939 PGNs (engine RPM, exhaust
temperature, oil pressure, coolant temperature) over `python-can` — implemented
and unit-tested. NMEA 2000, vibration and sonar are planned:

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

Intended targets. **Nothing has been benchmarked on any of them** — no frame
rate below is a measurement.

| Platform | Use Case | Planned runtime |
|---|---|---|
| NVIDIA Jetson Orin Nano | Primary vision processor | ONNX Runtime, CUDA EP |
| Intel Movidius Myriad X | Low-power secondary | OpenVINO |
| Coral Edge TPU | Ultra-low power | TFLite |
| Hailo-8 | High-throughput | HailoRT |

The ONNX export path from a training pipeline does not exist yet either.

---

## Ecosystem Integration

ARGOS is not a standalone system. It is the physical embodiment of a complete maritime intelligence stack:

| Component | Role in ARGOS |
|---|---|
| [**NautilusQuant**](https://github.com/hermandoronin/NautilusQuant) | Planned model compression for edge inference (deterministic rotation, ~1.9 KB ROM) |
| [**SYNIZ**](https://github.com/hermandoronin/SYNIZ) | TRIZ-based reasoning when encountering unknown defects or novel situations |
| [**TRITON-ML**](https://github.com/hermandoronin/TRITON-ML) | Predictive maintenance models (defect classification, RUL estimation) |
| [**POSEIDON-DIAG**](https://github.com/hermandoronin/POSEIDON-DIAG) | Ship systems interface (CAN/J1939/NMEA 2000 sensor data fusion) |
| [**AEGIS-MONITOR**](https://github.com/hermandoronin/AEGIS-MONITOR) | Operator dashboard (live video feed, 3D inspection map, alarm management) |

---

## Tech Stack

| Layer | Technology | In this repo |
|---|---|---|
| Language | Python 3.11+ | yes |
| Vision | OpenCV, ONNX Runtime | yes (wrappers; no model weights) |
| CAN interface | `python-can`, J1939 PGN decoding | yes |
| SYNIZ client | `websockets` | yes |
| Config | pydantic-settings, YAML | yes |
| Deployment | Docker Compose (ARGOS + Neo4j) | yes |
| Robot framework | ROS 2 | no — planned |
| Knowledge graph queries | Neo4j | no — connection settings only |
| ML training / ONNX export | PyTorch, SHAP | no — planned |
| Operator dashboard | React, Three.js | no — separate project |

---

## Classification Society Context

Remote inspection technologies are increasingly accepted by major classification societies:

- **DNV**: Rules for Classification, Pt.7 Ch.1 — remote inspection techniques (RIT)
- **Lloyd's Register**: ShipRight procedure for approval of service suppliers for remote inspection
- **Bureau Veritas**: NI 668 — Guidelines for remote surveys and inspections
- **IACS**: Recommendation 42 — Guidelines for use of remote inspection techniques

ARGOS is *designed* to generate reports compatible with these frameworks: every
detection carries a confidence score and sensor context. It has not been
submitted to, reviewed by or accepted by any classification society, and the
explainability field is currently a placeholder, not a SHAP attribution.

---

## Quick Start

```bash
git clone https://github.com/hermandoronin/ARGOS.git
cd ARGOS

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Lint, type-check and run the unit tests -- this is what CI runs
ruff check argos/ tests/
mypy argos/ --ignore-missing-imports
pytest tests/ -v
```

Running the inspection loop needs two things this repository does not ship: an
ONNX detector/classifier model under `models/`, and a camera:

```bash
cp config.example.yaml config.yaml   # then edit paths and endpoints
python -m argos --config config.yaml --interval 5
```

`docker-compose.yml` brings up ARGOS and Neo4j. There is no simulation compose
file and no Gazebo environment — both are on the roadmap.

---

## Roadmap

- [x] System architecture and ecosystem integration design
- [x] Inspection loop skeleton: detector, classifier, report builder, CLI
- [x] POSEIDON-DIAG CAN bridge — J1939 PGN decoding over `python-can`
- [x] SYNIZ WebSocket client for unknown-defect escalation
- [ ] NautilusQuant edge inference pipeline — currently an identity-passthrough stub
- [ ] Defect detection model training (corrosion, cracks, fouling)
- [ ] PDF report export — `ReportGenerator.render_pdf` raises `NotImplementedError`
- [ ] Real SHAP attributions instead of the argmax placeholder
- [ ] Neo4j knowledge-graph read/write
- [ ] ROS 2 robot framework with simulated hull inspection
- [ ] Gazebo simulation environment (ship hull, ballast tank)
- [ ] AEGIS-MONITOR live inspection view
- [ ] Field trials with magnetic crawler prototype

---

## Author

Marine engineer with 3+ years of hands-on ship power plant maintenance. I have crawled through ballast tanks and inspected hull plates in dry dock. ARGOS automates the inspection tasks I used to do manually --- in places where no one should have to go.

---

## License

MIT — see [`LICENSE`](LICENSE).
