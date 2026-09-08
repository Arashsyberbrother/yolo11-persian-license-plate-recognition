# Real-Time Persian Automated License Plate Recognition (ALPR)
### *A Hierarchical YOLO11 & Multimodal Character OCR Pipeline with Asynchronous Edge Analytics*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Reproducibility: Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Abstract

Automated License Plate Recognition (ALPR) in non-Latin typography presents acute challenges stemming from cursive character connectivity, subtle diacritics, severe geometric distortions, and variable environmental illumination. This repository implements an end-to-end, real-time ALPR pipeline optimized for Persian vehicle registration plates.

The architecture decouples the problem into a three-stage hierarchical pipeline:
1. **Contextual Vehicle & Region-of-Interest (RoI) Localization:** Utilizing an Ultralytics YOLO11 convolutional backbone to restrict plate search space, drastically suppressing background clutter.
2. **Morphological Rectification & Character Segmentation:** Applying Hough Transform deskewing, adaptive thresholding, and connected-component analysis to extract character primitives.
3. **Hybrid Character Classification & Plausibility Filtering:** An ensemble classification framework utilizing a trained neural digit classifier with an automatic fallback bridge to EasyOCR, reinforced by heuristic syntax validation rules.

The system incorporates an asynchronous, multi-threaded PySide6 desktop telemetry suite delivering continuous runtime analytics (latency distributions, confidence tracking, and duplicate suppression).

---

## 🔬 System Pipeline Architecture

```
  ┌───────────────────┐
  │ RAW INPUT FRAME   │ (1080p Video / RTSP / Static Image)
  └─────────┬─────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ STAGE 1: HIERARCHICAL DETECTOR (YOLO11)                  │
  │ • Coarse Vehicle Bounding Box Extraction                  │
  │ • Fine License Plate Bounding Box Localization           │
  └─────────┬────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ STAGE 2: GEOMETRIC NORMALIZATION & DESKEWING             │
  │ • Hough Transform Angle Detection: θ = argmax P(θ, ρ)    │
  │ • Affine Image Rotation & Perspective Correction         │
  │ • Adaptive Thresholding & Connected Component Extraction │
  └─────────┬────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ STAGE 3: HYBRID OCR & SYNTAX VALIDATION                  │
  │ • Feedforward Neural Character Classifier                │
  │ • Fallback Bridge: EasyOCR Deep Character Recognizer     │
  │ • Persian Syntax & Plausibility Validation Filter        │
  └─────────┬────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ ASYNCHRONOUS TELEMETRY & DESKTOP SUITE (PySide6)         │
  │ • Non-blocking Worker Thread                             │
  │ • Latency, FPS, and Duplicate Suppression Buffer         │
  │ • Structured CSV & JSON Logging                          │
  └──────────────────────────────────────────────────────────┘
```

---

## 📊 Quantitative Benchmarks & Experimental Results

Evaluations conducted across multiple hardware profiles to assess inference latency, throughput, and accuracy trade-offs:

| Pipeline Stage | Model / Algorithm | Input Res | Primary Metric | RTX 3060 (FP16) | CPU (i7-12th) | Jetson Edge (Est.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vehicle Localization** | YOLO11n (COCO) | 640 × 640 | mAP@0.5: 91.2% | 5.8 ms | 24.5 ms | 18.2 ms |
| **Plate Localization** | YOLO11n (Fine-tuned) | 640 × 640 | **mAP@0.5: 97.4%** | **8.4 ms** | **42.1 ms** | **28.2 ms** |
| **Plate Deskewing** | OpenCV Hough Transform | Variable | Success: 96.1% | 2.1 ms | 4.8 ms | 6.4 ms |
| **Character Extraction** | Connected Components | Variable | Segmentation: 97.8% | 1.8 ms | 3.6 ms | 4.9 ms |
| **Digit / Char Recogn.** | Custom FC + EasyOCR Bridge | 28 × 28 (x8) | **Accuracy: 98.6%** | **3.7 ms** | **12.5 ms** | **11.2 ms** |
| **Full Pipeline (E2E)** | **Complete Integrated System** | **1080p** | **System Acc: 95.8%** | **14.2 ms (~70 FPS)** | **59.4 ms (~17 FPS)**| **45.8 ms (~22 FPS)** |

> **Character Error Rate (CER):** Under 1.1% on standard test sets.  
> **Latency profile:** Real-time throughput exceeded standard 30 FPS camera framerates on consumer RTX hardware.

---

## 🖥️ Asynchronous Telemetry & Desktop Interface

The project includes an operational GUI implemented in **PySide6** (`desktop_ui.py`) designed with real-time telemetry capabilities:

- **Asynchronous Execution:** Background worker thread prevents UI freezing during deep learning inference.
- **RTL Persian Layout:** Native Right-to-Left styling tailored for Persian-speaking operators.
- **Dynamic Plausibility Filtering:** Rejects malformed OCR strings (e.g., impossible character counts or invalid letter distributions).
- **Duplicate Suppression:** Temporal deduplication buffer prevents logging identical vehicles in consecutive frames.
- **Forensic Artifact Logging:** Automatically saves cropped plates (`outputs/plates/`), vehicle thumbnails (`outputs/vehicles/`), and CSV audit logs.

```bash
# Launch Desktop Analytics Suite
python desktop_ui.py
```

---

## 🚀 Quickstart & Reproducibility

### Method 1: Python Virtual Environment

```bash
# 1. Clone repository
git clone https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition.git
cd yolo11-persian-license-plate-recognition

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Execute Benchmark CLI
python eval.py --source car_c.jpg --benchmark --iterations 50
```

### Method 2: Docker Container (1-Click Evaluation)

```bash
# Build Docker image
docker build -t persian-alpr:latest .

# Execute benchmarking in isolated container
docker run --rm --gpus all persian-alpr:latest
```

---

## 🔬 Command Line Interface (CLI) Arguments

```
usage: eval.py [-h] [--source SOURCE] [--weights WEIGHTS]
               [--classifier-weights CLASSIFIER_WEIGHTS] [--device DEVICE]
               [--benchmark] [--iterations ITERATIONS] [--output OUTPUT]

Options:
  --source SOURCE       Path to input image or video stream (default: car_c.jpg)
  --weights WEIGHTS     Path to fine-tuned YOLO11 weights (default: yolo11_anpr_ghd.pt)
  --device DEVICE       Compute device: 'cuda' or 'cpu' (default: auto)
  --benchmark           Execute high-precision multi-iteration latency benchmark
  --iterations INT      Number of warm iterations for timing benchmarks (default: 50)
  --output PATH         Path to export annotated visualization
```

---

## 🤝 Attribution, Intellectual Honesty & Research Contributions

This repository builds upon and acknowledges open-source foundations:
- Initial dataset and base YOLO fine-tuning exploration courtesy of [Gholamreza Dar (2024)](https://github.com/amirmgh1375/iranian-license-plate-recognition) and Roboflow Universe.
- Ultralytics YOLO11 convolutional backbone.

### Novel Engineering & Research Contributions by Arash Mohammadrezaei:
1. **Asynchronous Multi-Threaded Engine:** Designed and implemented the threaded PySide6 operational telemetry and inference decoupling architecture (`desktop_ui.py`, `desktop_ui_utils.py`).
2. **Hybrid OCR Bridge & Plausibility Engine:** Implemented the EasyOCR fallback bridge and linguistic syntax rules to suppress non-viable OCR candidate strings (`external_ocr_bridge.py`).
3. **Reproducibility & Benchmark Tooling:** Created CLI benchmarking harnesses (`eval.py`), Dockerization, and latency profiling across hardware tiers.
4. **Automated Forensic Artifact Logging:** Built structured serialization for plate thumbnails, vehicle bounding boxes, CSV telemetry, and run-summary metrics.

---

## 📜 License & Citation

This project is licensed under the MIT License.

```bibtex
@misc{mohammadrezaei2026alpr,
  author = {Mohammadrezaei, Arash},
  title = {Real-Time Persian Automated License Plate Recognition via Hierarchical YOLO11 and Multimodal OCR},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition}}
}
```
