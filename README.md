# Persian License Plate Recognition (YOLO + OpenCV + PyTorch)

[![CI](https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Arashsyberbrother/yolo11-persian-license-plate-recognition/branch/main/graph/badge.svg)](https://codecov.io/gh/Arashsyberbrother/yolo11-persian-license-plate-recognition)
[![Python 3.10](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/main/.github/workflows/ci.yml&query=%24.jobs%5B%27lint-and-test%27%5D.strategy.matrix%5B%27python-version%27%5D%5B0%5D&label=python)](https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/main/.github/workflows/ci.yml&query=%24.jobs%5B%27lint-and-test%27%5D.strategy.matrix%5B%27python-version%27%5D%5B1%5D&label=python)](https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/dynamic/yaml?url=https://raw.githubusercontent.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/main/.github/workflows/ci.yml&query=%24.jobs%5B%27lint-and-test%27%5D.strategy.matrix%5B%27python-version%27%5D%5B2%5D&label=python)](https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Arashsyberbrother/yolo11-persian-license-plate-recognition)](LICENSE)

## Problem statement

This project detects Persian vehicle license plates and reads plate characters from image/video inputs using a YOLO detector, OpenCV-based plate extraction, and a PyTorch classifier with optional EasyOCR fallback.  
The goal is to provide a reproducible engineering pipeline without overstating performance claims.

## End-to-end architecture / pipeline

1. **Vehicle/plate detection** (`eval.py`, YOLO weights)
2. **Plate extraction and normalization** (`license_plate_extractor.py`, OpenCV)
3. **Character classification** (`image_classifier.py`, `persian_digit_classifier.pt`)
4. **Optional OCR fallback + plate plausibility filters** (`external_ocr_bridge.py`, `desktop_ui_utils.py`)
5. **Desktop operational UI** (`desktop_ui.py`)

## Repository layout

```text
yolo11-persian-license-plate-recognition/
├── .github/workflows/ci.yml
├── tests/
│   ├── test_desktop_ui_utils.py               # lightweight unit tests
│   └── integration/test_eval_integration.py   # optional model-dependent tests
├── eval.py
├── desktop_ui.py
├── desktop_ui_utils.py
├── image_classifier.py
├── license_plate_extractor.py
├── external_ocr_bridge.py
├── requirements.txt
├── Dockerfile
├── *.ipynb                                    # experiments/notebooks
└── results/                                   # sample generated outputs
```

## Model and data requirements

- Default detector weights: `yolo11_anpr_ghd.pt`
- Character classifier weights: `persian_digit_classifier.pt`
- Base YOLO model (general): `yolo11n.pt`
- Example sample images: `car_a.jpg` ... `car_d.jpg`

> Existing tracked assets are kept as-is. New large model/data artifacts should **not** be committed; store them externally and document retrieval steps in PRs.

## Benchmark table (only measured values tracked in this repo)

| Item | Measured value | Evidence |
|---|---:|---|
| Lightweight unit tests | 14 passed | `python -m unittest -v tests/test_desktop_ui_utils.py` |
| CI coverage artifact | `coverage.xml` generated per Python matrix job | `.github/workflows/ci.yml` |

The repository does **not** claim fixed production accuracy/FPS/latency numbers in source control. Use local hardware benchmarking via `eval.py --benchmark` for your environment.

## Local quickstart (exact commands)

```bash
git clone https://github.com/Arashsyberbrother/yolo11-persian-license-plate-recognition.git
cd yolo11-persian-license-plate-recognition
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Single image inference path
python eval.py --source car_c.jpg --weights yolo11_anpr_ghd.pt --classifier-weights persian_digit_classifier.pt
```

## Docker quickstart (exact commands)

```bash
docker build -t persian-alpr:latest .
docker run --rm persian-alpr:latest
```

If running GPU inference in Docker, configure your runtime with NVIDIA Container Toolkit and pass `--gpus all`.

## Testing

### Lightweight (CI default)

```bash
pip install pytest pytest-cov ruff flake8
ruff check --select E9,F desktop_ui_utils.py external_ocr_bridge.py tests
flake8 --select=E9,F63,F7,F82 desktop_ui_utils.py external_ocr_bridge.py tests
pytest tests -m "not integration" --cov=desktop_ui_utils --cov-report=term-missing --cov-report=xml:coverage.xml
```

### Optional integration tests (model dependent)

```bash
RUN_INTEGRATION_TESTS=1 pytest tests/integration -m integration
```

## Deployment notes

- **Desktop app**: run `python desktop_ui.py` for operator workflow.
- **Batch/CLI**: use `eval.py` in scheduled/batch pipelines.
- **Production recommendation**: wrap `eval.py`/pipeline calls behind a service API and externalize model paths/config for environment-specific deployment.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
