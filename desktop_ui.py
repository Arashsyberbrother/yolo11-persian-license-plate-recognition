import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtCore import QMutex, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO
from desktop_ui_utils import ensure_output_dir_writable, normalize_plate_text, register_plate_event

DEFAULT_MODEL_NAME = "yolo11_anpr_ghd.pt"
DEFAULT_OCR_MODEL_NAME = "persian_digit_classifier.pt"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SETTINGS_ORG = "Arashsyberbrother"
DEFAULT_SETTINGS_APP = "PersianPlateDesktopUI"
THREAD_STOP_TIMEOUT_MS = 2000
OCR_MIN_AREA = 0.005
OCR_MAX_AREA = 0.05
OCR_MAX_ASPECT_DIFF = 10
OCR_DIGIT_SIZE = 26
OCR_PADDING = 2
OCR_CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "Alef", "BE", "ch", "d", "ein", "f", "g", "ghaf", "ghein", "h2",
    "hj", "j", "k", "kh", "l", "m", "n", "p", "r", "s",
    "sad", "sh", "t", "ta", "th", "Vav", "y", "z", "za", "zad", "zal", "zh",
]
OCR_LABEL_TO_CHAR = {
    "Alef": "ا",
    "BE": "ب",
    "ch": "چ",
    "d": "د",
    "ein": "ع",
    "f": "ف",
    "g": "گ",
    "ghaf": "ق",
    "ghein": "غ",
    "h2": "ه",
    "hj": "ح",
    "j": "ج",
    "k": "ک",
    "kh": "خ",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "p": "پ",
    "r": "ر",
    "s": "س",
    "sad": "ص",
    "sh": "ش",
    "t": "ت",
    "ta": "ط",
    "th": "ث",
    "Vav": "و",
    "y": "ی",
    "z": "ز",
    "za": "ض",
    "zad": "ظ",
    "zal": "ذ",
    "zh": "ژ",
}


def straighten_skewed_rectangle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) < 2:
        return img
    longest_lines = sorted(lines, key=lambda l: np.linalg.norm((l[0][2] - l[0][0], l[0][3] - l[0][1])), reverse=True)[:2]
    angles = []
    for line in longest_lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    average_angle = np.mean(angles)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, average_angle, 1.0)
    cos_theta = abs(matrix[0, 0])
    sin_theta = abs(matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return cv2.warpAffine(img, matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class FCModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PlateCharClassifier:
    def __init__(self, weights_path, class_names):
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FCModel(len(class_names))
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predicted = int(torch.argmax(outputs, dim=1).item())
        return self.class_names[predicted]


@dataclass
class InferenceConfig:
    source_type: str
    input_path: str
    weights_path: str
    device: str
    conf: float
    iou: float
    output_dir: str
    dedupe_interval: int
    save_annotated: bool
    auto_save_log: bool
    debug_ocr: bool


class InferenceThread(QThread):
    frame_ready = Signal(object)
    detection_ready = Signal(dict)
    status_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self._stop = False
        self._pause = False
        self._mutex = QMutex()
        self._records = []
        self._ocr_status_message = ""
        self._classifier = None
        self._ocr_debug_dir = None
        self._plate_last_seen = {}
        self._plate_duplicate_counts = {}
        self._fps_sum = 0.0
        self._fps_samples = 0
        self._frames_processed = 0
        self._confidence_sum = 0.0
        self._started_at = None

    def stop(self):
        self._mutex.lock()
        self._stop = True
        self._pause = False
        self._mutex.unlock()

    def toggle_pause(self):
        self._mutex.lock()
        self._pause = not self._pause
        paused = self._pause
        self._mutex.unlock()
        return paused

    def _check_flags(self):
        self._mutex.lock()
        stop, pause = self._stop, self._pause
        self._mutex.unlock()
        return stop, pause

    def run(self):
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._started_at = datetime.now()
            model = YOLO(self.config.weights_path)
            self._init_ocr()
            self.status_ready.emit({"state": "در حال اجرا", **self._stats_payload()})

            if self.config.source_type == "image":
                frame = cv2.imread(self.config.input_path)
                if frame is None:
                    raise ValueError("تصویر ورودی قابل خواندن نیست.")
                annotated, detections, fps = self._process_frame(model, frame, 0)
                self.frame_ready.emit(annotated)
                self._emit_detections(detections)
                self._update_stats(detections, fps)
                self.status_ready.emit({"fps": fps, "state": "اتمام", **self._stats_payload()})
            else:
                source = 0 if self.config.source_type == "webcam" else self.config.input_path
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise ValueError("منبع ویدئویی باز نشد.")

                frame_idx = 0
                while True:
                    stop, pause = self._check_flags()
                    if stop:
                        break
                    if pause:
                        self.status_ready.emit({"state": "مکث"})
                        time.sleep(0.05)
                        continue

                    ok, frame = cap.read()
                    if not ok:
                        break

                    annotated, detections, fps = self._process_frame(model, frame, frame_idx)
                    self.frame_ready.emit(annotated)
                    self._emit_detections(detections)
                    self._update_stats(detections, fps)
                    self.status_ready.emit({"fps": fps, "state": "در حال اجرا", "frame": frame_idx, **self._stats_payload()})
                    frame_idx += 1

                cap.release()
                if not self._stop:
                    self.status_ready.emit({"state": "اتمام", **self._stats_payload()})

            if self.config.auto_save_log and self._records:
                log_name = datetime.now().strftime("results_%Y%m%d_%H%M%S.csv")
                self._write_log(Path(self.config.output_dir) / log_name)
            self._write_summary(Path(self.config.output_dir) / datetime.now().strftime("summary_%Y%m%d_%H%M%S.json"))
        except Exception as exc:
            self.error.emit(str(exc))

    def _init_ocr(self):
        try:
            weights_path = Path(__file__).with_name(DEFAULT_OCR_MODEL_NAME)
            if not weights_path.is_file():
                self._ocr_status_message = f"مدل OCR یافت نشد ({weights_path.name})"
                return
            self._classifier = PlateCharClassifier(str(weights_path), OCR_CLASS_NAMES)
            if self.config.debug_ocr:
                self._ocr_debug_dir = Path(self.config.output_dir) / "ocr_debug"
                self._ocr_debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._ocr_status_message = f"OCR غیرفعال شد: {exc}"
            self._classifier = None

    def _emit_detections(self, detections):
        for item in detections:
            self._records.append(item)
            self.detection_ready.emit(item)

    def _write_log(self, path: Path):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "frame_index",
                    "plate_text",
                    "confidence",
                    "duplicate_count",
                    "crop_path",
                    "annotated_path",
                ],
            )
            writer.writeheader()
            writer.writerows(self._records)

    def _update_stats(self, detections, fps):
        self._frames_processed += 1
        self._fps_sum += fps
        self._fps_samples += 1
        for item in detections:
            self._confidence_sum += float(item.get("confidence", 0.0))

    def _stats_payload(self):
        detections_count = len(self._records)
        avg_conf = self._confidence_sum / max(detections_count, 1)
        avg_fps = self._fps_sum / max(self._fps_samples, 1)
        return {
            "detections_count": detections_count,
            "avg_confidence": avg_conf,
            "avg_fps": avg_fps,
            "frames_processed": self._frames_processed,
        }

    def _write_summary(self, path: Path):
        finished_at = datetime.now()
        duration = (finished_at - self._started_at).total_seconds() if self._started_at else 0.0
        unique_plates = sorted({r.get("plate_text", "") for r in self._records if r.get("plate_text")})
        summary = {
            "started_at": self._started_at.isoformat() if self._started_at else "",
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(duration, 3),
            "frames_processed": self._frames_processed,
            "detections_saved": len(self._records),
            "unique_plate_count": len(unique_plates),
            "unique_plates": unique_plates,
            "average_confidence": round(self._confidence_sum / max(len(self._records), 1), 4),
            "average_fps": round(self._fps_sum / max(self._fps_samples, 1), 3),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _process_frame(self, model: YOLO, frame, frame_idx: int):
        start = time.perf_counter()
        results = model.predict(
            frame,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=False,
        )

        boxes = results[0].boxes if results and results[0].boxes is not None else None
        annotated = frame.copy()
        detections = []

        if boxes is not None and len(boxes) > 0:
            now = time.time()
            frame_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for det_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                frame_h, frame_w = frame.shape[:2]
                x1c = max(0, min(x1, frame_w))
                y1c = max(0, min(y1, frame_h))
                x2c = max(0, min(x2, frame_w))
                y2c = max(0, min(y2, frame_h))
                plate_text = ""
                crop = None
                if x2c > x1c and y2c > y1c:
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        plate_text = self._recognize_plate_text(crop, f"{frame_stamp}_{frame_idx}_{det_idx}")

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    annotated,
                    plate_text if plate_text else f"Plate {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

                if crop is None:
                    continue

                duplicate_count = 0
                should_emit = True
                if plate_text:
                    should_emit, duplicate_count = register_plate_event(
                        self._plate_last_seen,
                        self._plate_duplicate_counts,
                        plate_text,
                        now,
                        self.config.dedupe_interval,
                    )
                if not should_emit:
                    continue

                crop_name = f"{frame_stamp}_{frame_idx}_{det_idx}.jpg"
                crop_path = str(Path(self.config.output_dir) / crop_name)
                cv2.imwrite(crop_path, crop)

                detections.append(
                    {
                        "timestamp": frame_time,
                        "frame_index": frame_idx,
                        "plate_text": plate_text if plate_text else self._ocr_status_message,
                        "confidence": round(conf, 4),
                        "duplicate_count": duplicate_count,
                        "crop_path": crop_path,
                        "annotated_path": "",
                    }
                )

            if self.config.save_annotated and detections:
                ann_name = f"{frame_stamp}_{frame_idx}_annotated.jpg"
                annotated_path = str(Path(self.config.output_dir) / ann_name)
                cv2.imwrite(annotated_path, annotated)
                for item in detections:
                    item["annotated_path"] = annotated_path

        fps = 1.0 / max(time.perf_counter() - start, 1e-6)
        return annotated, detections, fps

    def _recognize_plate_text(self, plate_crop, debug_tag=None):
        if self._classifier is None:
            return ""
        try:
            rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            straight = straighten_skewed_rectangle(rgb)
            gray = cv2.cvtColor(straight, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            threshold_candidates = [
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    31,
                    5,
                ),
            ]
            area_ranges = [
                (OCR_MIN_AREA, OCR_MAX_AREA),
                (OCR_MIN_AREA * 0.5, max(0.12, OCR_MAX_AREA * 2.4)),
            ]

            digits = []
            best_thresh = threshold_candidates[0]
            for thresh in threshold_candidates:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
                img_area = thresh.shape[0] * thresh.shape[1]
                for min_area_ratio, max_area_ratio in area_ranges:
                    candidate_digits = []
                    for i in range(1, num_labels):
                        x = stats[i, cv2.CC_STAT_LEFT]
                        y = stats[i, cv2.CC_STAT_TOP]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area > min_area_ratio * img_area and area <= max_area_ratio * img_area and (
                            w <= h or abs(w - h) < OCR_MAX_ASPECT_DIFF
                        ):
                            component_mask = (labels == i).astype("uint8") * 255
                            digit = component_mask[y:y + h, x:x + w]
                            digit = cv2.resize(digit, (OCR_DIGIT_SIZE, OCR_DIGIT_SIZE), interpolation=cv2.INTER_AREA)
                            digit = np.pad(digit, (OCR_PADDING, OCR_PADDING), "constant", constant_values=0).astype(float) / 255.0
                            candidate_digits.append((x, digit))
                    if len(candidate_digits) > len(digits):
                        digits = candidate_digits
                        best_thresh = thresh
                    if len(digits) >= 6:
                        break
                if len(digits) >= 6:
                    break
            if not digits:
                return ""
            digits.sort(key=lambda item: item[0])
            predictions = [self._classifier.predict(1.0 - digit) for _, digit in digits]
            chars = [OCR_LABEL_TO_CHAR.get(pred, pred) for pred in predictions]
            if self.config.debug_ocr and self._ocr_debug_dir and debug_tag:
                cv2.imwrite(str(self._ocr_debug_dir / f"{debug_tag}_crop.jpg"), plate_crop)
                cv2.imwrite(str(self._ocr_debug_dir / f"{debug_tag}_straight.jpg"), cv2.cvtColor(straight, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(self._ocr_debug_dir / f"{debug_tag}_thresh.jpg"), best_thresh)
            return normalize_plate_text("".join(chars))
        except Exception:
            return ""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("سامانه تشخیص پلاک فارسی")
        self.resize(1500, 860)
        self.thread = None
        self.records = []
        self._fps_sum = 0.0
        self._fps_samples = 0
        self.settings = QSettings(DEFAULT_SETTINGS_ORG, DEFAULT_SETTINGS_APP)

        self._build_ui()
        self._load_settings()
        self._apply_style()

    def _build_ui(self):
        self._build_toolbar()

        root = QWidget()
        root.setObjectName("rootSurface")
        self.setCentralWidget(root)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["تصویر", "ویدئو", "وب‌کم"])

        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("انتخاب")
        self.input_btn.clicked.connect(self._open_input)

        self.weights_edit = QLineEdit(str(Path(__file__).with_name(DEFAULT_MODEL_NAME)))
        self.weights_btn = QPushButton("وزن مدل")
        self.weights_btn.clicked.connect(self._browse_weights)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA", "0")

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.35)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)

        self.output_edit = QLineEdit(str(Path(__file__).with_name(DEFAULT_OUTPUT_DIR)))
        self.output_btn = QPushButton("پوشه خروجی")
        self.output_btn.clicked.connect(self._browse_output)

        self.dedupe_interval_spin = QSpinBox()
        self.dedupe_interval_spin.setRange(0, 300)
        self.dedupe_interval_spin.setValue(2)
        self.dedupe_interval_spin.setSuffix(" ثانیه")

        self.save_annotated_cb = QCheckBox("ذخیره فریم‌های حاشیه‌نویسی‌شده")
        self.auto_log_cb = QCheckBox("ذخیره خودکار گزارش CSV")
        self.auto_log_cb.setChecked(True)
        self.debug_ocr_cb = QCheckBox("حالت دیباگ OCR (ذخیره مراحل میانی)")

        left_widget = QFrame()
        left_widget.setObjectName("glassPanel")
        left_layout = QVBoxLayout(left_widget)
        form = QFormLayout()
        form.addRow("نوع ورودی", self.source_combo)

        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.input_btn)
        form.addRow("فایل ورودی", input_row)

        weights_row = QWidget()
        weights_layout = QHBoxLayout(weights_row)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        weights_layout.addWidget(self.weights_edit)
        weights_layout.addWidget(self.weights_btn)
        form.addRow("وزن مدل", weights_row)

        form.addRow("دستگاه", self.device_combo)
        form.addRow("آستانه اطمینان", self.conf_spin)
        form.addRow("آستانه IoU", self.iou_spin)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_btn)
        form.addRow("پوشه خروجی", output_row)

        form.addRow("بازه حذف تکرار", self.dedupe_interval_spin)
        left_layout.addLayout(form)
        left_layout.addWidget(self.save_annotated_cb)
        left_layout.addWidget(self.auto_log_cb)
        left_layout.addWidget(self.debug_ocr_cb)
        self.open_output_btn = QPushButton("باز کردن پوشه خروجی")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        left_layout.addWidget(self.open_output_btn)
        self.clear_results_btn = QPushButton("پاک‌کردن نتایج")
        self.clear_results_btn.clicked.connect(self.clear_results)
        left_layout.addWidget(self.clear_results_btn)
        left_layout.addStretch(1)

        self.preview_label = QLabel("پیش‌نمایش")
        self.preview_label.setObjectName("previewGlass")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(700, 500)
        self.preview_label.setFrameStyle(QFrame.StyledPanel)

        center_widget = QWidget()
        center_widget.setObjectName("glassPanel")
        center_layout = QVBoxLayout(center_widget)
        center_layout.addWidget(self.preview_label)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setObjectName("glassTable")
        self.results_table.setHorizontalHeaderLabels(["زمان", "فریم", "متن پلاک", "اطمینان", "تصویر"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSortingEnabled(True)

        self.filter_text_edit = QLineEdit()
        self.filter_text_edit.setPlaceholderText("فیلتر متن پلاک")
        self.filter_text_edit.textChanged.connect(self._apply_results_filter)
        self.filter_conf_spin = QDoubleSpinBox()
        self.filter_conf_spin.setRange(0.0, 1.0)
        self.filter_conf_spin.setSingleStep(0.05)
        self.filter_conf_spin.setValue(0.0)
        self.filter_conf_spin.valueChanged.connect(self._apply_results_filter)

        right_widget = QWidget()
        right_widget.setObjectName("glassPanel")
        right_layout = QVBoxLayout(right_widget)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.filter_text_edit)
        filter_row.addWidget(self.filter_conf_spin)
        right_layout.addLayout(filter_row)
        right_layout.addWidget(self.results_table)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([320, 820, 360])

        layout = QVBoxLayout(root)
        layout.addWidget(splitter)

        self._build_status_bar()

    def _build_toolbar(self):
        toolbar = QToolBar("ابزار")
        self.addToolBar(toolbar)

        self.run_action = QAction(QIcon.fromTheme("media-playback-start"), "اجرا", self)
        self.run_action.triggered.connect(self.start_inference)
        toolbar.addAction(self.run_action)

        self.pause_action = QAction("مکث", self)
        self.pause_action.triggered.connect(self.toggle_pause)
        toolbar.addAction(self.pause_action)

        self.stop_action = QAction(QIcon.fromTheme("media-playback-stop"), "توقف", self)
        self.stop_action.triggered.connect(self.stop_inference)
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        self.open_action = QAction("باز کردن ورودی", self)
        self.open_action.triggered.connect(self._open_input)
        toolbar.addAction(self.open_action)

        self.export_action = QAction("خروجی نتایج", self)
        self.export_action.triggered.connect(self.export_results)
        toolbar.addAction(self.export_action)

        self.open_output_action = QAction("باز کردن خروجی", self)
        self.open_output_action.triggered.connect(self.open_output_folder)
        toolbar.addAction(self.open_output_action)

        self.clear_action = QAction("پاک‌کردن نتایج", self)
        self.clear_action.triggered.connect(self.clear_results)
        toolbar.addAction(self.clear_action)

    def _build_status_bar(self):
        bar = QStatusBar()
        bar.setObjectName("glassStatusBar")
        self.setStatusBar(bar)

        self.fps_label = QLabel("FPS: 0.0")
        self.avg_fps_label = QLabel("Avg FPS: 0.0")
        self.device_label = QLabel("دستگاه: CPU")
        self.state_label = QLabel("وضعیت: آماده")
        self.total_label = QLabel("پلاک‌ها: 0")
        self.avg_conf_label = QLabel("میانگین اطمینان: 0.00")
        self.error_label = QLabel("")

        bar.addPermanentWidget(self.fps_label)
        bar.addPermanentWidget(self.avg_fps_label)
        bar.addPermanentWidget(self.device_label)
        bar.addPermanentWidget(self.state_label)
        bar.addPermanentWidget(self.total_label)
        bar.addPermanentWidget(self.avg_conf_label)
        bar.addWidget(self.error_label, 1)

    def _apply_style(self):
        self.setLayoutDirection(Qt.RightToLeft)
        app_font = QFont("Vazirmatn")
        if app_font.family() == "":
            app_font = QFont("Segoe UI")
        app_font.setPointSize(10)
        QApplication.instance().setFont(app_font)

        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #0b1220, stop: 1 #111c32);
                color: #e2e8f0;
            }
            QWidget#rootSurface {
                background: transparent;
            }
            QFrame#glassPanel, QWidget#glassPanel {
                background: rgba(30, 41, 59, 0.58);
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 14px;
            }
            QToolBar {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(148, 163, 184, 0.18),
                    stop: 1 rgba(30, 41, 59, 0.58)
                );
                spacing: 8px;
                padding: 8px;
                border: 1px solid rgba(191, 219, 254, 0.32);
                border-radius: 12px;
                margin: 8px;
            }
            QStatusBar#glassStatusBar {
                background: rgba(15, 23, 42, 0.72);
                color: #e2e8f0;
                border-top: 1px solid rgba(148, 163, 184, 0.25);
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget {
                background: rgba(30, 41, 59, 0.62);
                color: #e5e7eb;
                border: 1px solid rgba(148, 163, 184, 0.38);
                border-radius: 10px;
                padding: 6px;
            }
            QPushButton {
                background: rgba(37, 99, 235, 0.92);
                color: white;
                border-radius: 10px;
                border: 1px solid rgba(191, 219, 254, 0.45);
                padding: 7px 10px;
            }
            QPushButton:hover { background: rgba(29, 78, 216, 0.95); }
            QLabel { color: #e2e8f0; }
            QLabel#previewGlass {
                background: rgba(15, 23, 42, 0.62);
                border: 1px solid rgba(148, 163, 184, 0.3);
                border-radius: 12px;
            }
            QHeaderView::section {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(191, 219, 254, 0.22),
                    stop: 1 rgba(30, 41, 59, 0.68)
                );
                color: #e2e8f0;
                padding: 6px;
                border: 1px solid rgba(191, 219, 254, 0.2);
            }
            QTableWidget#glassTable {
                gridline-color: rgba(148, 163, 184, 0.2);
                selection-background-color: rgba(37, 99, 235, 0.8);
            }
            """
        )

    def _open_input(self):
        source_type = self._source_type_key()
        if source_type == "image":
            path, _ = QFileDialog.getOpenFileName(self, "انتخاب تصویر", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        elif source_type == "video":
            path, _ = QFileDialog.getOpenFileName(self, "انتخاب ویدئو", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        else:
            self.input_edit.setText("")
            return

        if path:
            self.input_edit.setText(path)

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "انتخاب وزن مدل", "", "PyTorch (*.pt)")
        if path:
            self.weights_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "انتخاب پوشه خروجی")
        if path:
            self.output_edit.setText(path)

    def _source_type_key(self):
        text = self.source_combo.currentText()
        if text == "تصویر":
            return "image"
        if text == "ویدئو":
            return "video"
        return "webcam"

    def _save_settings(self):
        self.settings.setValue("source_index", self.source_combo.currentIndex())
        self.settings.setValue("input_path", self.input_edit.text())
        self.settings.setValue("weights_path", self.weights_edit.text())
        self.settings.setValue("device_index", self.device_combo.currentIndex())
        self.settings.setValue("conf", self.conf_spin.value())
        self.settings.setValue("iou", self.iou_spin.value())
        self.settings.setValue("output_dir", self.output_edit.text())
        self.settings.setValue("dedupe_interval", self.dedupe_interval_spin.value())
        self.settings.setValue("save_annotated", self.save_annotated_cb.isChecked())
        self.settings.setValue("auto_log", self.auto_log_cb.isChecked())
        self.settings.setValue("debug_ocr", self.debug_ocr_cb.isChecked())
        self.settings.setValue("filter_text", self.filter_text_edit.text())
        self.settings.setValue("filter_conf", self.filter_conf_spin.value())

    def _load_settings(self):
        self.source_combo.setCurrentIndex(self.settings.value("source_index", 0, type=int))
        self.input_edit.setText(self.settings.value("input_path", "", type=str))
        self.weights_edit.setText(self.settings.value("weights_path", self.weights_edit.text(), type=str))
        self.device_combo.setCurrentIndex(min(self.settings.value("device_index", 0, type=int), self.device_combo.count() - 1))
        self.conf_spin.setValue(self.settings.value("conf", 0.35, type=float))
        self.iou_spin.setValue(self.settings.value("iou", 0.45, type=float))
        self.output_edit.setText(self.settings.value("output_dir", self.output_edit.text(), type=str))
        self.dedupe_interval_spin.setValue(self.settings.value("dedupe_interval", 2, type=int))
        self.save_annotated_cb.setChecked(self.settings.value("save_annotated", False, type=bool))
        self.auto_log_cb.setChecked(self.settings.value("auto_log", True, type=bool))
        self.debug_ocr_cb.setChecked(self.settings.value("debug_ocr", False, type=bool))
        self.filter_text_edit.setText(self.settings.value("filter_text", "", type=str))
        self.filter_conf_spin.setValue(self.settings.value("filter_conf", 0.0, type=float))

    def _validate_config(self, config: InferenceConfig):
        if not config.weights_path:
            return "مسیر وزن مدل خالی است."
        if not os.path.isfile(config.weights_path):
            return f"فایل وزن مدل پیدا نشد:\n{config.weights_path}"
        if config.source_type in {"image", "video"}:
            if not config.input_path:
                return "فایل ورودی را انتخاب کنید."
            if not os.path.isfile(config.input_path):
                return f"فایل ورودی پیدا نشد:\n{config.input_path}"
        ok, message = ensure_output_dir_writable(config.output_dir)
        if not ok:
            return message
        return ""

    def _current_config(self):
        return InferenceConfig(
            source_type=self._source_type_key(),
            input_path=self.input_edit.text().strip(),
            weights_path=self.weights_edit.text().strip(),
            device=self.device_combo.currentData(),
            conf=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            output_dir=self.output_edit.text().strip(),
            dedupe_interval=int(self.dedupe_interval_spin.value()),
            save_annotated=self.save_annotated_cb.isChecked(),
            auto_save_log=self.auto_log_cb.isChecked(),
            debug_ocr=self.debug_ocr_cb.isChecked(),
        )

    def start_inference(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "در حال اجرا", "پردازش در حال اجرا است.")
            return

        config = self._current_config()
        validation_error = self._validate_config(config)
        if validation_error:
            QMessageBox.warning(self, "خطا", validation_error)
            return

        self.results_table.setRowCount(0)
        self.records = []
        self._fps_sum = 0.0
        self._fps_samples = 0
        self.error_label.setText("")
        self.device_label.setText(f"دستگاه: {self.device_combo.currentText()}")
        self.state_label.setText("وضعیت: در حال اجرا")

        self.thread = InferenceThread(config)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.detection_ready.connect(self._on_detection)
        self.thread.status_ready.connect(self._on_status)
        self.thread.error.connect(self._on_error)
        self.thread.finished.connect(lambda: self.state_label.setText("وضعیت: آماده"))
        self.thread.start()
        self._save_settings()

    def toggle_pause(self):
        if not self.thread or not self.thread.isRunning():
            return
        paused = self.thread.toggle_pause()
        self.pause_action.setText("ادامه" if paused else "مکث")

    def stop_inference(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            stopped = self.thread.wait(THREAD_STOP_TIMEOUT_MS)
            if not stopped:
                self.error_label.setText("خطا: توقف پردازش زمان‌بر شد.")
            self.state_label.setText("وضعیت: متوقف")
            self.pause_action.setText("مکث")

    def _on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    def _on_detection(self, item):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(item["timestamp"]))
        self.results_table.setItem(row, 1, QTableWidgetItem(str(item["frame_index"])))
        self.results_table.setItem(row, 2, QTableWidgetItem(item.get("plate_text", "")))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{item['confidence']:.2f}"))

        thumb_item = QTableWidgetItem(Path(item["crop_path"]).name)
        pix = QPixmap(item["crop_path"]).scaled(100, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumb_item.setIcon(QIcon(pix))
        self.results_table.setItem(row, 4, thumb_item)
        self.records.append(item)
        self._apply_results_filter()

    def _on_status(self, status):
        if "fps" in status:
            self.fps_label.setText(f"FPS: {status['fps']:.1f}")
            self._fps_sum += status["fps"]
            self._fps_samples += 1
            self.avg_fps_label.setText(f"Avg FPS: {self._fps_sum / max(self._fps_samples, 1):.1f}")
        if "state" in status:
            self.state_label.setText(f"وضعیت: {status['state']}")
        if "detections_count" in status:
            self.total_label.setText(f"پلاک‌ها: {int(status['detections_count'])}")
        if "avg_confidence" in status:
            self.avg_conf_label.setText(f"میانگین اطمینان: {status['avg_confidence']:.2f}")

    def _on_error(self, message):
        self.error_label.setText(f"خطا: {message}")
        QMessageBox.critical(self, "خطا", message)

    def _apply_results_filter(self):
        filter_text = normalize_plate_text(self.filter_text_edit.text())
        min_conf = float(self.filter_conf_spin.value())
        for row in range(self.results_table.rowCount()):
            plate_item = self.results_table.item(row, 2)
            conf_item = self.results_table.item(row, 3)
            plate_text = normalize_plate_text(plate_item.text() if plate_item else "")
            try:
                conf = float(conf_item.text()) if conf_item else 0.0
            except (ValueError, TypeError):
                conf = 0.0
            visible = (not filter_text or filter_text in plate_text) and conf >= min_conf
            self.results_table.setRowHidden(row, not visible)

    def clear_results(self):
        self.results_table.setRowCount(0)
        self.records = []
        self.total_label.setText("پلاک‌ها: 0")
        self.avg_conf_label.setText("میانگین اطمینان: 0.00")
        self.fps_label.setText("FPS: 0.0")
        self.avg_fps_label.setText("Avg FPS: 0.0")

    def open_output_folder(self):
        output_dir = self.output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "خطا", "پوشه خروجی مشخص نشده است.")
            return
        ok, message = ensure_output_dir_writable(output_dir)
        if not ok:
            QMessageBox.warning(self, "خطا", message)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def export_results(self):
        if not self.records:
            QMessageBox.information(self, "خروجی", "نتیجه‌ای برای ذخیره وجود ندارد.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "ذخیره نتایج", "results.csv", "CSV (*.csv)")
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "frame_index",
                    "plate_text",
                    "confidence",
                    "duplicate_count",
                    "crop_path",
                    "annotated_path",
                ],
            )
            writer.writeheader()
            writer.writerows(self.records)

        QMessageBox.information(self, "خروجی", "فایل نتایج ذخیره شد.")

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.RightToLeft)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
