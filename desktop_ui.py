import csv
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
from PySide6.QtCore import QMutex, QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QIcon, QImage, QPixmap
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

DEFAULT_MODEL_NAME = "yolo11_anpr_ghd.pt"
DEFAULT_OCR_MODEL_NAME = "persian_digit_classifier.pt"
DEFAULT_OUTPUT_DIR = "outputs"
THREAD_STOP_TIMEOUT_MS = 2000
OCR_MIN_AREA = 0.005
OCR_MAX_AREA = 0.05
OCR_CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "Alef", "BE", "ch", "d", "ein", "f", "g", "ghaf", "ghein", "h2",
    "hj", "j", "k", "kh", "l", "m", "n", "p", "r", "s",
    "sad", "sh", "t", "ta", "th", "Vav", "y", "z", "za", "zad", "zal", "zh",
]


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
        image /= 255.0 if image.max() > 1.0 else 1.0
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
        self._last_saved_time = 0.0
        self._records = []
        self._ocr_status_message = ""
        self._classifier = None

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
            model = YOLO(self.config.weights_path)
            self._init_ocr()
            self.status_ready.emit({"state": "در حال اجرا"})

            if self.config.source_type == "image":
                frame = cv2.imread(self.config.input_path)
                if frame is None:
                    raise ValueError("تصویر ورودی قابل خواندن نیست.")
                annotated, detections, fps = self._process_frame(model, frame, 0)
                self.frame_ready.emit(annotated)
                self._emit_detections(detections)
                self.status_ready.emit({"fps": fps, "state": "اتمام"})
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
                    self.status_ready.emit({"fps": fps, "state": "در حال اجرا", "frame": frame_idx})
                    frame_idx += 1

                cap.release()
                if not self._stop:
                    self.status_ready.emit({"state": "اتمام"})

            if self.config.auto_save_log and self._records:
                log_name = datetime.now().strftime("results_%Y%m%d_%H%M%S.csv")
                self._write_log(Path(self.config.output_dir) / log_name)
        except Exception as exc:
            self.error.emit(str(exc))

    def _init_ocr(self):
        try:
            weights_path = Path(__file__).with_name(DEFAULT_OCR_MODEL_NAME)
            if not weights_path.is_file():
                self._ocr_status_message = f"مدل OCR یافت نشد: {weights_path}"
                return
            self._classifier = PlateCharClassifier(str(weights_path), OCR_CLASS_NAMES)
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
                fieldnames=["timestamp", "frame_index", "plate_text", "confidence", "crop_path", "annotated_path"],
            )
            writer.writeheader()
            writer.writerows(self._records)

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
            allow_save = self.config.dedupe_interval <= 0 or (now - self._last_saved_time) >= self.config.dedupe_interval
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
                        plate_text = self._recognize_plate_text(crop)

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

                if allow_save:
                    if crop is None:
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
                            "crop_path": crop_path,
                            "annotated_path": "",
                        }
                    )

            if allow_save and self.config.save_annotated and detections:
                ann_name = f"{frame_stamp}_{frame_idx}_annotated.jpg"
                annotated_path = str(Path(self.config.output_dir) / ann_name)
                cv2.imwrite(annotated_path, annotated)
                for item in detections:
                    item["annotated_path"] = annotated_path

            if allow_save:
                self._last_saved_time = now

        fps = 1.0 / max(time.perf_counter() - start, 1e-6)
        return annotated, detections, fps

    def _recognize_plate_text(self, plate_crop):
        if self._classifier is None:
            return ""
        try:
            rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            straight = straighten_skewed_rectangle(rgb)
            gray = cv2.cvtColor(straight, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
            img_area = thresh.shape[0] * thresh.shape[1]
            digits = []
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                if area > OCR_MIN_AREA * img_area and area <= OCR_MAX_AREA * img_area and (w <= h or abs(w - h) < 10):
                    component_mask = (labels == i).astype("uint8") * 255
                    digit = component_mask[y:y + h, x:x + w]
                    digit = cv2.resize(digit, (26, 26), interpolation=cv2.INTER_AREA)
                    digit = np.pad(digit, (2, 2), "constant", constant_values=0).astype(float) / 255.0
                    digits.append((x, digit))
            if not digits:
                return ""
            digits.sort(key=lambda item: item[0])
            chars = [self._classifier.predict(1.0 - digit) for _, digit in digits]
            return "".join(chars)
        except Exception:
            return ""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("سامانه تشخیص پلاک فارسی")
        self.resize(1500, 860)
        self.thread = None
        self.records = []

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        self._build_toolbar()

        root = QWidget()
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

        left_widget = QFrame()
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
        left_layout.addStretch(1)

        self.preview_label = QLabel("پیش‌نمایش")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(700, 500)
        self.preview_label.setFrameStyle(QFrame.StyledPanel)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.addWidget(self.preview_label)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["زمان", "فریم", "متن پلاک", "اطمینان", "تصویر"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
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

    def _build_status_bar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)

        self.fps_label = QLabel("FPS: 0.0")
        self.device_label = QLabel("دستگاه: CPU")
        self.state_label = QLabel("وضعیت: آماده")
        self.error_label = QLabel("")

        bar.addPermanentWidget(self.fps_label)
        bar.addPermanentWidget(self.device_label)
        bar.addPermanentWidget(self.state_label)
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
            QMainWindow { background: #0f172a; color: #e2e8f0; }
            QToolBar { background: #111827; spacing: 8px; padding: 8px; border: none; }
            QStatusBar { background: #111827; color: #e2e8f0; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget {
                background: #1f2937; color: #e5e7eb; border: 1px solid #374151; border-radius: 8px; padding: 6px;
            }
            QPushButton { background: #2563eb; color: white; border-radius: 8px; padding: 7px 10px; }
            QPushButton:hover { background: #1d4ed8; }
            QLabel { color: #e2e8f0; }
            QHeaderView::section { background: #111827; color: #e2e8f0; padding: 6px; border: none; }
            QTableWidget { gridline-color: #374151; selection-background-color: #1d4ed8; }
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
        )

    def start_inference(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "در حال اجرا", "پردازش در حال اجرا است.")
            return

        config = self._current_config()
        if not config.weights_path or not os.path.isfile(config.weights_path):
            QMessageBox.warning(self, "خطا", "مسیر وزن مدل معتبر نیست.")
            return
        if config.source_type in {"image", "video"} and (not config.input_path or not os.path.isfile(config.input_path)):
            QMessageBox.warning(self, "خطا", "مسیر ورودی معتبر نیست.")
            return

        self.results_table.setRowCount(0)
        self.records = []
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

    def _on_status(self, status):
        if "fps" in status:
            self.fps_label.setText(f"FPS: {status['fps']:.1f}")
        if "state" in status:
            self.state_label.setText(f"وضعیت: {status['state']}")

    def _on_error(self, message):
        self.error_label.setText(f"خطا: {message}")
        QMessageBox.critical(self, "خطا", message)

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
                fieldnames=["timestamp", "frame_index", "plate_text", "confidence", "crop_path", "annotated_path"],
            )
            writer.writeheader()
            writer.writerows(self.records)

        QMessageBox.information(self, "خروجی", "فایل نتایج ذخیره شد.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.RightToLeft)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
