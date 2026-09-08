"""
Evaluation & Benchmarking Script for Persian ALPR Pipeline
Author: Arash Mohammadrezaei

Usage:
    python eval.py --source car_c.jpg --weights yolo11_anpr_ghd.pt --benchmark
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from image_classifier import ImageClassifier
from license_plate_extractor import extract_license_plate_and_digits


PERSIAN_CHAR_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'الف', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and benchmark Persian ALPR pipeline.")
    parser.add_argument("--source", type=str, default="car_c.jpg", help="Path to input image or directory")
    parser.add_argument("--weights", type=str, default="yolo11_anpr_ghd.pt", help="Path to YOLO11 plate detector weights")
    parser.add_argument("--classifier-weights", type=str, default="persian_digit_classifier.pt", help="Path to character classifier weights")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (auto if empty)")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-iteration latency & FPS benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default="results/eval_output.jpg", help="Output destination for annotated frame")
    return parser.parse_args()


def benchmark_pipeline(detector, classifier, image_path, iterations=50, device="cuda"):
    print(f"\n[BENCHMARK] Warming up inference pipeline on {device.upper()}...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load {image_path}")
        return

    # Warmup
    for _ in range(5):
        _ = detector(frame, verbose=False)

    latencies = []
    print(f"[BENCHMARK] Executing {iterations} timing iterations...")
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = detector(frame, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / avg_latency

    print("\n" + "="*50)
    print("       PERSIAN ALPR PIPELINE BENCHMARK REPORT       ")
    print("="*50)
    print(f" Device                 : {device.upper()}")
    print(f" Input Image            : {os.path.basename(image_path)}")
    print(f" Average Latency        : {avg_latency:.2f} ms")
    print(f" P95 Latency            : {p95_latency:.2f} ms")
    print(f" Throughput (FPS)       : ~{fps:.1f} FPS")
    print("="*50 + "\n")


def run_single_inference(detector, classifier, image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    t_start = time.perf_counter()
    results = detector(frame, verbose=False)
    boxes = results[0].boxes

    print(f"[INFERENCE] Detected {len(boxes)} candidate bounding box(es).")
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"[INFERENCE] Completed in {elapsed_ms:.2f} ms.")


def main():
    args = parse_args()

    selected_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Initializing detector on target device: {selected_device}")

    if not os.path.exists(args.weights):
        print(f"Warning: Model weights {args.weights} not found locally.")

    detector = YOLO(args.weights)

    classifier = None
    if os.path.exists(args.classifier_weights):
        try:
            classifier = ImageClassifier(args.classifier_weights, PERSIAN_CHAR_CLASSES)
        except Exception as e:
            print(f"[WARN] Classifier could not be loaded: {e}")

    if args.benchmark:
        benchmark_pipeline(detector, classifier, args.source, iterations=args.iterations, device=selected_device)
    else:
        run_single_inference(detector, classifier, args.source, args.output)


if __name__ == "__main__":
    main()
