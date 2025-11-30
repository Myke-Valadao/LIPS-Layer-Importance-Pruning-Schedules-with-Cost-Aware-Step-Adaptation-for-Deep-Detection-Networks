#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_yolo11n_video.py

Compare two YOLOv11n models (original vs pruned) on a video,
displaying side-by-side inference results with GFLOPs and
inference time statistics.

Example usage:
  python comparing.py \
    --video /path/to/video.mp4 \
    --model_orig yolov11n.pt \
    --model_pruned yolo11n_pruned.pt \
    --args_yaml args.yaml \
    --save comparison_output.mp4
"""

import argparse
import time
import yaml
import cv2
import numpy as np
from ultralytics import YOLO


def load_args_yaml(path: str):
    """Load args.yaml and extract inference parameters with safe defaults."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    imgsz = int(data.get("imgsz", 640))

    # Handle conf that may be null/None in YAML
    conf = data.get("conf", None)
    if conf is None:
        conf = 0.25  # default confidence threshold
    else:
        conf = float(conf)

    # Same idea for IOU (but usually it comes as a valid number)
    iou = data.get("iou", 0.7)
    if iou is None:
        iou = 0.7
    else:
        iou = float(iou)

    device = data.get("device", "0")
    half = bool(data.get("half", False))
    max_det = int(data.get("max_det", 300))
    classes = data.get("classes", None)

    return {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": device,
        "half": half,
        "max_det": max_det,
        "classes": classes,
    }



def try_get_gflops(model) -> float | None:
    """Try to obtain GFLOPs via model.info(); fallback to None."""
    try:
        info = model.info(detailed=False, verbose=False)
        if isinstance(info, dict):
            for k in ["gf", "GFLOPs", "gflops"]:
                if k in info and info[k] is not None:
                    return float(info[k])
    except Exception:
        pass
    return None


def put_text_block(
    img: np.ndarray,
    lines: list[str],
    org=(10, 25),
    line_height: int = 22,
    font_scale: float = 0.7,
    thickness: int = 2,
):
    """Draw multiple lines of text on top of an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        y += line_height


def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv11n original vs pruned models on video")

    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model_orig", type=str, required=True, help="Original YOLOv11n model (.pt)")
    parser.add_argument("--model_pruned", type=str, required=True, help="Pruned YOLOv11n model (.pt)")
    parser.add_argument("--args_yaml", type=str, default="args.yaml", help="Path to args.yaml")

    parser.add_argument("--gflops_orig", type=float, default=None, help="GFLOPs of original model (override)")
    parser.add_argument("--gflops_pruned", type=float, default=None, help="GFLOPs of pruned model (override)")

    parser.add_argument("--frame_step", type=int, default=1, help="Process every Nth frame (default = 1)")

    parser.add_argument("--save", type=str, default=None, help="Output video file path to save comparison")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1) Load args.yaml inference parameters
    # ---------------------------------------------------------
    infer_cfg = load_args_yaml(args.args_yaml)
    print("Loaded inference parameters from args.yaml:")
    for k, v in infer_cfg.items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------------
    # 2) Load models
    # ---------------------------------------------------------
    print(f"\nLoading original model: {args.model_orig}")
    model_orig = YOLO(args.model_orig)

    print(f"Loading pruned model:   {args.model_pruned}")
    model_pruned = YOLO(args.model_pruned)

    # ---------------------------------------------------------
    # 3) Get GFLOPs if not manually provided
    # ---------------------------------------------------------
    gflops_orig = args.gflops_orig or try_get_gflops(model_orig)
    gflops_pruned = args.gflops_pruned or try_get_gflops(model_pruned)

    print("\nEstimated GFLOPs:")
    print(f"  Original: {gflops_orig if gflops_orig else 'Unknown'}")
    print(f"  Pruned:   {gflops_pruned if gflops_pruned else 'Unknown'}")

    # ---------------------------------------------------------
    # 4) Open video
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Optional video writer
    writer = None
    if args.save is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = width * 2
        out_h = height
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (out_w, out_h))
        print(f"\nSaving comparison video to: {args.save} ({out_w}x{out_h} @ {fps_in:.2f} FPS)")

    # ---------------------------------------------------------
    # 5) Inference loop
    # ---------------------------------------------------------
    frame_idx = 0
    total_time_orig = 0.0
    total_time_pruned = 0.0
    processed_frames = 0

    window_name = "YOLOv11n Original (left) vs Pruned (right)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.frame_step != 0:
            continue

        processed_frames += 1

        # -------------- Original model inference --------------
        t0 = time.perf_counter()
        results_orig = model_orig(
            frame,
            imgsz=infer_cfg["imgsz"],
            conf=infer_cfg["conf"],
            iou=infer_cfg["iou"],
            device=infer_cfg["device"],
            half=infer_cfg["half"],
            classes=infer_cfg["classes"],
            max_det=infer_cfg["max_det"],
            verbose=False,
        )
        t1 = time.perf_counter()
        dt_orig = (t1 - t0) * 1000.0
        total_time_orig += dt_orig

        # -------------- Pruned model inference --------------
        t2 = time.perf_counter()
        results_pruned = model_pruned(
            frame,
            imgsz=infer_cfg["imgsz"],
            conf=infer_cfg["conf"],
            iou=infer_cfg["iou"],
            device=infer_cfg["device"],
            half=infer_cfg["half"],
            classes=infer_cfg["classes"],
            max_det=infer_cfg["max_det"],
            verbose=False,
        )
        t3 = time.perf_counter()
        dt_pruned = (t3 - t2) * 1000.0
        total_time_pruned += dt_pruned

        # Plot annotated frames
        img_orig = results_orig[0].plot()
        img_pruned = results_pruned[0].plot()

        # Resize if needed
        if img_orig.shape[:2] != img_pruned.shape[:2]:
            h = min(img_orig.shape[0], img_pruned.shape[0])
            w = min(img_orig.shape[1], img_pruned.shape[1])
            img_orig = cv2.resize(img_orig, (w, h))
            img_pruned = cv2.resize(img_pruned, (w, h))

        combined = np.hstack([img_orig, img_pruned])

        # Compute averages and FPS
        avg_ms_orig = total_time_orig / processed_frames
        avg_ms_pruned = total_time_pruned / processed_frames
        fps_orig = 1000.0 / avg_ms_orig
        fps_pruned = 1000.0 / avg_ms_pruned

        # Overlay text
        lines = [
            f"Frame {frame_idx} (processed: {processed_frames})",
            f"Original: {gflops_orig if gflops_orig else 'N/A'} GFLOPs | {avg_ms_orig:.1f} ms avg ~ {fps_orig:.1f} FPS",
            f"Pruned:   {gflops_pruned if gflops_pruned else 'N/A'} GFLOPs | {avg_ms_pruned:.1f} ms avg ~ {fps_pruned:.1f} FPS",
        ]

        put_text_block(combined, lines, org=(10, 25))
        put_text_block(combined, [f"{window_name}"], org=(10, 200))

        cv2.imshow(window_name, combined)

        if writer is not None:
            writer.write(combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    # ---------------------------------------------------------
    # 6) Final summary
    # ---------------------------------------------------------
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if processed_frames > 0:
        avg_ms_orig = total_time_orig / processed_frames
        avg_ms_pruned = total_time_pruned / processed_frames
        fps_orig = 1000.0 / avg_ms_orig
        fps_pruned = 1000.0 / avg_ms_pruned

        print("\n===== FINAL SUMMARY =====")
        print(f"Frames processed: {processed_frames}")
        print(
            f"Original: {avg_ms_orig:.2f} ms/frame avg  ~ {fps_orig:.2f} FPS"
            f" | GFLOPs: {gflops_orig if gflops_orig else 'N/A'}"
        )
        print(
            f"Pruned:   {avg_ms_pruned:.2f} ms/frame avg ~ {fps_pruned:.2f} FPS"
            f" | GFLOPs: {gflops_pruned if gflops_pruned else 'N/A'}"
        )
    else:
        print("No frames were processed. Check video path or frame_step value.")


if __name__ == "__main__":
    main()
