"""Standalone YOLO training script invoked by TrainingWorker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on a custom dataset")
    parser.add_argument("--weights", required=True, help="Base weights .pt path")
    parser.add_argument("--data-yaml", required=True, help="Dataset data.yaml path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in output dir")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [0]

    from ultralytics import YOLO

    last_pt = Path(args.output_dir) / "train" / "weights" / "last.pt"
    use_resume = args.resume and last_pt.is_file()
    if use_resume:
        print(f"[YOLO] Resuming from checkpoint: {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"[YOLO] Loading weights: {args.weights}")
        model = YOLO(args.weights)

    print(f"[YOLO] Training GPUs (physical): {gpu_ids}")
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr,
        imgsz=args.imgsz,
        project=args.output_dir,
        name="train",
        exist_ok=True,
        resume=use_resume,
        device=gpu_ids[0] if len(gpu_ids) == 1 else gpu_ids,
        verbose=True,
    )
    best = Path(args.output_dir) / "train" / "weights" / "best.pt"
    if best.exists():
        print(json.dumps({"best_checkpoint": str(best), "resumed": use_resume}, ensure_ascii=False))
    print(f"[YOLO] Done. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
