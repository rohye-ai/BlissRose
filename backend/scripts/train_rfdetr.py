"""Standalone RF-DETR training script invoked by TrainingWorker."""

from __future__ import annotations

import argparse
import json
import sys


MODEL_MAP = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on a custom dataset")
    parser.add_argument("--model-size", default="medium", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated physical GPU indices, e.g. 0,1")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [0]

    import rfdetr

    class_name = MODEL_MAP[args.model_size]
    ModelClass = getattr(rfdetr, class_name)
    kwargs = {}
    if args.resume_checkpoint:
        kwargs["pretrain_weights"] = args.resume_checkpoint

    print(f"[RF-DETR] Loading {class_name} ...")
    print(f"[RF-DETR] Training GPUs (physical): {gpu_ids}")
    model = ModelClass(**kwargs)
    print(f"[RF-DETR] Training on {args.dataset_dir}")

    train_kwargs: dict = {
        "dataset_dir": args.dataset_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "output_dir": args.output_dir,
    }

    # CUDA_VISIBLE_DEVICES is set by TrainingWorker; use devices=int, not device="cuda:N".
    # Passing device="cuda:0" makes rfdetr emit devices=[0] (a list), which breaks build_trainer.
    train_kwargs["accelerator"] = "gpu"
    if len(gpu_ids) == 1:
        train_kwargs["devices"] = 1
        print(f"[RF-DETR] Single-GPU mode -> devices=1 (physical GPU {gpu_ids[0]})")
    else:
        train_kwargs["devices"] = len(gpu_ids)
        train_kwargs["strategy"] = "ddp"
        print(f"[RF-DETR] Multi-GPU DDP mode -> {len(gpu_ids)} devices")

    model.train(**train_kwargs)
    print(f"[RF-DETR] Done. Checkpoints saved to {args.output_dir}")
    print(json.dumps({"gpu_ids": gpu_ids, "devices": len(gpu_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
