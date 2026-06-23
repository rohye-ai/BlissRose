from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit import apply_update_audit
from .config import ROOT_DIR, config_store
from .database import SessionLocal
from .db_models import DatasetRecord, ModelRecord, TrainingJobRecord
from .model_manager import infer_model_size
from .schemas import TrainState


class TrainingWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._state = TrainState.IDLE
        self._message = "未开始训练"
        self._progress: dict[str, Any] = {}
        self._log_path: Path | None = None
        self._current_job_id: str | None = None

    @property
    def state(self) -> TrainState:
        return self._state

    @property
    def message(self) -> str:
        return self._message

    @property
    def progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._progress)

    @property
    def current_job_id(self) -> str | None:
        return self._current_job_id

    def start(self, job_id: str | None = None) -> None:
        with self._lock:
            if self._state == TrainState.RUNNING:
                raise RuntimeError("训练任务已在运行")
            if job_id:
                self._start_job(job_id)
            else:
                self._start_legacy()

    def _start_legacy(self) -> None:
        cfg = config_store.get().training
        if not cfg.dataset_dir:
            raise ValueError("请配置训练数据集路径 dataset_dir")
        dataset = Path(cfg.dataset_dir)
        if not dataset.is_absolute():
            dataset = ROOT_DIR / dataset
        if not dataset.exists():
            raise ValueError(f"数据集目录不存在: {dataset}")

        output = Path(cfg.output_dir)
        if not output.is_absolute():
            output = ROOT_DIR / output
        output.mkdir(parents=True, exist_ok=True)

        model_cfg = config_store.get().model
        self._state = TrainState.RUNNING
        self._message = "训练已启动"
        self._progress = {"epoch": 0, "total_epochs": cfg.epochs, "log_tail": [], "job_id": None}
        self._log_path = output / "train.log"
        self._current_job_id = None
        self._thread = threading.Thread(
            target=self._run_training,
            kwargs={
                "model_size": infer_model_size(model_cfg.checkpoint or ""),
                "checkpoint": model_cfg.checkpoint,
                "dataset": dataset,
                "output": output,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "grad_accum_steps": cfg.grad_accum_steps,
                "lr": cfg.lr,
                "gpu_ids": list(cfg.gpu_ids),
            },
            daemon=True,
        )
        self._thread.start()

    def _start_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if not job:
                raise ValueError(f"训练任务不存在: {job_id}")
            if job.state == "running":
                raise RuntimeError("该训练任务已在运行")
            model = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
            dataset = db.query(DatasetRecord).filter(DatasetRecord.id == job.dataset_id).first()
            if not model or not dataset:
                raise ValueError("训练任务关联的模型或数据集不存在")

            output = ROOT_DIR / job.output_dir
            output.mkdir(parents=True, exist_ok=True)
            dataset_path = ROOT_DIR / dataset.path
            gpu_ids = json.loads(job.gpu_ids or "[0]")

            job.state = "running"
            job.message = "训练进行中"
            audit = apply_update_audit("system")
            job.updated_at = audit["updated_at"]
            job.updated_by = audit["updated_by"]
            db.commit()

            self._state = TrainState.RUNNING
            self._message = f"训练任务 {job.name} 已启动"
            self._progress = {
                "epoch": 0,
                "total_epochs": job.epochs,
                "log_tail": [],
                "job_id": job_id,
            }
            self._log_path = output / "train.log"
            self._current_job_id = job_id
            self._thread = threading.Thread(
                target=self._run_training,
                kwargs={
                    "model_size": infer_model_size(model.file_path or model.name),
                    "checkpoint": model.file_path,
                    "dataset": dataset_path,
                    "output": output,
                    "epochs": job.epochs,
                    "batch_size": job.batch_size,
                    "grad_accum_steps": job.grad_accum_steps,
                    "lr": job.lr,
                    "gpu_ids": gpu_ids,
                    "job_id": job_id,
                },
                daemon=True,
            )
            self._thread.start()
        finally:
            db.close()

    def stop(self, job_id: str | None = None) -> None:
        with self._lock:
            if self._state != TrainState.RUNNING:
                return
            if job_id and self._current_job_id != job_id:
                raise RuntimeError("该训练任务未在运行")
            self._state = TrainState.STOPPING
            self._message = "正在停止训练..."
            proc = self._process
            current_job = self._current_job_id
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if current_job:
            db = SessionLocal()
            try:
                job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == current_job).first()
                if job and job.state == "running":
                    job.state = "failed"
                    job.message = "训练已手动停止"
                    audit = apply_update_audit("system")
                    job.updated_at = audit["updated_at"]
                    job.updated_by = audit["updated_by"]
                    db.commit()
            finally:
                db.close()

    def _run_training(
        self,
        model_size: str,
        checkpoint: str,
        dataset: Path,
        output: Path,
        epochs: int,
        batch_size: int,
        grad_accum_steps: int,
        lr: float,
        gpu_ids: list[int],
        job_id: str | None = None,
    ) -> None:
        script = ROOT_DIR / "backend" / "scripts" / "train_rfdetr.py"
        cmd = [
            sys.executable,
            str(script),
            "--model-size",
            model_size,
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--grad-accum-steps",
            str(grad_accum_steps),
            "--lr",
            str(lr),
        ]
        if checkpoint:
            ckpt = ROOT_DIR / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
            if ckpt.exists():
                cmd.extend(["--resume-checkpoint", str(ckpt)])

        gpu_ids = gpu_ids or [0]
        cmd.extend(["--gpu-ids", ",".join(str(g) for g in gpu_ids)])

        log_path = self._log_path or (output / "train.log")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        final_code = 1
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(ROOT_DIR),
                    env=env,
                )
                with self._lock:
                    self._process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self._parse_log_line(line.strip())
                final_code = proc.wait()
            with self._lock:
                self._process = None
                if self._state == TrainState.STOPPING:
                    self._state = TrainState.IDLE
                    self._message = "训练已停止"
                elif final_code == 0:
                    self._state = TrainState.COMPLETED
                    self._message = f"训练完成，输出目录: {output}"
                else:
                    self._state = TrainState.FAILED
                    self._message = f"训练失败，退出码 {final_code}，详见 {log_path}"

            if job_id:
                success = final_code == 0 and self._state != TrainState.STOPPING
                self._finalize_job(job_id, success, output)
        except Exception as exc:
            with self._lock:
                self._state = TrainState.FAILED
                self._message = f"训练异常: {exc}"
                self._process = None
            if job_id:
                self._finalize_job(job_id, False, output, str(exc))

    def _finalize_job(self, job_id: str, success: bool, output: Path, err: str = "") -> None:
        db = SessionLocal()
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if not job:
                return
            if success:
                job.state = "completed"
                job.message = "训练完成"
                for name in ("checkpoint_best_regular.pth", "checkpoint_best_total.pth"):
                    candidate = output / name
                    if candidate.exists():
                        try:
                            job.checkpoint_path = str(candidate.relative_to(ROOT_DIR)).replace("\\", "/")
                        except ValueError:
                            job.checkpoint_path = str(candidate)
                        break
            else:
                job.state = "failed"
                job.message = err or "训练失败"
            job.completed_at = datetime.utcnow()
            audit = apply_update_audit("system")
            job.updated_at = audit["updated_at"]
            job.updated_by = audit["updated_by"]
            db.commit()
        finally:
            db.close()
        with self._lock:
            self._current_job_id = None

    def _parse_log_line(self, line: str) -> None:
        if not line:
            return
        with self._lock:
            tail: list[str] = self._progress.get("log_tail", [])
            tail.append(line)
            self._progress["log_tail"] = tail[-50:]
            if "Epoch" in line:
                self._progress["last_line"] = line
            for token in line.split():
                if token.startswith("epoch="):
                    try:
                        self._progress["epoch"] = int(token.split("=")[1].split("/")[0])
                    except ValueError:
                        pass


training_worker = TrainingWorker()
