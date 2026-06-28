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
from .training_checkpoints import find_resume_checkpoint, rel_checkpoint_path
from .yolo_dataset import prepare_yolo_data_yaml


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

    def start(self, job_id: str | None = None, resume: bool = True) -> None:
        with self._lock:
            if job_id:
                if self._state == TrainState.RUNNING:
                    self._enqueue_job(job_id)
                    return
                self._start_job(job_id, resume=resume)
            else:
                if self._state == TrainState.RUNNING:
                    raise RuntimeError("训练任务已在运行")
                self._start_legacy()

    def _enqueue_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if not job:
                raise ValueError(f"训练任务不存在: {job_id}")
            if job.state in ("running", "completed"):
                raise RuntimeError(f"任务状态为 {job.state}，无法加入队列")
            if job.state != "queued":
                job.state = "queued"
                job.message = "已加入训练队列，等待执行"
                audit = apply_update_audit("system")
                job.updated_at = audit["updated_at"]
                job.updated_by = audit["updated_by"]
                db.commit()
            queued = db.query(TrainingJobRecord).filter(TrainingJobRecord.state == "queued").count()
            self._message = f"任务 {job.name} 已加入队列（排队 {queued} 个）"
        finally:
            db.close()

    def _next_queued_job_id(self) -> str | None:
        db = SessionLocal()
        try:
            job = (
                db.query(TrainingJobRecord)
                .filter(TrainingJobRecord.state.in_(["queued", "pending"]))
                .order_by(TrainingJobRecord.created_at.asc())
                .first()
            )
            return job.id if job else None
        finally:
            db.close()

    def _try_start_next(self) -> None:
        with self._lock:
            if self._state == TrainState.RUNNING:
                return
        next_id = self._next_queued_job_id()
        if not next_id:
            with self._lock:
                if self._state not in (TrainState.RUNNING, TrainState.STOPPING):
                    self._state = TrainState.IDLE
                    self._message = "队列空闲"
            return
        try:
            with self._lock:
                self._start_job(next_id)
        except Exception as exc:
            with self._lock:
                self._message = f"启动队列任务失败: {exc}"

    def list_queue(self) -> dict[str, Any]:
        db = SessionLocal()
        try:
            rows = (
                db.query(TrainingJobRecord)
                .filter(TrainingJobRecord.state.in_(["queued", "pending", "running"]))
                .order_by(TrainingJobRecord.created_at.asc())
                .all()
            )
            return {
                "running_job_id": self._current_job_id,
                "worker_state": self._state.value,
                "items": [
                    {"id": r.id, "name": r.name, "state": r.state, "created_at": r.created_at.isoformat()}
                    for r in rows
                ],
            }
        finally:
            db.close()

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
                "model_type": "rf-detr",
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

    def _start_job(self, job_id: str, resume: bool = True) -> None:
        db = SessionLocal()
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if not job:
                raise ValueError(f"训练任务不存在: {job_id}")
            if job.state == "running":
                raise RuntimeError("该训练任务已在运行")
            if job.state == "queued":
                pass
            model = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
            dataset = db.query(DatasetRecord).filter(DatasetRecord.id == job.dataset_id).first()
            if not model or not dataset:
                raise ValueError("训练任务关联的模型或数据集不存在")

            output = ROOT_DIR / job.output_dir
            output.mkdir(parents=True, exist_ok=True)
            dataset_path = ROOT_DIR / dataset.path
            gpu_ids = json.loads(job.gpu_ids or "[0]")

            resume_ckpt = find_resume_checkpoint(output, model.model_type) if resume else None
            if resume and resume_ckpt:
                job.message = f"从断点继续训练: {resume_ckpt.name}"
            elif resume:
                job.message = "训练进行中"
            else:
                job.message = "从头开始训练"

            job.state = "running"
            audit = apply_update_audit("system")
            job.updated_at = audit["updated_at"]
            job.updated_by = audit["updated_by"]
            db.commit()

            self._state = TrainState.RUNNING
            self._message = f"训练任务 {job.name} 已启动" + ("（断点续训）" if resume_ckpt else "")
            self._progress = {
                "epoch": 0,
                "total_epochs": job.epochs,
                "log_tail": [],
                "job_id": job_id,
                "resuming": bool(resume_ckpt),
            }
            self._log_path = output / "train.log"
            self._current_job_id = job_id
            self._thread = threading.Thread(
                target=self._run_training,
                kwargs={
                    "model_type": model.model_type,
                    "model_size": infer_model_size(model.file_path or model.name),
                    "checkpoint": model.file_path,
                    "dataset": dataset_path,
                    "data_yaml": dataset.data_yaml,
                    "output": output,
                    "epochs": job.epochs,
                    "batch_size": job.batch_size,
                    "grad_accum_steps": job.grad_accum_steps,
                    "lr": job.lr,
                    "gpu_ids": gpu_ids,
                    "job_id": job_id,
                    "resume_checkpoint": str(resume_ckpt) if resume_ckpt else "",
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
                    model = db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
                    output = ROOT_DIR / job.output_dir
                    ckpt = find_resume_checkpoint(output, model.model_type if model else "rf-detr")
                    if ckpt:
                        job.checkpoint_path = rel_checkpoint_path(ckpt)
                    audit = apply_update_audit("system")
                    job.updated_at = audit["updated_at"]
                    job.updated_by = audit["updated_by"]
                    db.commit()
            finally:
                db.close()

    def _run_training(
        self,
        model_type: str,
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
        data_yaml: str = "",
        resume_checkpoint: str = "",
    ) -> None:
        gpu_ids = gpu_ids or [0]
        if model_type == "yolo":
            cmd = self._build_yolo_cmd(
                checkpoint, dataset, data_yaml, output, epochs, batch_size, lr, gpu_ids, bool(resume_checkpoint)
            )
        else:
            cmd = self._build_rfdetr_cmd(
                model_size,
                checkpoint,
                dataset,
                output,
                epochs,
                batch_size,
                grad_accum_steps,
                lr,
                gpu_ids,
                resume_checkpoint,
            )

        log_path = self._log_path or (output / "train.log")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        final_code = 1
        log_mode = "a" if resume_checkpoint and log_path.is_file() else "w"
        try:
            with log_path.open(log_mode, encoding="utf-8") as log_file:
                if log_mode == "a":
                    log_file.write(f"\n{'=' * 60}\n[BlissRose] 断点续训 {datetime.utcnow().isoformat()}Z\n{'=' * 60}\n")
                    log_file.flush()
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
                self._finalize_job(job_id, success, output, err="", model_type=model_type)
        except Exception as exc:
            with self._lock:
                self._state = TrainState.FAILED
                self._message = f"训练异常: {exc}"
                self._process = None
            if job_id:
                self._finalize_job(job_id, False, output, str(exc), model_type=model_type)

    def _pick_best_checkpoint(self, output: Path, model_type: str) -> Path | None:
        if model_type == "yolo":
            for candidate in (
                output / "train" / "weights" / "best.pt",
                output / "weights" / "best.pt",
            ):
                if candidate.is_file():
                    return candidate
            return find_resume_checkpoint(output, model_type)
        for candidate in (
            output / "checkpoint_best_total.pth",
            output / "checkpoint_best_regular.pth",
        ):
            if candidate.is_file():
                return candidate
        return find_resume_checkpoint(output, model_type)

    def _finalize_job(
        self, job_id: str, success: bool, output: Path, err: str = "", model_type: str = "rf-detr"
    ) -> None:
        db = SessionLocal()
        try:
            job = db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()
            if not job:
                return
            metrics = self._collect_training_metrics(output)
            if metrics:
                job.metrics_json = json.dumps(metrics, ensure_ascii=False)
            if success:
                job.state = "completed"
                job.message = "训练完成"
            else:
                job.state = "failed"
                job.message = err or "训练失败"
            ckpt = self._pick_best_checkpoint(output, model_type)
            if ckpt:
                job.checkpoint_path = rel_checkpoint_path(ckpt)
            if not success and ckpt and not err:
                job.message = f"{job.message}（可断点续训: {ckpt.name}）"
            job.completed_at = datetime.utcnow()
            audit = apply_update_audit("system")
            job.updated_at = audit["updated_at"]
            job.updated_by = audit["updated_by"]
            db.commit()
        finally:
            db.close()
        with self._lock:
            self._current_job_id = None
        self._try_start_next()

    def _build_rfdetr_cmd(
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
        resume_checkpoint: str = "",
    ) -> list[str]:
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
            "--gpu-ids",
            ",".join(str(g) for g in gpu_ids),
        ]
        if resume_checkpoint:
            ckpt = Path(resume_checkpoint)
            if ckpt.is_file():
                cmd.extend(["--resume", str(ckpt)])
                return cmd
        if checkpoint:
            ckpt = ROOT_DIR / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
            if ckpt.exists():
                cmd.extend(["--pretrain-weights", str(ckpt)])
        return cmd

    def _build_yolo_cmd(
        self,
        checkpoint: str,
        dataset: Path,
        data_yaml: str,
        output: Path,
        epochs: int,
        batch_size: int,
        lr: float,
        gpu_ids: list[int],
        resume: bool = False,
    ) -> list[str]:
        script = ROOT_DIR / "backend" / "scripts" / "train_yolo.py"
        yaml_path = ROOT_DIR / data_yaml if data_yaml else dataset / "data.yaml"
        if not yaml_path.is_file():
            yaml_path = dataset / "data.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml 不存在: {yaml_path}")
        prepared_yaml = prepare_yolo_data_yaml(yaml_path)
        ckpt = ROOT_DIR / checkpoint if checkpoint and not Path(checkpoint).is_absolute() else Path(checkpoint or "")
        if not ckpt.exists():
            raise FileNotFoundError(f"YOLO 权重不存在: {checkpoint}")
        cmd = [
            sys.executable,
            str(script),
            "--weights",
            str(ckpt),
            "--data-yaml",
            str(prepared_yaml),
            "--output-dir",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--gpu-ids",
            ",".join(str(g) for g in gpu_ids),
        ]
        if resume:
            cmd.append("--resume")
        return cmd

    def _collect_training_metrics(self, output: Path) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metrics_csv = output / "metrics.csv"
        if metrics_csv.is_file():
            try:
                lines = metrics_csv.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(",")]
                    last = [v.strip() for v in lines[-1].split(",")]
                    for key, val in zip(headers, last):
                        if key and val:
                            try:
                                metrics[key] = float(val)
                            except ValueError:
                                metrics[key] = val
            except Exception:
                pass
        log_path = output / "train.log"
        if log_path.is_file():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                for line in reversed(text.splitlines()):
                    if "mAP" in line or "map" in line.lower():
                        metrics["last_eval_line"] = line.strip()[:500]
                        break
            except Exception:
                pass
        return metrics

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
