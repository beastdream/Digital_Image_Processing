from __future__ import annotations

import csv
import json
import mimetypes
import os
import shutil
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_LOG_CHARS = 5000
EXPERIMENT_NAMES = ("A_raw", "B_gaussian", "C_median", "D_clahe", "E_brightness", "F_combined")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json_file(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _suite_paths(suite: str) -> tuple[Path, Path, Path]:
    if suite == "video":
        return (
            ROOT / "experiments" / "yolo_video",
            ROOT / "results" / "video_experiments",
            ROOT / "results" / "evaluations" / "video",
        )
    if suite == "image":
        return (
            ROOT / "experiments" / "yolo",
            ROOT / "results" / "experiments",
            ROOT / "results" / "evaluations",
        )
    raise ValueError("suite must be image or video")


def _experiment(suite: str, name: str) -> Path:
    if name not in EXPERIMENT_NAMES:
        raise ValueError("Unknown experiment")
    project_dir, _, _ = _suite_paths(suite)
    path = project_dir / name
    if not (path / "weights" / "best.pt").is_file():
        raise FileNotFoundError(f"{name} has no weights/best.pt")
    if not (path / "experiment_config.yaml").is_file():
        raise FileNotFoundError(f"{name} has no experiment_config.yaml")
    return path


def _available_experiments(suite: str) -> list[dict]:
    project_dir, _, _ = _suite_paths(suite)
    items = []
    for name in EXPERIMENT_NAMES:
        run = project_dir / name
        checkpoint = run / "weights" / "best.pt"
        config = run / "experiment_config.yaml"
        if checkpoint.is_file() and config.is_file():
            status = _json_file(run / "training_status.json") or {}
            items.append({
                "name": name,
                "checkpoint": _relative(checkpoint),
                "status": status.get("status", "AVAILABLE"),
                "size_bytes": checkpoint.stat().st_size,
            })
    return items


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _experiment_results(suite: str) -> dict:
    _, results_dir, _ = _suite_paths(suite)
    table = _csv_rows(results_dir / "experiment_results.csv")
    analysis = _json_file(results_dir / "experiment_analysis.json")
    visual = _csv_rows(results_dir / "visual_inspection.csv")
    return {"suite": suite, "rows": table, "analysis": analysis, "visual_inspection": visual}


def _evaluation(suite: str, name: str) -> dict:
    run = _experiment(suite, name)
    _, _, evaluations_dir = _suite_paths(suite)
    evaluation_dir = evaluations_dir / name
    metrics_path = evaluation_dir / "metrics.json"
    output_roots = ROOT / "results" / "evaluations" / "roadscan" / suite / name
    if output_roots.is_dir():
        candidates = sorted(
            output_roots.glob("*/metrics.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            metrics_path = candidates[0]
            evaluation_dir = metrics_path.parent
    metrics = _json_file(metrics_path)
    if metrics is None:
        return {"experiment": name, "metrics": None, "confusion_analysis": None, "class_metrics": []}
    return {
        "experiment": name,
        "metrics": metrics,
        "confusion_analysis": _json_file(evaluation_dir / "confusion_analysis.json"),
        "class_metrics": _csv_rows(evaluation_dir / "class_metrics.csv"),
        "metrics_path": _relative(metrics_path),
    }


def _parse_byte_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """Parse a single "bytes=start-end" Range header; None means send the whole file."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not match or match.groups() == ("", ""):
        return None
    first, last = match.groups()
    if first:
        start, end = int(first), int(last) if last else size - 1
    else:
        start, end = max(0, size - int(last)), size - 1
    end = min(end, size - 1)
    if size == 0 or start > end:
        return "invalid"
    return start, end


def _multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, str, bytes]]]:
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n"
        b"MIME-Version: 1.0\r\n\r\n" + body
    )
    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data")
    fields: dict[str, str] = {}
    files: list[tuple[str, str, bytes]] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append((name, Path(filename).name, payload))
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return fields, files


def _int_field(fields: dict[str, str], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(fields.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _float_field(fields: dict[str, str], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(fields.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _new_job(job_id: str, command: list[str], upload_dir: Path, output_dir: Path, kind: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "output_dir": _relative(output_dir),
            "log": "",
            "error": None,
            "files": [],
            "summary": None,
        }

    def run() -> None:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            log = (completed.stdout + "\n" + completed.stderr).strip()
            with JOBS_LOCK:
                JOBS[job_id]["log"] = log[-MAX_LOG_CHARS:]
                if completed.returncode:
                    JOBS[job_id]["status"] = "failed"
                    JOBS[job_id]["error"] = f"Prediction exited with code {completed.returncode}"
                else:
                    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
                    JOBS[job_id]["files"] = [
                        {"name": path.name, "url": "/files/" + _relative(path)}
                        for path in output_files
                    ]
                    summary_name = {
                        "image": "prediction_summary.json",
                        "video": "video_summary.json",
                        "evaluation": "metrics.json",
                    }[kind]
                    JOBS[job_id]["summary"] = _json_file(output_dir / summary_name)
                    JOBS[job_id]["status"] = "completed"
        except OSError as exc:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = f"Could not run prediction: {exc}"
        finally:
            if upload_dir.exists():
                try:
                    shutil.rmtree(upload_dir)
                except OSError as exc:
                    with JOBS_LOCK:
                        JOBS[job_id]["status"] = "failed"
                        JOBS[job_id]["error"] = f"Could not remove temporary upload: {exc}"

    threading.Thread(target=run, name=f"roadscan-{job_id}", daemon=True).start()


class RoadScanHandler(SimpleHTTPRequestHandler):
    server_version = "RoadScan/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format_string: str, *args) -> None:
        print(f"[RoadScan] {self.address_string()} {format_string % args}")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _send_json(self, value, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, message: str, status: int) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            try:
                query = parse_qs(parsed.query)
                if parsed.path == "/api/experiments":
                    suite = query.get("suite", ["video"])[0]
                    _suite_paths(suite)
                    self._send_json({"experiments": _available_experiments(suite)})
                    return
                if parsed.path == "/api/results/experiments":
                    suite = query.get("suite", ["video"])[0]
                    self._send_json(_experiment_results(suite))
                    return
                if parsed.path == "/api/results/evaluation":
                    suite = query.get("suite", ["video"])[0]
                    name = query.get("experiment", ["A_raw"])[0]
                    self._send_json(_evaluation(suite, name))
                    return
                if parsed.path == "/api/results/class-balance":
                    result = _json_file(ROOT / "results" / "analysis" / "class_balance" /
                                        "class_imbalance_and_bbox_analysis.json")
                    self._send_json({"available": result is not None, "report": result})
                    return
                match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", parsed.path)
                if match:
                    with JOBS_LOCK:
                        job = JOBS.get(match.group(1))
                    if job is None:
                        self._send_error_json("Job not found", 404)
                    else:
                        self._send_json(job)
                    return
                self._send_error_json("Unknown API endpoint", 404)
                return
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                self._send_error_json(str(exc), 400)
                return
        if parsed.path.startswith("/files/"):
            self._serve_result_file(unquote(parsed.path[len("/files/"):]))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        elif parsed.path not in {"/index.html", "/styles.css", "/app.js"}:
            self._send_error_json("File not found", 404)
            return
        super().do_GET()

    def _serve_result_file(self, relative: str) -> None:
        requested = (ROOT / relative).resolve()
        try:
            requested.relative_to((ROOT / "results").resolve())
        except ValueError:
            self._send_error_json("Only files below results/ can be served", 403)
            return
        if not requested.is_file():
            self._send_error_json("Result file not found", 404)
            return
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        size = requested.stat().st_size
        # Browsers request videos in byte ranges (to read the MP4 index and to seek).
        start, end = 0, size - 1
        byte_range = _parse_byte_range(self.headers.get("Range"), size)
        if byte_range == "invalid":
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        if byte_range:
            start, end = byte_range
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with requested.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0 and (chunk := handle.read(min(1024 * 1024, remaining))):
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # the browser cancelled this range (normal while seeking)
                remaining -= len(chunk)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if not origin or urlparse(origin).netloc.lower() != host.lower():
            self._send_error_json("Cross-origin requests are not allowed", 403)
            return
        if parsed.path == "/api/predict/image" or parsed.path == "/api/predict/video":
            self._start_prediction(parsed.path)
            return
        if parsed.path == "/api/evaluate":
            self._start_evaluation()
            return
        self._send_error_json("Unknown API endpoint", 404)

    def _request_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("Request body is empty")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("Upload is larger than the 1 GiB limit")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Request body ended before the declared Content-Length")
        return body

    def _start_prediction(self, endpoint: str) -> None:
        job_id = uuid.uuid4().hex
        upload_dir = ROOT / "results" / "predictions" / ".roadscan_uploads" / job_id
        output_dir = ROOT / "results" / "predictions" / "roadscan" / job_id
        try:
            fields, files = _multipart(self.headers.get("Content-Type", ""), self._request_body())
            kind = "image" if endpoint.endswith("image") else "video"
            video_source_mode = fields.get("source_mode", "video")
            suite = fields.get("suite", "video")
            experiment_name = fields.get("experiment", "A_raw")
            experiment_dir = _experiment(suite, experiment_name)
            webcam = kind == "video" and video_source_mode == "webcam"
            if webcam and files:
                raise ValueError("Webcam mode does not accept an uploaded video")
            if kind == "video" and video_source_mode not in {"video", "webcam"}:
                raise ValueError("source_mode must be video or webcam")
            if not webcam and not files:
                raise ValueError("Choose at least one file")
            if not webcam:
                suffixes = IMAGE_SUFFIXES if kind == "image" else VIDEO_SUFFIXES
                if any(Path(name).suffix.lower() not in suffixes for _, name, _ in files):
                    expected = ", ".join(sorted(suffixes))
                    raise ValueError(f"Unsupported file type. Accepted: {expected}")
            if kind == "video" and not webcam and len(files) != 1:
                raise ValueError("Choose one video file")
            upload_dir.mkdir(parents=True, exist_ok=False)
            input_dir = upload_dir / "input"
            input_dir.mkdir()
            for index, (_, filename, content) in enumerate(files):
                safe_name = f"{index:04d}_{Path(filename).name}"
                target = input_dir / safe_name
                if not content or not target.resolve().is_relative_to(upload_dir.resolve()):
                    raise ValueError("Uploaded file is empty or invalid")
                target.write_bytes(content)

            conf = _float_field(fields, "conf", 0.25, 0.0, 1.0)
            command = [
                sys.executable, str(ROOT / "predict.py"),
                "--experiment", str(experiment_dir),
                "--conf", str(conf),
                "--output", str(output_dir),
            ]
            if kind == "image":
                command.extend(["--source", str(input_dir)])
                command.extend(["--max-images", str(_int_field(fields, "max_images", 20, 1, 500))])
                if fields.get("show_ground_truth") == "true":
                    command.append("--show-ground-truth")
                if fields.get("side_by_side") == "true":
                    if fields.get("show_ground_truth") != "true":
                        raise ValueError("Side-by-side requires ground-truth comparison")
                    command.append("--side-by-side")
            else:
                command.extend([
                    "--video",
                    str(_int_field(fields, "camera_index", 0, 0, 16)) if webcam else str(next(input_dir.iterdir())),
                    "--frame-step", str(_int_field(fields, "frame_step", 1, 1, 1000)),
                    "--frame-size", str(_int_field(fields, "frame_size", 640, 0, 4096)),
                    "--min-track-frames", str(_int_field(fields, "min_track_frames", 3, 1, 1000)),
                ])
                if webcam:
                    command.extend(["--max-frames", str(_int_field(fields, "max_frames", 300, 1, 1000000))])
                if fields.get("save_frame_detections") == "true":
                    command.append("--save-frame-detections")
            _new_job(job_id, command, upload_dir, output_dir, kind)
            self._send_json({"job_id": job_id, "status": "queued"}, 202)
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            if upload_dir.exists():
                for item in upload_dir.rglob("*"):
                    if item.is_file():
                        item.unlink()
                for item in sorted(upload_dir.rglob("*"), key=lambda path: len(path.parts), reverse=True):
                    if item.is_dir():
                        item.rmdir()
                upload_dir.rmdir()
            self._send_error_json(str(exc), 400)
        except OSError as exc:
            self._send_error_json(f"Could not save uploaded file: {exc}", 500)

    def _start_evaluation(self) -> None:
        try:
            body = json.loads(self._request_body().decode("utf-8"))
            suite = body.get("suite", "video")
            name = body.get("experiment", "A_raw")
            experiment_dir = _experiment(suite, name)
            dataset = ROOT / "data" / "processed" / (
                "pothole_video_frames" if suite == "video" else "road_damage_detection"
            )
            dataset_yaml = dataset / "dataset.yaml"
            if not dataset_yaml.is_file():
                raise FileNotFoundError(f"Dataset is missing: {dataset_yaml}")
            output_dir = ROOT / "results" / "evaluations" / "roadscan" / suite / name / uuid.uuid4().hex
            job_id = uuid.uuid4().hex
            command = [
                sys.executable, "-m", "src.detection.evaluate",
                "--model", str(experiment_dir / "weights" / "best.pt"),
                "--experiment-config", str(experiment_dir / "experiment_config.yaml"),
                "--data", str(dataset_yaml),
                "--split", "test",
                "--output", str(output_dir),
            ]
            upload_dir = ROOT / "results" / "evaluations" / "roadscan" / ".empty" / job_id
            upload_dir.mkdir(parents=True, exist_ok=False)
            _new_job(job_id, command, upload_dir, output_dir, "evaluation")
            self._send_json({"job_id": job_id, "status": "queued"}, 202)
        except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as exc:
            self._send_error_json(str(exc), 400)


def main() -> None:
    host = os.environ.get("ROADSCAN_HOST", "127.0.0.1")
    port = int(os.environ.get("ROADSCAN_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), RoadScanHandler)
    print(f"RoadScan is ready: http://{host}:{port}")
    print("Keep this terminal open while using the interface. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RoadScan...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
