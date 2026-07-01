import json
import os
import subprocess
import time
from pathlib import Path

from backend.app.services.job_store import job_store


def _find_free_port(start: int = 5173, end: int = 5273) -> int:
    import socket

    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port available for preview")


def _has_npm_dev_script(project_path: Path) -> bool:
    package_json = project_path / "package.json"
    if not package_json.exists():
        return False
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"package.json is not valid JSON: {exc}") from exc
    scripts = data.get("scripts", {})
    return "dev" in scripts


def _safe_project_file(project_path: Path, relative_path: str) -> Path:
    root = project_path.resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Invalid file path: {relative_path}")
    return target


def materialize_files(project_path: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        target = _safe_project_file(project_path, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _build_preview_url(port: int) -> str:
    template = os.getenv("PREVIEW_URL_TEMPLATE")
    if template:
        return template.format(port=port)

    scheme = os.getenv("PREVIEW_SCHEME", "http")
    host = os.getenv("PREVIEW_HOST", "localhost")
    return f"{scheme}://{host}:{port}"


def start_preview(job_id: str, files: dict[str, str] | None = None) -> dict:
    job = job_store.get(job_id)
    if not job:
        raise ValueError("Job not found")

    project_path = Path(job.project_path)
    if files:
        materialize_files(project_path, files)

    if not _has_npm_dev_script(project_path):
        raise ValueError("No package.json with a 'dev' script found in generated project")

    if job.preview_container_id:
        stop_preview(job_id)

    port = _find_free_port()
    container_name = f"slothcode-preview-{job_id[:8]}"

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{port}:5173",
        "-v",
        f"{project_path.resolve()}:/app",
        "-w",
        "/app",
        os.getenv("PREVIEW_NODE_IMAGE", "node:20-alpine"),
        "sh",
        "-c",
        "npm install && npm run dev -- --host 0.0.0.0 --port 5173",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start preview container: {result.stderr.strip()}")

    container_id = result.stdout.strip()
    preview_url = _build_preview_url(port)

    job.preview_url = preview_url
    job.preview_port = port
    job.preview_container_id = container_id

    return {
        "preview_url": preview_url,
        "port": port,
        "container_id": container_id,
        "message": "Preview starting — it may take a minute for npm install to finish.",
    }


def get_preview_status(job_id: str) -> dict:
    job = job_store.get(job_id)
    if not job:
        raise ValueError("Job not found")

    if not job.preview_container_id:
        return {"running": False, "preview_url": None}

    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", job.preview_container_id],
        capture_output=True,
        text=True,
    )
    running = result.stdout.strip() == "true"
    return {
        "running": running,
        "preview_url": job.preview_url if running else None,
        "port": job.preview_port,
    }


def stop_preview(job_id: str):
    job = job_store.get(job_id)
    if not job or not job.preview_container_id:
        return

    subprocess.run(
        ["docker", "rm", "-f", job.preview_container_id],
        capture_output=True,
        text=True,
    )
    job.preview_container_id = None
    job.preview_url = None
    job.preview_port = None


def wait_for_preview_ready(job_id: str, timeout: int = 120) -> bool:
    status = get_preview_status(job_id)
    if not status.get("preview_url"):
        return False

    port = status.get("port")
    if not port:
        return False

    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(2)
    return False
