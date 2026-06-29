import asyncio
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobEvent(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Job(BaseModel):
    id: str
    prompt: str
    status: JobStatus = JobStatus.PENDING
    project_path: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None
    plan: dict | None = None
    events: list[JobEvent] = Field(default_factory=list)
    preview_url: str | None = None
    preview_port: int | None = None
    preview_container_id: str | None = None


PROJECTS_DIR = Path(os.getenv("PROJECTS_DIR", Path.cwd() / "projects"))


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def create(self, prompt: str) -> Job:
        job_id = str(uuid.uuid4())
        project_path = PROJECTS_DIR / job_id
        project_path.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, prompt=prompt, project_path=str(project_path.resolve()))
        self._jobs[job_id] = job
        self._subscribers[job_id] = []
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None):
        job = self._jobs[job_id]
        job.status = status
        job.updated_at = datetime.now(timezone.utc).isoformat()
        if error:
            job.error = error

    def add_event(self, job_id: str, event_type: str, data: dict | None = None):
        job = self._jobs[job_id]
        event = JobEvent(type=event_type, data=data or {})
        job.events.append(event)
        job.updated_at = datetime.now(timezone.utc).isoformat()
        if event_type == "plan":
            job.plan = data
        for queue in self._subscribers.get(job_id, []):
            queue.put_nowait(event)

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        subs = self._subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)


job_store = JobStore()


def build_project_zip(project_path: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(project_path).as_posix()
                zf.write(file_path, arcname)
    return buffer.getvalue()
