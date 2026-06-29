import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from agent.tools import list_project_files
from backend.app.services.agent_runner import run_job
from backend.app.services.job_store import JobStatus, build_project_zip, job_store
from backend.app.services import preview as preview_service

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    prompt: str = Field(min_length=1)
    recursion_limit: int = Field(default=100, ge=10, le=500)


class JobResponse(BaseModel):
    id: str
    prompt: str
    status: JobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    plan: dict | None = None
    preview_url: str | None = None
    file_count: int = 0


def _job_to_response(job) -> JobResponse:
    project_path = Path(job.project_path)
    return JobResponse(
        id=job.id,
        prompt=job.prompt,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        plan=job.plan,
        preview_url=job.preview_url,
        file_count=len(list_project_files(project_path)),
    )


@router.post("", response_model=JobResponse)
async def create_job(body: CreateJobRequest, background_tasks: BackgroundTasks):
    job = job_store.create(body.prompt)
    background_tasks.add_task(run_job, job.id, body.prompt, body.recursion_limit)
    return _job_to_response(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/{job_id}/events")
async def stream_events(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        for event in job.events:
            yield f"data: {json.dumps({'type': event.type, 'data': event.data, 'timestamp': event.timestamp})}\n\n"

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return

        queue = await job_store.subscribe(job_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps({'type': event.type, 'data': event.data, 'timestamp': event.timestamp})}\n\n"
                    current = job_store.get(job_id)
                    if current and current.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            job_store.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/{job_id}/files")
async def list_files(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    files = list_project_files(Path(job.project_path))
    return {"files": files}


@router.get("/{job_id}/files/content")
async def get_file_content(job_id: str, path: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    project_path = Path(job.project_path).resolve()
    file_path = (project_path / path).resolve()
    if project_path not in file_path.parents and file_path != project_path:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return {"path": path, "content": file_path.read_text(encoding="utf-8")}


@router.get("/{job_id}/download")
async def download_project(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    project_path = Path(job.project_path)
    if not project_path.exists() or not any(project_path.rglob("*")):
        raise HTTPException(status_code=404, detail="No generated files yet")

    zip_bytes = build_project_zip(project_path)
    filename = f"project-{job_id[:8]}.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{job_id}/preview/start")
async def start_preview(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job must be completed before preview")

    try:
        result = preview_service.start_preview(job_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{job_id}/preview")
async def get_preview(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        return preview_service.get_preview_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{job_id}/preview/stop")
async def stop_preview(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    preview_service.stop_preview(job_id)
    return {"stopped": True}
