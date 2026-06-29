import asyncio
from pathlib import Path

from agent.graph import build_agent
from backend.app.services.job_store import JobStatus, job_store


def _run_agent_sync(job_id: str, prompt: str, project_path: Path, recursion_limit: int):
    def on_event(event_type: str, data: dict):
        job_store.add_event(job_id, event_type, data)

    agent = build_agent(project_root=project_path, on_event=on_event)
    return agent.invoke({"user_prompt": prompt}, {"recursion_limit": recursion_limit})


async def run_job(job_id: str, prompt: str, recursion_limit: int = 100):
    job = job_store.get(job_id)
    if not job:
        return

    project_path = Path(job.project_path)
    job_store.update_status(job_id, JobStatus.RUNNING)
    job_store.add_event(job_id, "status", {"phase": "started", "message": "Job started"})

    try:
        await asyncio.to_thread(
            _run_agent_sync, job_id, prompt, project_path, recursion_limit
        )
        job_store.update_status(job_id, JobStatus.COMPLETED)
        job_store.add_event(job_id, "completed", {"message": "Project generated successfully"})
    except Exception as exc:
        job_store.update_status(job_id, JobStatus.FAILED, str(exc))
        job_store.add_event(job_id, "error", {"message": str(exc)})
