import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  cancelJob,
  getDownloadUrl,
  getJob,
  listFiles,
  subscribeToJobEvents,
} from "../api/client";
import FileExplorer from "../components/FileExplorer";
import JobProgress from "../components/JobProgress";
import PreviewPanel from "../components/PreviewPanel";
import { useProjectStore } from "../store/useProjectStore";
import type { Job, JobEvent } from "../types";

const EMPTY_EVENTS: JobEvent[] = [];
const EMPTY_FILES: Record<string, string> = {};
const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

function appendEvent(events: JobEvent[], event: JobEvent): JobEvent[] {
  const alreadyExists = events.some(
    (item) => item.timestamp === event.timestamp && item.type === event.type,
  );
  return alreadyExists ? events : [...events, event];
}

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const cachedJob = useProjectStore((state) =>
    jobId ? state.jobsById[jobId] : undefined,
  );
  const cachedEvents =
    useProjectStore((state) => (jobId ? state.eventsByJob[jobId] : undefined)) ??
    EMPTY_EVENTS;
  const cachedFiles =
    useProjectStore((state) => (jobId ? state.filesByJob[jobId] : undefined)) ??
    EMPTY_FILES;
  const setCachedJob = useProjectStore((state) => state.setJob);
  const updateCachedJob = useProjectStore((state) => state.updateJob);
  const addCachedEvent = useProjectStore((state) => state.addEvent);
  const setCachedFile = useProjectStore((state) => state.setFile);
  const [job, setJob] = useState<Job | null>(cachedJob ?? null);
  const [events, setEvents] = useState<JobEvent[]>(cachedEvents);
  const [files, setFiles] = useState<string[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    setJob(cachedJob ?? null);
    setEvents(cachedEvents);
    setError(null);
  }, [cachedEvents, cachedJob, jobId]);

  const refreshFiles = useCallback(async () => {
    if (!jobId) return;
    try {
      const fileList = await listFiles(jobId);
      const merged = Array.from(new Set([...fileList, ...Object.keys(cachedFiles)])).sort();
      setFiles(merged);
      if (merged.length > 0 && !selectedPath) {
        setSelectedPath(merged[0]);
      }
    } catch {
      const cached = Object.keys(cachedFiles).sort();
      setFiles(cached);
      if (cached.length > 0 && !selectedPath) {
        setSelectedPath(cached[0]);
      }
    }
  }, [cachedFiles, jobId, selectedPath]);

  async function handleCancel() {
    if (!jobId || !job || TERMINAL_STATUSES.has(job.status)) return;

    setCancelling(true);
    try {
      const updated = await cancelJob(jobId);
      setJob(updated);
      setCachedJob(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel job");
    } finally {
      setCancelling(false);
    }
  }

  useEffect(() => {
    if (!jobId) return;
    if (job && TERMINAL_STATUSES.has(job.status)) {
      refreshFiles();
      return;
    }

    let cancelled = false;

    getJob(jobId)
      .then((data) => {
        if (!cancelled) {
          setJob(data);
          setCachedJob(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Job not found");
        }
      });

    const unsubscribe = subscribeToJobEvents(
      jobId,
      (event) => {
        setEvents((prev) => appendEvent(prev, event));
        addCachedEvent(jobId, event);
        if (event.type === "plan") {
          updateCachedJob(jobId, { plan: event.data });
          setJob((prev) => (prev ? { ...prev, plan: event.data } : prev));
        }
        if (
          event.type === "file_written" &&
          typeof event.data.filepath === "string" &&
          typeof event.data.content === "string"
        ) {
          setCachedFile(jobId, event.data.filepath, event.data.content);
        }
        if (event.type === "file_written" || event.type === "completed") {
          refreshFiles();
        }
        if (event.type === "completed" || event.type === "error" || event.type === "cancelled") {
          getJob(jobId)
            .then((data) => {
              if (!cancelled) {
                setJob(data);
                setCachedJob(data);
              }
            })
            .catch(() => {
              if (event.type === "completed") {
                updateCachedJob(jobId, { status: "completed" });
                setJob((prev) => (prev ? { ...prev, status: "completed" } : prev));
              }
              if (event.type === "cancelled") {
                updateCachedJob(jobId, {
                  status: "cancelled",
                  error: "Job cancelled by user",
                });
                setJob((prev) =>
                  prev
                    ? { ...prev, status: "cancelled", error: "Job cancelled by user" }
                    : prev,
                );
              }
              if (event.type === "error" && typeof event.data.message === "string") {
                const message = event.data.message;
                updateCachedJob(jobId, { status: "failed", error: message });
                setJob((prev) =>
                  prev ? { ...prev, status: "failed", error: message } : prev,
                );
              }
            });
        }
      },
      () => {
        getJob(jobId)
          .then((data) => {
            if (!cancelled) {
              setJob(data);
              setCachedJob(data);
            }
          })
          .catch(() => {});
      },
    );

    const poll = setInterval(() => {
      getJob(jobId)
        .then((data) => {
          if (!cancelled) {
            setJob(data);
            setCachedJob(data);
          }
        })
        .catch(() => {});
      refreshFiles();
    }, 5000);

    refreshFiles();

    return () => {
      cancelled = true;
      unsubscribe();
      clearInterval(poll);
    };
  }, [
    addCachedEvent,
    job?.status,
    jobId,
    refreshFiles,
    setCachedFile,
    setCachedJob,
    updateCachedJob,
  ]);

  if (error) {
    return (
      <div className="page error-page">
        <p className="error-text">{error}</p>
        <Link to="/" className="btn secondary">
          Back home
        </Link>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="page loading-page">
        <div className="spinner" />
        <p>Loading job...</p>
      </div>
    );
  }

  const isDone = TERMINAL_STATUSES.has(job.status);
  const canCancel = job.status === "pending" || job.status === "running";
  const title =
    job.plan?.name
      ? String(job.plan.name)
      : job.status === "failed"
        ? "Generation failed"
        : job.status === "cancelled"
          ? "Generation cancelled"
          : "Generating project";
  const displayEvents =
    events.length > 0 || !job.error
      ? events
      : [
          {
            type: "error",
            data: { message: job.error },
            timestamp: job.updated_at,
          },
        ];

  return (
    <div className="page job-page">
      <header className="job-header">
        <div>
          <Link to="/" className="back-link">
            ← New project
          </Link>
          <h1>{title}</h1>
          <p className="job-prompt">{job.prompt}</p>
        </div>
        <div className="job-actions">
          {canCancel && (
            <button
              type="button"
              className="btn secondary"
              onClick={handleCancel}
              disabled={cancelling}
            >
              {cancelling ? "Stopping..." : "Stop job"}
            </button>
          )}
          {isDone && (job.file_count > 0 || files.length > 0) && (
            <a href={getDownloadUrl(job.id)} className="btn secondary" download>
              Download ZIP
            </a>
          )}
        </div>
      </header>

      {job.error && <div className="error-banner">{job.error}</div>}

      <div className="job-grid">
        <JobProgress events={displayEvents} status={job.status} />
        <FileExplorer
          jobId={job.id}
          files={files}
          selectedPath={selectedPath}
          onSelect={setSelectedPath}
        />
        <PreviewPanel jobId={job.id} jobCompleted={job.status === "completed"} />
      </div>
    </div>
  );
}
