import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  getDownloadUrl,
  getJob,
  listFiles,
  subscribeToJobEvents,
} from "../api/client";
import FileExplorer from "../components/FileExplorer";
import JobProgress from "../components/JobProgress";
import PreviewPanel from "../components/PreviewPanel";
import type { Job, JobEvent } from "../types";

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshFiles = useCallback(async () => {
    if (!jobId) return;
    try {
      const fileList = await listFiles(jobId);
      setFiles(fileList);
      if (fileList.length > 0 && !selectedPath) {
        setSelectedPath(fileList[0]);
      }
    } catch {
      // files may not exist yet
    }
  }, [jobId, selectedPath]);

  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;

    getJob(jobId)
      .then((data) => {
        if (!cancelled) setJob(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Job not found");
      });

    const unsubscribe = subscribeToJobEvents(
      jobId,
      (event) => {
        setEvents((prev) => [...prev, event]);
        if (event.type === "file_written" || event.type === "completed") {
          refreshFiles();
        }
        if (event.type === "completed" || event.type === "error") {
          getJob(jobId).then((data) => {
            if (!cancelled) setJob(data);
          });
        }
      },
      () => {
        getJob(jobId).then((data) => {
          if (!cancelled) setJob(data);
        });
      },
    );

    const poll = setInterval(() => {
      getJob(jobId).then((data) => {
        if (!cancelled) setJob(data);
      });
      refreshFiles();
    }, 5000);

    refreshFiles();

    return () => {
      cancelled = true;
      unsubscribe();
      clearInterval(poll);
    };
  }, [jobId, refreshFiles]);

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

  const isDone = job.status === "completed" || job.status === "failed";

  return (
    <div className="page job-page">
      <header className="job-header">
        <div>
          <Link to="/" className="back-link">
            ← New project
          </Link>
          <h1>{job.plan?.name ? String(job.plan.name) : "Generating project"}</h1>
          <p className="job-prompt">{job.prompt}</p>
        </div>
        <div className="job-actions">
          {isDone && job.file_count > 0 && (
            <a href={getDownloadUrl(job.id)} className="btn secondary" download>
              Download ZIP
            </a>
          )}
        </div>
      </header>

      {job.error && <div className="error-banner">{job.error}</div>}

      <div className="job-grid">
        <JobProgress events={events} status={job.status} />
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
