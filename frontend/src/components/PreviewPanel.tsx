import { useEffect, useState } from "react";
import { getPreviewStatus, startPreview, stopPreview } from "../api/client";

interface PreviewPanelProps {
  jobId: string;
  jobCompleted: boolean;
}

export default function PreviewPanel({ jobId, jobCompleted }: PreviewPanelProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!jobCompleted) return;
    getPreviewStatus(jobId)
      .then((status) => {
        setRunning(status.running);
        setPreviewUrl(status.preview_url);
      })
      .catch(() => {});
  }, [jobId, jobCompleted]);

  async function handleStart() {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const result = await startPreview(jobId);
      setPreviewUrl(result.preview_url);
      setRunning(true);
      setMessage(result.message ?? "Preview container started.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start preview");
    } finally {
      setLoading(false);
    }
  }

  async function handleStop() {
    setLoading(true);
    setError(null);
    try {
      await stopPreview(jobId);
      setRunning(false);
      setPreviewUrl(null);
      setMessage("Preview stopped.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stop preview");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel preview-panel">
      <div className="panel-header">
        <h2>Live preview</h2>
        <div className="preview-actions">
          {running ? (
            <button type="button" className="btn secondary" onClick={handleStop} disabled={loading}>
              Stop preview
            </button>
          ) : (
            <button
              type="button"
              className="btn primary"
              onClick={handleStart}
              disabled={!jobCompleted || loading}
            >
              {loading ? "Starting..." : "Start preview"}
            </button>
          )}
        </div>
      </div>

      {!jobCompleted && (
        <p className="muted empty-state">
          Complete the generation first. Preview requires a project with a{" "}
          <code>package.json</code> dev script.
        </p>
      )}

      {message && <p className="info-text">{message}</p>}
      {error && <p className="error-text">{error}</p>}

      {previewUrl && running && (
        <div className="preview-frame-wrap">
          <div className="preview-url-bar">
            <a href={previewUrl} target="_blank" rel="noreferrer">
              {previewUrl}
            </a>
          </div>
          <iframe
            title="Project preview"
            src={previewUrl}
            className="preview-frame"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          />
        </div>
      )}
    </section>
  );
}
