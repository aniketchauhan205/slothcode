import { useEffect, useState } from "react";
import type { JobEvent } from "../types";

interface JobProgressProps {
  events: JobEvent[];
  status: string;
}

function eventLabel(event: JobEvent): string {
  const { type, data } = event;
  if (type === "status" && typeof data.message === "string") return data.message;
  if (type === "plan" && typeof data.name === "string") return `Planned: ${data.name}`;
  if (type === "task_plan" && typeof data.steps === "number") {
    return `Created ${data.steps} implementation steps`;
  }
  if (type === "warning" && typeof data.message === "string") return data.message;
  if (type === "coding" && typeof data.message === "string") return data.message;
  if (type === "file_written" && typeof data.filepath === "string") {
    return `Saved ${data.filepath}`;
  }
  if (type === "completed" && typeof data.message === "string") return data.message;
  if (type === "error" && typeof data.message === "string") return data.message;
  return type;
}

export default function JobProgress({ events, status }: JobProgressProps) {
  const [visible, setVisible] = useState<JobEvent[]>([]);

  useEffect(() => {
    setVisible(events);
  }, [events]);

  const statusClass =
    status === "completed"
      ? "badge success"
      : status === "failed"
        ? "badge error"
        : status === "running"
          ? "badge running"
          : "badge pending";

  return (
    <section className="panel progress-panel">
      <div className="panel-header">
        <h2>Progress</h2>
        <span className={statusClass}>{status}</span>
      </div>
      <ol className="event-log">
        {visible.length === 0 ? (
          <li className="event-item muted">Waiting for agent to start...</li>
        ) : (
          visible.map((event, i) => (
            <li key={`${event.timestamp}-${i}`} className={`event-item type-${event.type}`}>
              <span className="event-time">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
              <span className="event-text">{eventLabel(event)}</span>
            </li>
          ))
        )}
      </ol>
    </section>
  );
}
