import { useEffect, useState } from "react";
import { getFileContent } from "../api/client";
import { useProjectStore } from "../store/useProjectStore";

interface FileExplorerProps {
  jobId: string;
  files: string[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}

export default function FileExplorer({ jobId, files, selectedPath, onSelect }: FileExplorerProps) {
  const [preview, setPreview] = useState<string>("");
  const cachedFiles = useProjectStore((state) => state.filesByJob[jobId] ?? {});
  const setCachedFile = useProjectStore((state) => state.setFile);

  useEffect(() => {
    if (!selectedPath) {
      setPreview("");
      return;
    }
    if (cachedFiles[selectedPath]) {
      setPreview(cachedFiles[selectedPath]);
      return;
    }
    getFileContent(jobId, selectedPath)
      .then((data) => {
        setPreview(data.content);
        setCachedFile(jobId, data.path, data.content);
      })
      .catch(() => setPreview("Unable to load file."));
  }, [cachedFiles, jobId, selectedPath, setCachedFile]);

  return (
    <section className="panel files-panel">
      <div className="panel-header">
        <h2>Generated files</h2>
        <span className="file-count">{files.length} files</span>
      </div>

      {files.length === 0 ? (
        <p className="muted empty-state">Files will appear here as the agent writes them.</p>
      ) : (
        <div className="files-layout">
          <ul className="file-tree">
            {files.map((path) => (
              <li key={path}>
                <button
                  type="button"
                  className={`file-item ${selectedPath === path ? "active" : ""}`}
                  onClick={() => onSelect(path)}
                >
                  {path}
                </button>
              </li>
            ))}
          </ul>
          <div className="file-viewer">
            {selectedPath ? (
              <>
                <div className="file-viewer-header">{selectedPath}</div>
                <pre className="code-block">
                  <code>{preview}</code>
                </pre>
              </>
            ) : (
              <p className="muted empty-state">Select a file to view its contents.</p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
