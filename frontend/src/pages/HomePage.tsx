import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createJob } from "../api/client";
import { useProjectStore } from "../store/useProjectStore";

const EXAMPLE_PROMPTS = [
  "Create a colourful modern todo app with React and Vite",
  "Build a landing page for a coffee shop using HTML, CSS, and JavaScript",
  "Create a simple blog API in FastAPI with a SQLite database",
];

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const setCachedJob = useProjectStore((state) => state.setJob);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const job = await createJob(prompt.trim());
      setCachedJob(job);
      navigate(`/job/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create job");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page home-page">
      <header className="hero">
        <div className="hero-badge">AI app builder</div>
        <h1>Slothcode</h1>
        <p className="hero-subtitle">
          Describe the app you want. Our agents plan, architect, and code it — then preview it live.
        </p>
      </header>

      <form className="prompt-form panel" onSubmit={handleSubmit}>
        <label htmlFor="prompt">What do you want to build?</label>
        <textarea
          id="prompt"
          rows={5}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. Create a React dashboard with charts and dark mode..."
          disabled={loading}
        />
        {error && <p className="error-text">{error}</p>}
        <button type="submit" className="btn primary large" disabled={loading || !prompt.trim()}>
          {loading ? "Starting..." : "Generate project"}
        </button>
      </form>

      <section className="examples">
        <h3>Try an example</h3>
        <div className="example-chips">
          {EXAMPLE_PROMPTS.map((example) => (
            <button
              key={example}
              type="button"
              className="chip"
              onClick={() => setPrompt(example)}
              disabled={loading}
            >
              {example}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
