import { Link, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import JobPage from "./pages/JobPage";

export default function App() {
  return (
    <div className="app-shell">
      <nav className="top-nav">
        <Link to="/" className="brand">
          <span className="brand-icon">🦥</span>
          Slothcode
        </Link>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/job/:jobId" element={<JobPage />} />
        </Routes>
      </main>
    </div>
  );
}
