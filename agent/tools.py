import pathlib
import subprocess
from typing import Callable, Tuple

from langchain_core.tools import tool

DEFAULT_PROJECT_ROOT = pathlib.Path.cwd() / "generated_project"
PROJECT_ROOT = DEFAULT_PROJECT_ROOT


def _make_safe_path(project_root: pathlib.Path) -> Callable[[str], pathlib.Path]:
    root = project_root.resolve()

    def safe_path_for_project(path: str) -> pathlib.Path:
        p = (project_root / path).resolve()
        if root not in p.parents and root != p:
            raise ValueError("Attempt to access outside project root")
        return p

    return safe_path_for_project


def create_file_tools(project_root: pathlib.Path):
    """Create file tools bound to a specific project directory."""
    safe_path = _make_safe_path(project_root)

    @tool
    def write_file(path: str, content: str) -> str:
        """Writes content to a file at the specified path within the project root."""
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"WROTE:{p}"

    @tool
    def read_file(path: str) -> str:
        """Reads content from a file at the specified path within the project root."""
        p = safe_path(path)
        if not p.exists():
            return ""
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    @tool
    def get_current_directory() -> str:
        """Returns the current working directory."""
        return str(project_root.resolve())

    @tool
    def list_files(directory: str = ".") -> str:
        """Lists all files in the specified directory within the project root."""
        p = safe_path(directory)
        if not p.is_dir():
            return f"ERROR: {p} is not a directory"
        files = [
            str(f.relative_to(project_root.resolve()))
            for f in p.glob("**/*")
            if f.is_file()
        ]
        return "\n".join(files) if files else "No files found."

    @tool
    def run_cmd(cmd: str, cwd: str = None, timeout: int = 30) -> Tuple[int, str, str]:
        """Runs a shell command in the specified directory and returns the result."""
        cwd_dir = safe_path(cwd) if cwd else project_root
        res = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return res.returncode, res.stdout, res.stderr

    return [write_file, read_file, list_files, get_current_directory, run_cmd]


# Default tools for CLI backward compatibility
_default_tools = create_file_tools(DEFAULT_PROJECT_ROOT)
write_file = _default_tools[0]
read_file = _default_tools[1]
list_files = _default_tools[2]
get_current_directory = _default_tools[3]
run_cmd = _default_tools[4]


def init_project_root(project_root: pathlib.Path | None = None) -> str:
    root = project_root or PROJECT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return str(root.resolve())


def list_project_files(project_root: pathlib.Path) -> list[str]:
    """List all files under a project root (for API use)."""
    if not project_root.exists():
        return []
    root = project_root.resolve()
    return [
        str(f.relative_to(root)).replace("\\", "/")
        for f in root.glob("**/*")
        if f.is_file()
    ]
