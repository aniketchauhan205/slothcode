import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain_core.output_parsers import PydanticOutputParser
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from agent.prompts import architect_prompt, coder_system_prompt, planner_prompt
from agent.states import CoderState, File as ProjectFile, ImplementationTask, Plan, TaskPlan
from agent.tools import create_file_tools, read_file, write_file, list_files

_ = load_dotenv()

EventCallback = Callable[[str, dict], None]
CancelCallback = Callable[[], bool]

DEFAULT_HF_MODEL = (
    "DavidAU/Mistral-Nemo-2407-12B-Thinking-Claude-Gemini-GPT5.2-"
    "Uncensored-HERETIC:featherless-ai"
)


@dataclass
class HuggingFaceMessage:
    content: str


class HuggingFaceRouterClient:
    """Hugging Face Router client using the OpenAI-compatible SDK."""

    def __init__(self):
        self.base_url = _normalize_hf_base_url(
            os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1")
        )
        self.model = os.getenv("LLM_MODEL", DEFAULT_HF_MODEL)
        self.api_key = os.getenv("HF_TOKEN")
        self.timeout = int(os.getenv("HF_TIMEOUT", "120"))

        if not self.api_key:
            raise RuntimeError("HF_TOKEN is required")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def invoke(self, prompt: str) -> HuggingFaceMessage:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        except APIStatusError as exc:
            body = _api_error_body(exc)
            raise RuntimeError(
                f"Hugging Face Router rejected the request with HTTP "
                f"{exc.status_code}: {body}"
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Hugging Face Router request failed: {exc}") from exc
        except APIError as exc:
            raise RuntimeError(f"Hugging Face Router API error: {exc}") from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Hugging Face Router response: {completion}"
            ) from exc

        if not content:
            raise RuntimeError(f"Hugging Face Router returned empty content: {completion}")

        return HuggingFaceMessage(content=content)


def _normalize_hf_base_url(url: str) -> str:
    """Accept either the SDK base URL or the older full chat-completions URL."""
    normalized = url.rstrip("/")
    return normalized.removesuffix("/chat/completions")


def _api_error_body(exc: APIStatusError) -> str:
    response = getattr(exc, "response", None)
    body = ""
    if response is not None:
        try:
            body = response.text.strip()
        except Exception:
            body = ""

    if not body:
        body = str(exc)

    if "api.featherless.ai" in body and "Error 1010" in body:
        return (
            "api.featherless.ai blocked the request with Cloudflare Error 1010. "
            "The Hugging Face Router request reached the Featherless provider, "
            "but that provider rejected this backend/client signature."
        )

    body = body[:1200]
    if (
        exc.status_code == 400
        and "request was rejected as invalid" in body.lower()
    ):
        body += (
            "\n\nHint: Hugging Face accepted the request but rejected it before "
            "generation. With this Featherless model, a common cause is that the "
            "model is exposed as a Text Generation model while this code is using "
            "the OpenAI-compatible Chat Completions endpoint."
        )
    return body


def _get_llm():
    return HuggingFaceRouterClient()


def _text_from_response(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _invoke_text(llm, prompt: str, phase: str) -> str:
    try:
        return _text_from_response(llm.invoke(prompt))
    except Exception as exc:
        raise RuntimeError(f"{phase} model request failed: {exc}") from exc


def _plan_format_instructions() -> str:
    return """
Return only valid JSON with this exact shape.
Do not include markdown, explanations, or <think> blocks:
Use the key "files" exactly, not "_files".
File paths must be relative paths without a leading slash.
{
  "name": "Project name",
  "description": "One sentence description",
  "techstack": "Main technologies",
  "features": ["feature one", "feature two"],
  "files": [
    {"path": "relative/path.ext", "purpose": "why this file exists"}
  ]
}
"""


def _task_plan_format_instructions() -> str:
    return """
Return only valid JSON with this exact shape.
Do not include markdown, explanations, or <think> blocks:
Use the key "implementation_steps" exactly.
File paths must be relative paths without a leading slash.
{
  "implementation_steps": [
    {
      "filepath": "relative/path.ext",
      "task_description": "Specific implementation instructions for this file"
    }
  ]
}
"""


def _build_fallback_task_plan(plan: Plan) -> TaskPlan:
    tasks: list[ImplementationTask] = []
    seen: set[str] = set()

    def add_task(filepath: str, task_description: str):
        normalized = filepath.replace("\\", "/").strip().lstrip("/")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        tasks.append(
            ImplementationTask(
                filepath=normalized,
                task_description=task_description,
            )
        )

    for file in plan.files:
        add_task(
            file.path,
            (
                f"Create the complete file for '{file.path}'. Purpose: {file.purpose}. "
                f"The project is '{plan.name}', described as: {plan.description}. "
                f"Use this tech stack: {plan.techstack}. Include all required imports, "
                "exports, styles, markup, and working logic for this file."
            ),
        )

    add_task(
        "package.json",
        (
            "Create a package.json for the generated project. If this is a web app, "
            "make it Vite-compatible and include dev, build, and preview scripts. "
            "Include only dependencies that the generated files actually import."
        ),
    )
    add_task(
        "README.md",
        (
            f"Create a concise README for {plan.name} with setup, dev, build, "
            "and preview instructions."
        ),
    )

    return TaskPlan(implementation_steps=tasks)


def _normalize_relative_path(path: object) -> str:
    return str(path).replace("\\", "/").strip().lstrip("/")


def _normalize_plan_payload(payload: dict) -> dict:
    normalized = dict(payload)

    if "files" not in normalized and "_files" in normalized:
        normalized["files"] = normalized.pop("_files")

    files = normalized.get("files")
    if isinstance(files, list):
        normalized["files"] = [
            {
                **file,
                "path": _normalize_relative_path(file.get("path", "")),
            }
            for file in files
            if isinstance(file, dict)
        ]

    return normalized


def _normalize_task_plan_payload(payload: dict) -> dict:
    normalized = dict(payload)

    if "implementation_steps" not in normalized and "_implementation_steps" in normalized:
        normalized["implementation_steps"] = normalized.pop("_implementation_steps")

    steps = normalized.get("implementation_steps")
    if isinstance(steps, list):
        normalized_steps = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            normalized_step = dict(step)
            filepath = (
                normalized_step.get("filepath")
                or normalized_step.get("file_path")
                or normalized_step.get("filePath")
                or normalized_step.get("path")
            )
            if filepath is not None:
                normalized_step["filepath"] = _normalize_relative_path(filepath)
            normalized_steps.append(normalized_step)
        normalized["implementation_steps"] = normalized_steps

    return normalized


def _build_fallback_plan(user_prompt: str) -> Plan:
    cleaned_prompt = re.sub(r"\s+", " ", user_prompt).strip()
    title_seed = re.sub(r"^(create|build|make|generate)\s+", "", cleaned_prompt, flags=re.I)
    title_words = re.findall(r"[A-Za-z0-9]+", title_seed)[:6]
    project_name = " ".join(word.capitalize() for word in title_words) or "Generated Web App"

    return Plan(
        name=project_name,
        description=f"A generated React/Vite project for: {cleaned_prompt}",
        techstack="React, Vite, TypeScript, HTML, CSS",
        features=[
            "Responsive user interface",
            "Content and structure based on the user prompt",
            "Runnable Vite preview with npm run dev",
        ],
        files=[
            ProjectFile(path="package.json", purpose="Defines dependencies and dev/build/preview scripts"),
            ProjectFile(path="index.html", purpose="Vite HTML entry file"),
            ProjectFile(path="src/main.tsx", purpose="React application entry point"),
            ProjectFile(path="src/App.tsx", purpose="Main app component for the requested project"),
            ProjectFile(path="src/App.css", purpose="Application styling"),
            ProjectFile(path="README.md", purpose="Setup and run instructions"),
        ],
    )


def _ensure_core_web_tasks(plan: Plan, task_plan: TaskPlan) -> TaskPlan:
    techstack = plan.techstack.lower()
    paths = {
        step.filepath.replace("\\", "/").strip().lstrip("/")
        for step in task_plan.implementation_steps
    }

    def has_file(path: str) -> bool:
        return path in paths

    def add_task(path: str, description: str):
        if has_file(path):
            return
        paths.add(path)
        task_plan.implementation_steps.append(
            ImplementationTask(filepath=path, task_description=description)
        )

    is_web_app = any(
        term in techstack
        for term in ["react", "vite", "html", "css", "javascript", "typescript", "web"]
    )
    if not is_web_app:
        return task_plan

    has_index = any(path.lower() == "index.html" for path in paths)
    has_package = any(path.lower() == "package.json" for path in paths)

    if not has_package:
        add_task(
            "package.json",
            "Create a Vite package.json with dev, build, and preview scripts.",
        )
    if not has_index:
        add_task(
            "index.html",
            "Create the Vite HTML entry file with a root element and module script.",
        )
    if not any(path.startswith("src/main.") for path in paths):
        add_task(
            "src/main.tsx",
            "Create the React/Vite entry point that mounts the App component.",
        )
    if not any(path.startswith("src/App.") for path in paths):
        add_task(
            "src/App.tsx",
            "Create the main React App component for the requested project.",
        )

    return task_plan


def _component_name_from_path(path: str) -> str:
    stem = pathlib.Path(path).stem
    name = re.sub(r"[^0-9A-Za-z]+", " ", stem).title().replace(" ", "")
    if not name:
        name = "GeneratedComponent"
    if name[0].isdigit():
        name = f"Component{name}"
    return name


def _fallback_file_content(plan: Plan | None, task: ImplementationTask) -> str:
    path = task.filepath.replace("\\", "/")
    lower_path = path.lower()
    project_name = plan.name if plan else "Generated App"
    description = plan.description if plan else "A generated web application."
    features = plan.features if plan else []

    if lower_path == "package.json":
        return json.dumps(
            {
                "name": re.sub(r"[^a-z0-9-]+", "-", project_name.lower()).strip("-")
                or "generated-app",
                "private": True,
                "version": "0.1.0",
                "type": "module",
                "scripts": {
                    "dev": "vite",
                    "build": "vite build",
                    "preview": "vite preview",
                },
                "dependencies": {
                    "@vitejs/plugin-react": "^4.6.0",
                    "vite": "^7.0.4",
                    "typescript": "~5.8.3",
                    "react": "^19.1.0",
                    "react-dom": "^19.1.0",
                },
                "devDependencies": {},
            },
            indent=2,
        )

    if lower_path == "index.html":
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""

    if lower_path.endswith("src/main.tsx") or lower_path.endswith("src/main.jsx"):
        app_import = "./App"
        css_import = "\nimport './App.css';" if "tsx" in lower_path else "\nimport './App.css';"
        return f"""import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '{app_import}';{css_import}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
"""

    if lower_path.endswith("app.css") or lower_path.endswith("style.css") or lower_path.endswith(".css"):
        return """* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7fb;
  color: #18202f;
}

.app {
  min-height: 100vh;
  padding: 48px 20px;
}

.content {
  max-width: 960px;
  margin: 0 auto;
}

.hero {
  margin-bottom: 32px;
}

.hero h1 {
  margin: 0 0 12px;
  font-size: clamp(2rem, 5vw, 4rem);
  line-height: 1;
}

.hero p {
  margin: 0;
  max-width: 720px;
  color: #566176;
  font-size: 1.1rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.card {
  background: #ffffff;
  border: 1px solid #dfe4ee;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 10px 30px rgba(26, 35, 55, 0.08);
}

.card h2 {
  margin: 0 0 8px;
  font-size: 1.1rem;
}

.card p {
  margin: 0;
  color: #566176;
  line-height: 1.6;
}
"""

    if lower_path.endswith("readme.md"):
        return f"""# {project_name}

{description}

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```
"""

    if lower_path.endswith(".tsx") or lower_path.endswith(".jsx"):
        component_name = _component_name_from_path(path)
        if component_name.lower() == "app":
            card_items = features or [
                "Simple, focused content",
                "Responsive layout",
                "Clean visual presentation",
            ]
            cards = "\n".join(
                f"""        <article className="card">
          <h2>{feature}</h2>
          <p>{description}</p>
        </article>"""
                for feature in card_items
            )
            return f"""export default function App() {{
  return (
    <main className="app">
      <section className="content">
        <header className="hero">
          <h1>{project_name}</h1>
          <p>{description}</p>
        </header>
        <section className="grid" aria-label="Highlights">
{cards}
        </section>
      </section>
    </main>
  );
}}
"""

        return f"""import type {{ ReactNode }} from 'react';

interface {component_name}Props {{
  title?: string;
  children?: ReactNode;
}}

export default function {component_name}({{ title = '{component_name}', children }}: {component_name}Props) {{
  return (
    <section className="card">
      <h2>{{title}}</h2>
      <p>{{children ?? {json.dumps(task.task_description)}}}</p>
    </section>
  );
}}
"""

    if lower_path.endswith(".ts") or lower_path.endswith(".js"):
        return f"""export const projectName = {json.dumps(project_name)};
export const projectDescription = {json.dumps(description)};
export const projectFeatures = {json.dumps(features, indent=2)};
"""

    return f"{task.task_description}\n"


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    full_fence = re.fullmatch(r"```[\w.+-]*\s*\n(?P<body>.*?)\n```", text, re.DOTALL)
    if full_fence:
        return full_fence.group("body").strip()

    partial_fence = re.search(r"```[\w.+-]*\s*\n(?P<body>.*?)\n```", text, re.DOTALL)
    if partial_fence:
        return partial_fence.group("body").strip()

    return text


def _strip_reasoning_blocks(content: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


def _extract_json_object(content: str) -> str:
    text = _strip_code_fence(_strip_reasoning_blocks(content))
    decoder = json.JSONDecoder()

    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return json.dumps(parsed)

    raise ValueError(f"No JSON object found in model output: {text[:500]}")


def build_agent(
    project_root: pathlib.Path | None = None,
    on_event: EventCallback | None = None,
    should_cancel: CancelCallback | None = None,
):
    """Build a LangGraph agent, optionally scoped to a project directory."""
    llm = _get_llm()

    if project_root is not None:
        tool_write, tool_read, tool_list, _, _ = create_file_tools(project_root)
    else:
        tool_read = read_file
        tool_write = write_file
        tool_list = list_files

    def emit(event_type: str, data: dict | None = None):
        if on_event:
            on_event(event_type, data or {})

    def check_cancelled():
        if should_cancel and should_cancel():
            raise RuntimeError("Job cancelled by user")

    def planner_agent(state: dict) -> dict:
        check_cancelled()
        emit("status", {"phase": "planning", "message": "Planning project..."})
        user_prompt = state["user_prompt"]

        parser = PydanticOutputParser(pydantic_object=Plan)
        full_prompt = f"{planner_prompt(user_prompt)}\n\n{_plan_format_instructions()}"
        fallback_mode = False

        try:
            response_payload = json.loads(
                _extract_json_object(_invoke_text(llm, full_prompt, "Planner"))
            )
            response_text = json.dumps(_normalize_plan_payload(response_payload))
            resp = parser.parse(response_text)
        except Exception as exc:
            fallback_mode = True
            emit(
                "warning",
                {
                    "phase": "planning",
                    "message": (
                        "Planner model request failed, so a local fallback "
                        "plan was created from the user prompt."
                    ),
                    "detail": str(exc),
                },
            )
            resp = _build_fallback_plan(user_prompt)

        if resp is None:
            raise ValueError("Planner did not return a valid response.")
        emit("plan", resp.model_dump())
        return {"plan": resp, "coder_fallback_mode": fallback_mode}

    def architect_agent(state: dict) -> dict:
        check_cancelled()
        emit("status", {"phase": "architecting", "message": "Creating implementation plan..."})
        plan: Plan = state["plan"]
        parser = PydanticOutputParser(pydantic_object=TaskPlan)
        fallback_mode = bool(state.get("coder_fallback_mode"))

        try:
            if fallback_mode:
                raise RuntimeError("Planner used local fallback mode.")
            full_prompt = (
                f"{architect_prompt(plan=plan.model_dump_json())}\n\n"
                f"{_task_plan_format_instructions()}"
            )
            response_payload = json.loads(
                _extract_json_object(_invoke_text(llm, full_prompt, "Architect"))
            )
            response_text = json.dumps(_normalize_task_plan_payload(response_payload))
            resp = parser.parse(response_text)
        except Exception as exc:
            fallback_mode = True
            emit(
                "warning",
                {
                    "phase": "architecting",
                    "message": (
                        "Architect model request failed, so a deterministic "
                        "implementation plan was created from the planner output."
                    ),
                    "detail": str(exc),
                },
            )
            resp = _build_fallback_task_plan(plan)

        if resp is None:
            raise ValueError("Architect did not return a valid response.")
        resp = _ensure_core_web_tasks(plan, resp)
        resp.plan = plan
        emit("task_plan", {"steps": len(resp.implementation_steps), "plan": resp.model_dump()})
        return {"task_plan": resp, "coder_fallback_mode": fallback_mode}

    def coder_agent(state: dict) -> dict:
        check_cancelled()
        coder_state: CoderState = state.get("coder_state")
        if coder_state is None:
            coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

        steps = coder_state.task_plan.implementation_steps
        if coder_state.current_step_idx >= len(steps):
            emit("status", {"phase": "done", "message": "All files generated."})
            return {"coder_state": coder_state, "status": "DONE"}

        current_task = steps[coder_state.current_step_idx]
        check_cancelled()
        step_num = coder_state.current_step_idx + 1
        emit(
            "coding",
            {
                "step": step_num,
                "total": len(steps),
                "filepath": current_task.filepath,
                "message": f"Writing {current_task.filepath} ({step_num}/{len(steps)})",
            },
        )

        plan = getattr(coder_state.task_plan, "plan", state.get("plan"))
        fallback_mode = bool(state.get("coder_fallback_mode"))

        if fallback_mode:
            content = _fallback_file_content(plan, current_task)
        else:
            existing_content = tool_read.invoke({"path": current_task.filepath})
            current_files = tool_list.invoke({"directory": "."})
            system_prompt = coder_system_prompt()
            file_prompt = (
                f"{system_prompt}\n\n"
                "Write the full contents for exactly one file.\n"
                f"File path: {current_task.filepath}\n"
                f"Task: {current_task.task_description}\n\n"
                f"Current project files:\n{current_files}\n\n"
                f"Existing content for this file:\n{existing_content}\n\n"
                "Return only the complete file content. Do not include explanations, "
                "markdown fences, or placeholder comments."
            )

            try:
                content = _strip_code_fence(
                    _strip_reasoning_blocks(_invoke_text(llm, file_prompt, "Coder"))
                )
            except Exception as exc:
                fallback_mode = True
                emit(
                    "warning",
                    {
                        "phase": "coding",
                        "filepath": current_task.filepath,
                        "message": (
                            "Coder model request failed, so local fallback file "
                            "generation will be used for this and remaining files."
                        ),
                        "detail": str(exc),
                    },
                )
                content = _fallback_file_content(plan, current_task)

        if not content:
            raise ValueError(f"Coder returned empty content for {current_task.filepath}")

        check_cancelled()
        tool_write.invoke({"path": current_task.filepath, "content": content})

        coder_state.current_step_idx += 1
        emit("file_written", {"filepath": current_task.filepath, "content": content})
        return {"coder_state": coder_state, "coder_fallback_mode": fallback_mode}

    graph = StateGraph(dict)
    graph.add_node("planner", planner_agent)
    graph.add_node("architect", architect_agent)
    graph.add_node("coder", coder_agent)
    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "coder")
    graph.add_conditional_edges(
        "coder",
        lambda s: "END" if s.get("status") == "DONE" else "coder",
        {"END": END, "coder": "coder"},
    )
    graph.set_entry_point("planner")
    return graph.compile()


class _LazyAgent:
    """Build the default CLI agent only when it is first used."""

    def __init__(self):
        self._compiled_agent = None

    def _get_agent(self):
        if self._compiled_agent is None:
            self._compiled_agent = build_agent()
        return self._compiled_agent

    def invoke(self, *args, **kwargs):
        return self._get_agent().invoke(*args, **kwargs)


# Default agent for CLI backward compatibility.
agent = _LazyAgent()

if __name__ == "__main__":
    result = agent.invoke(
        {"user_prompt": "Build a colourful modern todo app in html css and js"},
        {"recursion_limit": 100},
    )
    print("Final State:", result)
