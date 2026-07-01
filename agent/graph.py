import os
import pathlib
import re
from typing import Callable

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain_core.output_parsers import PydanticOutputParser

from agent.prompts import architect_prompt, coder_system_prompt, planner_prompt
from agent.states import CoderState, ImplementationTask, Plan, TaskPlan
from agent.tools import create_file_tools, read_file, write_file, list_files

_ = load_dotenv()

EventCallback = Callable[[str, dict], None]

DEFAULT_HF_MODEL = (
    "DavidAU/Mistral-Nemo-2407-12B-Thinking-Claude-Gemini-GPT5.2-"
    "Uncensored-HERETIC:featherless-ai"
)
DEFAULT_GOOGLE_MODEL = "gemini-3.5-flash"
SHUT_DOWN_GEMINI_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
}


def _is_gemini_model(model: str | None) -> bool:
    return bool(model and model.lower().startswith("gemini-"))


def _normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.lower().strip()
    if normalized in {"hf", "hugging_face", "huggingface"}:
        return "huggingface"
    return normalized


def _infer_provider(configured_provider: str | None, model: str | None) -> str:
    provider = _normalize_provider(configured_provider)
    if provider:
        return provider

    if os.getenv("HF_TOKEN"):
        return "huggingface"
    if _is_gemini_model(model):
        return "google"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    return "huggingface"


def _google_model(model: str | None) -> str:
    if not model:
        return DEFAULT_GOOGLE_MODEL
    normalized = model.lower()
    if normalized in SHUT_DOWN_GEMINI_MODELS:
        print(
            f"LLM_MODEL={model} is no longer available; using "
            f"{DEFAULT_GOOGLE_MODEL} instead."
        )
        return DEFAULT_GOOGLE_MODEL
    return model


def _huggingface_model(model: str | None) -> str:
    if not model:
        return DEFAULT_HF_MODEL
    if _is_gemini_model(model):
        print(
            f"LLM_MODEL={model} is not a Hugging Face Router model id; using "
            f"{DEFAULT_HF_MODEL} instead."
        )
        return DEFAULT_HF_MODEL
    return model


def _get_llm():
    model = os.getenv("LLM_MODEL")
    provider = _infer_provider(os.getenv("LLM_PROVIDER"), model)
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is required when using Gemini models"
            )
        return ChatGoogleGenerativeAI(
            model=_google_model(model),
            google_api_key=api_key,
            temperature=temperature,
        )

    if provider == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("HF_TOKEN is required when LLM_PROVIDER=huggingface")

        return ChatOpenAI(
            model=_huggingface_model(model),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://router.huggingface.co/v1"),
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        if _is_gemini_model(model):
            raise RuntimeError(
                "LLM_MODEL is a Gemini model, but LLM_PROVIDER=openai. "
                "Set LLM_PROVIDER=google with GOOGLE_API_KEY/GEMINI_API_KEY, "
                "or use an OpenAI model such as gpt-4o-mini."
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER={provider}")


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
Return only valid JSON with this exact shape, and do not wrap it in markdown:
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
Return only valid JSON with this exact shape, and do not wrap it in markdown:
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


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    full_fence = re.fullmatch(r"```[\w.+-]*\s*\n(?P<body>.*?)\n```", text, re.DOTALL)
    if full_fence:
        return full_fence.group("body").strip()

    partial_fence = re.search(r"```[\w.+-]*\s*\n(?P<body>.*?)\n```", text, re.DOTALL)
    if partial_fence:
        return partial_fence.group("body").strip()

    return text


def build_agent(
    project_root: pathlib.Path | None = None,
    on_event: EventCallback | None = None,
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

    def planner_agent(state: dict) -> dict:
        emit("status", {"phase": "planning", "message": "Planning project..."})
        user_prompt = state["user_prompt"]

        parser = PydanticOutputParser(pydantic_object=Plan)
        full_prompt = f"{planner_prompt(user_prompt)}\n\n{_plan_format_instructions()}"
        response_text = _strip_code_fence(_invoke_text(llm, full_prompt, "Planner"))
        resp = parser.parse(response_text)

        if resp is None:
            raise ValueError("Planner did not return a valid response.")
        emit("plan", resp.model_dump())
        return {"plan": resp}

    def architect_agent(state: dict) -> dict:
        emit("status", {"phase": "architecting", "message": "Creating implementation plan..."})
        plan: Plan = state["plan"]
        parser = PydanticOutputParser(pydantic_object=TaskPlan)

        try:
            full_prompt = (
                f"{architect_prompt(plan=plan.model_dump_json())}\n\n"
                f"{_task_plan_format_instructions()}"
            )
            response_text = _strip_code_fence(_invoke_text(llm, full_prompt, "Architect"))
            resp = parser.parse(response_text)
        except Exception as exc:
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
        resp.plan = plan
        emit("task_plan", {"steps": len(resp.implementation_steps), "plan": resp.model_dump()})
        return {"task_plan": resp}

    def coder_agent(state: dict) -> dict:
        coder_state: CoderState = state.get("coder_state")
        if coder_state is None:
            coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

        steps = coder_state.task_plan.implementation_steps
        if coder_state.current_step_idx >= len(steps):
            emit("status", {"phase": "done", "message": "All files generated."})
            return {"coder_state": coder_state, "status": "DONE"}

        current_task = steps[coder_state.current_step_idx]
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

        content = _strip_code_fence(_invoke_text(llm, file_prompt, "Coder"))
        if not content:
            raise ValueError(f"Coder returned empty content for {current_task.filepath}")

        tool_write.invoke({"path": current_task.filepath, "content": content})

        coder_state.current_step_idx += 1
        emit("file_written", {"filepath": current_task.filepath, "content": content})
        return {"coder_state": coder_state}

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
