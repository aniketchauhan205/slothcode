import os
import pathlib
from typing import Callable

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agent.prompts import architect_prompt, coder_system_prompt, planner_prompt
from agent.states import CoderState, Plan, TaskPlan
from agent.tools import create_file_tools, read_file, write_file, get_current_directory, list_files

_ = load_dotenv()

EventCallback = Callable[[str, dict], None]


def _get_llm():
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0.2)


def build_agent(
    project_root: pathlib.Path | None = None,
    on_event: EventCallback | None = None,
):
    """Build a LangGraph agent, optionally scoped to a project directory."""
    llm = _get_llm()

    if project_root is not None:
        tool_write, tool_read, tool_list, tool_cwd, _ = create_file_tools(project_root)
        coder_tools = [tool_read, tool_write, tool_list, tool_cwd]
    else:
        tool_read = read_file
        tool_write = write_file
        tool_list = list_files
        tool_cwd = get_current_directory
        coder_tools = [tool_read, tool_write, tool_list, tool_cwd]

    def emit(event_type: str, data: dict | None = None):
        if on_event:
            on_event(event_type, data or {})

    def planner_agent(state: dict) -> dict:
        emit("status", {"phase": "planning", "message": "Planning project..."})
        user_prompt = state["user_prompt"]
        resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
        if resp is None:
            raise ValueError("Planner did not return a valid response.")
        emit("plan", resp.model_dump())
        return {"plan": resp}

    def architect_agent(state: dict) -> dict:
        emit("status", {"phase": "architecting", "message": "Creating implementation plan..."})
        plan: Plan = state["plan"]
        resp = llm.with_structured_output(TaskPlan).invoke(
            architect_prompt(plan=plan.model_dump_json())
        )
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

        existing_content = tool_read.run(current_task.filepath)
        system_prompt = coder_system_prompt()
        user_prompt = (
            f"Task: {current_task.task_description}\n"
            f"File: {current_task.filepath}\n"
            f"Existing content:\n{existing_content}\n"
            "Use write_file(path, content) to save your changes."
        )

        react_agent = create_react_agent(llm, coder_tools)
        react_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            }
        )

        coder_state.current_step_idx += 1
        emit("file_written", {"filepath": current_task.filepath})
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


# Default agent for CLI backward compatibility
agent = build_agent()

if __name__ == "__main__":
    result = agent.invoke(
        {"user_prompt": "Build a colourful modern todo app in html css and js"},
        {"recursion_limit": 100},
    )
    print("Final State:", result)
