"""The agent: a LangGraph loop of  model -> tools -> model -> ... -> answer.

    START -> agent --(asks for tools, under step limit)--> tools -> agent
                   --(no tool calls: final answer)-----------> END
                   --(asks for tools, step limit hit)-------> give_up -> END

LangChain provides the pieces (chat model, tool definitions, messages).
LangGraph provides the control flow (state, routing, when to stop).
"""
import json
import operator
import os
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from qa.tools import run_tool

MAX_STEPS = 8

SYSTEM_PROMPT = """You are a regression triage assistant for hardware validation engineers.
A CI build has finished. Investigate it with the tools and write a short triage note.

How to work:
1. Call get_failures for the build to see what failed and the failure pattern.
2. Compare the failing test against the previous build with compare_builds. If the failure
   is only at one temperature or on some units, pass temp_c or sensor_id to isolate it.
3. Search the docs twice: once for the triage guide, and once describing the SYMPTOM
   in words (e.g. "calibration offset fails only in the cold"), not just the test ID.
   For each candidate change, look for evidence for AND against it.
4. Decide which changelog entry best explains the failure, using evidence, not keywords.

Rules:
- Every number you state must come from a tool result. Never calculate or estimate.
- Cite documents by chunk_id in square brackets, e.g. [RDAR-118204#s1].
- If no change explains the failure (e.g. one isolated measurement), say so.
- You only suggest; an engineer decides.

End with exactly these two lines:
LIKELY CAUSE: <exact changelog entry, or "none: <reason>">
CONFIDENCE: <low|medium|high>"""


# ---- Tools the model can see. The docstring is the description the model reads, and
# the type hints become the parameter schema. Each one goes through run_tool, so bad
# arguments come back as {"error": ...} instead of crashing the loop.

@tool
def get_failures(build_id: int) -> str:
    """List failing tests in a CI build: which sensors and temperatures fail, the failing
    values, the test limits, the previous build, and this build's changelog. Start here."""
    return json.dumps(run_tool("get_failures", {"build_id": build_id}))


@tool
def compare_builds(test_id: str, build_a: int, build_b: int,
                   temp_c: Optional[int] = None, sensor_id: Optional[str] = None) -> str:
    """Compare one test between two builds (usually the previous build and the failing one):
    mean/min/max per build from ALL matching measurements, the change in mean, and every
    changelog entry between them. Use temp_c (-20, 25 or 60) or sensor_id (e.g. S05) to
    isolate a failure pattern."""
    args = {"test_id": test_id, "build_a": build_a, "build_b": build_b,
            "temp_c": temp_c, "sensor_id": sensor_id}
    return json.dumps(run_tool("compare_builds", args))


@tool
def search_docs(query: str, test_id: Optional[str] = None,
                doc_type: Optional[str] = None) -> str:
    """Search past Radars (doc_type='radar') and wiki pages (doc_type='wiki'), including the
    regression triage guide. Use it to find past incidents with the same symptom and how
    they were resolved. Include test IDs when you know them."""
    return json.dumps(run_tool("search_docs", {"query": query, "test_id": test_id,
                                               "doc_type": doc_type}))


TOOLS = [get_failures, compare_builds, search_docs]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---- State: what flows through the graph. `add_messages` and `operator.add` tell
# LangGraph to append new items instead of replacing the list.

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    trace: Annotated[list[dict], operator.add]
    decision: str           # set by the human reviewer: "approved", "rejected" or "edited"


def make_llm():
    """The model sits behind one function: swapping it is a config change."""
    provider = os.environ.get("QA_LLM_PROVIDER", "ollama")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.environ.get("QA_LLM_MODEL", "qwen2.5:7b"), temperature=0)
    raise ValueError(f"Unknown provider {provider}")


def build_graph(llm, max_steps=MAX_STEPS):
    model = llm.bind_tools(TOOLS)       # tells the model which tools exist and their schemas

    def agent_node(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    def tools_node(state: State):
        results, trace = [], []
        for call in state["messages"][-1].tool_calls:
            t0 = time.perf_counter()
            output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            trace.append({"step": state["steps"] + 1, "tool": call["name"],
                          "args": call["args"], "ms": round((time.perf_counter() - t0) * 1000),
                          "result_preview": output[:200]})
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results, "steps": state["steps"] + 1, "trace": trace}

    def give_up_node(state: State):
        return {"messages": [AIMessage(content=f"Stopped after {max_steps} tool steps.\n"
                                       "LIKELY CAUSE: none: step limit reached\nCONFIDENCE: low")]}

    def review_node(state: State):
        """Human in the loop. interrupt() pauses the whole graph here and hands the draft
        to the caller. The graph's state is saved by the checkpointer, so the run can wait
        for minutes (or days) and resume exactly where it stopped."""
        draft = state["messages"][-1].content
        answer = interrupt({"draft": draft, "question": "Approve, reject, or send an edit?"})
        # Execution continues here only after someone resumes with Command(resume=...).
        if answer == "approve":
            return {"decision": "approved"}
        if answer == "reject":
            return {"decision": "rejected"}
        return {"decision": "edited", "messages": [AIMessage(content=answer)]}  # their edit wins

    def route(state: State):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "review"                                  # final draft -> human review
        return "give_up" if state["steps"] >= max_steps else "tools"

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("give_up", give_up_node)
    g.add_node("review", review_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route,
                            {"tools": "tools", "give_up": "give_up", "review": "review"})
    g.add_edge("tools", "agent")
    g.add_edge("give_up", "review")
    g.add_edge("review", END)
    # A checkpointer saves state after every step; interrupt() needs it to pause and resume.
    # InMemorySaver lives in this process. In production you'd use SQLite or Postgres, so a
    # run can survive restarts and be approved later from another process (a web UI, Slack).
    return g.compile(checkpointer=InMemorySaver())


def start_triage(build_id, llm=None, max_steps=MAX_STEPS):
    """Run until the graph pauses for review. Returns the draft plus what's needed to resume."""
    graph = build_graph(llm or make_llm(), max_steps)
    config = {"configurable": {"thread_id": f"triage-{build_id}-{time.time_ns()}"},
              "recursion_limit": 2 * max_steps + 5}    # thread_id = which saved run this is
    t0 = time.perf_counter()
    state = graph.invoke({"messages": [SystemMessage(SYSTEM_PROMPT),
                                       HumanMessage(f"Triage CI build {build_id}.")],
                          "steps": 0, "trace": []}, config)
    return {"graph": graph, "config": config, "build_id": build_id,
            "draft": state["__interrupt__"][0].value["draft"],
            "trace": state["trace"], "steps": state["steps"],
            "seconds": round(time.perf_counter() - t0, 1)}


def resume_triage(run, answer):
    """Resume the paused graph with the reviewer's answer: 'approve', 'reject' or edited text."""
    state = run["graph"].invoke(Command(resume=answer), run["config"])
    return {"decision": state["decision"], "note": state["messages"][-1].content}