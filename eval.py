"""Replay every planted case through the agent and score the answers.

    python eval.py               # each case 3 times
    python eval.py --runs 1      # quicker

Checks per run:
  cause     LIKELY CAUSE names the planted culprit (or "none" for flaky / no-failure builds)
  grounded  every decimal number in the note appears in some tool output
  searched  the agent searched the docs at least once
  temp      for the cold-only case, compare_builds was filtered to -20 C
The answer key (data/ground_truth.json) is only read here, never by the agent.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import ToolMessage

from qa.agent import start_triage


def load_cases():
    cases = json.loads(Path("data/ground_truth.json").read_text())
    # A fix build with no failures: the right answer is "nothing to triage".
    cases.append({"case": "N1", "build": 34, "test_id": None,
                  "pattern": "fix build, no failures", "culprit": None})
    return cases


def likely_cause(note):
    m = re.search(r"LIKELY CAUSE:\s*(.+)", note)
    return m.group(1).strip() if m else None


def cause_correct(case, cause):
    if cause is None:
        return False                          # missing the required line counts as wrong
    if case["culprit"] is None:
        return cause.lower().startswith("none")
    # The component path ("firmware/touch_filter") identifies the culprit; distractors
    # in the same build use different paths, so this can't be passed by accident.
    component = case["culprit"].split(":")[0].lower()
    return component in cause.lower()


def ungrounded_numbers(note, tool_outputs):
    body = note.split("LIKELY CAUSE:")[0]     # the cause line quotes the changelog verbatim
    return [n for n in re.findall(r"-?\d+\.\d+", body) if n not in tool_outputs]


def run_case(case):
    run = start_triage(case["build"])
    messages = run["graph"].get_state(run["config"]).values["messages"]
    tool_outputs = " ".join(m.content for m in messages if isinstance(m, ToolMessage))
    calls = [t["tool"] for t in run["trace"]]
    cause = likely_cause(run["draft"])
    temp_ok = None
    if case["case"] == "R1":                  # the cold-only case should be sliced by temperature
        temp_ok = any(t["tool"] == "compare_builds" and t["args"].get("temp_c") == -20
                      for t in run["trace"])
    return {"case": case["case"], "build": case["build"], "cause": cause,
            "correct": cause_correct(case, cause),
            "ungrounded": ungrounded_numbers(run["draft"], tool_outputs),
            "searched": "search_docs" in calls, "temp_filter": temp_ok,
            "steps": run["steps"], "seconds": run["seconds"], "draft": run["draft"],
            "trace": run["trace"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    a = p.parse_args()

    results = []
    for case in load_cases():
        for i in range(a.runs):
            r = run_case(case)
            results.append(r)
            flag = "PASS" if r["correct"] else "FAIL"
            print(f"{case['case']} build {case['build']} run {i + 1}: {flag}  {r['cause']}")

    print("\n| Case | Build | Pattern | Correct | Grounded | Searched | Avg steps | Avg time |")
    print("|---|---|---|---|---|---|---|---|")
    for case in load_cases():
        rs = [r for r in results if r["case"] == case["case"]]
        n = len(rs)
        print(f"| {case['case']} | {case['build']} | {case['pattern']} | "
              f"{sum(r['correct'] for r in rs)}/{n} | "
              f"{sum(not r['ungrounded'] for r in rs)}/{n} | "
              f"{sum(r['searched'] for r in rs)}/{n} | "
              f"{sum(r['steps'] for r in rs) / n:.1f} | "
              f"{sum(r['seconds'] for r in rs) / n:.0f}s |")
    total = sum(r["correct"] for r in results)
    print(f"\nOverall: {total}/{len(results)} correct root causes")

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Details (drafts and traces): {out}")


if __name__ == "__main__":
    main()