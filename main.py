"""Command line entry point.

    python main.py triage --build 33
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from qa.agent import resume_triage, start_triage


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage")
    t.add_argument("--build", type=int, required=True)
    t.add_argument("--max-steps", type=int, default=8)
    a = p.parse_args()

    run = start_triage(a.build, max_steps=a.max_steps)          # runs until it pauses

    print(run["draft"])
    print(f"\n--- {len(run['trace'])} tool calls in {run['steps']} steps, {run['seconds']}s")
    for t in run["trace"]:
        print(f"  step {t['step']}: {t['tool']}({json.dumps(t['args'])})")

    # The graph is paused at the review node. Nothing is posted until a person decides.
    answer = input("\n[a]pprove, [r]eject, or type a corrected note: ").strip()
    answer = {"a": "approve", "r": "reject", "": "reject"}.get(answer.lower(), answer)
    final = resume_triage(run, answer)                          # resumes from the saved state
    print(f"Decision: {final['decision']}")

    Path("runs").mkdir(exist_ok=True)
    path = Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}-build{a.build}.json"
    path.write_text(json.dumps({"build_id": a.build, "draft": run["draft"],
                                "decision": final["decision"], "final_note": final["note"],
                                "trace": run["trace"], "seconds": run["seconds"]}, indent=2))
    print(f"Trace saved to {path}")


if __name__ == "__main__":
    main()