"""Typed tools. The LLM picks a tool and fills in parameters; this code does the
filtering and the math, and returns numbers plus what they were based on.

Rules:
  * Numbers come from exact SQL filters, never from search.
  * Read-only: the database is opened in read-only mode.
  * Bad inputs return {"error": ...} instead of crashing, so the agent can recover.
  * Numeric answers report rows_used, so the basis of every number is visible.
"""
import json
import sqlite3
import statistics

from qa.search import hybrid_search

DB_PATH = "data/qa.db"
VALID_TEMPS = (-20, 25, 60)


class ToolError(Exception):
    pass


def _connect():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)    # read-only by construction
    con.row_factory = sqlite3.Row                                  # rows behave like dicts
    return con


def _check_build(con, build_id):
    row = con.execute("SELECT * FROM builds WHERE build_id = ?", (build_id,)).fetchone()
    if row is None:
        raise ToolError(f"Build {build_id} not found.")
    return row


def _check_test(con, test_id):
    row = con.execute("SELECT * FROM tests WHERE test_id = ?", (test_id,)).fetchone()
    if row is None:
        valid = [r[0] for r in con.execute("SELECT test_id FROM tests")]
        raise ToolError(f"Unknown test_id '{test_id}'. Valid: {', '.join(valid)}")
    return row


def _stats(values):
    if not values:
        return {}
    return {"mean": round(statistics.fmean(values), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


# ------------------------------------------------------------------ tools

def get_failures(build_id: int) -> dict:
    """What failed in a build, and the PATTERN: which units, which temperatures."""
    with _connect() as con:
        build = _check_build(con, build_id)
        prev = con.execute("SELECT MAX(build_id) FROM builds WHERE build_id < ?",
                           (build_id,)).fetchone()[0]
        failures = []
        for (test_id,) in con.execute("SELECT DISTINCT test_id FROM test_results "
                                      "WHERE build_id = ? AND passed = 0", (build_id,)):
            test = _check_test(con, test_id)
            rows = con.execute("SELECT sensor_id, temp_c, value FROM test_results "
                               "WHERE build_id = ? AND test_id = ? AND passed = 0",
                               (build_id, test_id)).fetchall()
            total = con.execute("SELECT COUNT(*) FROM test_results WHERE build_id = ? "
                                "AND test_id = ?", (build_id, test_id)).fetchone()[0]
            failures.append({
                "test_id": test_id, "name": test["name"], "unit": test["unit"],
                "limit_low": test["limit_low"], "limit_high": test["limit_high"],
                "failing": len(rows), "of": total,
                "sensors": sorted({r["sensor_id"] for r in rows}),
                "temperatures_c": sorted({r["temp_c"] for r in rows}),
                "failing_values": _stats([r["value"] for r in rows]),
            })
        return {"build_id": build_id, "previous_build": prev, "failures": failures,
                "changelog": json.loads(build["changelog"])}


def compare_builds(test_id: str, build_a: int, build_b: int,
                   temp_c: int | None = None, sensor_id: str | None = None) -> dict:
    """One test in two builds, from ALL matching rows, plus every change between them."""
    with _connect() as con:
        _check_build(con, build_a)
        _check_build(con, build_b)
        test = _check_test(con, test_id)
        if temp_c is not None and temp_c not in VALID_TEMPS:
            raise ToolError(f"temp_c must be one of {VALID_TEMPS}")

        where, args = "", []
        if temp_c is not None:
            where += " AND temp_c = ?"
            args.append(temp_c)
        if sensor_id is not None:
            where += " AND sensor_id = ?"
            args.append(sensor_id)

        out = {"test_id": test_id, "name": test["name"], "unit": test["unit"],
               "limit_low": test["limit_low"], "limit_high": test["limit_high"],
               "filters": {"temp_c": temp_c, "sensor_id": sensor_id}, "builds": {}}
        for b in (build_a, build_b):
            rows = con.execute("SELECT value, passed FROM test_results WHERE build_id = ? "
                               "AND test_id = ?" + where, (b, test_id, *args)).fetchall()
            out["builds"][str(b)] = {**_stats([r["value"] for r in rows]),
                                     "failing": sum(1 for r in rows if not r["passed"]),
                                     "rows_used": len(rows)}
        a, b = out["builds"][str(build_a)], out["builds"][str(build_b)]
        if a and b:
            out["mean_change"] = round(b["mean"] - a["mean"], 3)

        lo, hi = sorted((build_a, build_b))
        out["changes_between"] = [
            {"build": r["build_id"], "change": change}
            for r in con.execute("SELECT build_id, changelog FROM builds WHERE build_id > ? "
                                 "AND build_id <= ? ORDER BY build_id", (lo, hi))
            for change in json.loads(r["changelog"])]
        return out


def search_docs(query: str, test_id: str | None = None, doc_type: str | None = None) -> dict:
    """Hybrid search over past Radars and wiki pages."""
    return {"query": query, "results": hybrid_search(query, k=5, test_id=test_id,
                                                     doc_type=doc_type)}


TOOLS = {f.__name__: f for f in (get_failures, compare_builds, search_docs)}


def run_tool(name: str, args: dict) -> dict:
    """Run a tool by name. Errors become data the agent can read and react to."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}"}
    try:
        return TOOLS[name](**args)
    except ToolError as e:
        return {"error": str(e)}
    except (TypeError, ValueError) as e:          # wrong or missing arguments from the model
        return {"error": f"Invalid arguments for {name}: {e}"}


if __name__ == "__main__":
    print(json.dumps(run_tool("get_failures", {"build_id": 33}), indent=2))
    print(json.dumps(run_tool("compare_builds", {"test_id": "TS-4471", "build_a": 32,
                                                 "build_b": 33, "temp_c": -20}), indent=2))
    print(run_tool("compare_builds", {"test_id": "TS-9999", "build_a": 32, "build_b": 33}))
    print(run_tool("get_failures", {"build_id": 99}))