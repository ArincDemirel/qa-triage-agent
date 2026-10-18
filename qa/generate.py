"""Simulated lab: fills the tables with realistic measurements and planted regressions.

Run:  python -m qa.generate
Writes data/qa.db (the tables) and data/ground_truth.json (the answer key).
The agent never sees ground_truth.json; only the evaluator does.
"""
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEED = 7                       # same seed -> identical data every run
BUILDS = range(30, 40)         # 10 builds
SENSORS = [f"S0{i}" for i in range(1, 9)]   # 8 physical units, S01..S08
TEMPS = [-20, 25, 60]          # chamber setpoints, Celsius

# test_id: (name, unit, limit_low, limit_high, normal_mean, normal_sigma)
TESTS = {
    "TS-4471": ("Calibration offset", "mV", -1.0, 1.0, 0.35, 0.12),
    "TS-2210": ("Touch latency",      "ms", None, 12.0, 8.5,  0.6),
    "TS-6002": ("Noise floor",        "uV", None, 40.0, 22.0, 3.0),
}

DISTRACTORS = [
    "firmware/logging: reduce verbose logging in touch driver",
    "firmware/power: idle timeout 30 s -> 20 s",
    "firmware/haptics: tune click waveform amplitude",
    "sw/ui: add touch sensitivity toggle in settings",
    "firmware/gesture: add two-finger swipe recognizer",
    "tools/station: update fixture calibration script to v2.3",
]

# The answer key. Each case: where it shows up, and what really caused it.
CASES = [
    {"case": "R1", "build": 33, "test_id": "TS-4471",
     "pattern": "all units, only at -20C",
     "culprit": "firmware/touch_filter: IIR coefficient 0.82 -> 0.65 (smoother scrolling)",
     "fix": "firmware/touch_filter: revert IIR coefficient to 0.82"},
    {"case": "R2", "build": 35, "test_id": "TS-2210",
     "pattern": "all units, all temperatures",
     "culprit": "firmware/scan: panel scan rate 240 Hz -> 120 Hz (power savings)",
     "fix": "firmware/scan: restore panel scan rate to 240 Hz"},
    {"case": "R3", "build": 37, "test_id": "TS-6002",
     "pattern": "only units S05 and S06, all temperatures",
     "culprit": "hw/bom: switch S05, S06 to sensor lot B (second supplier)",
     "fix": "hw/bom: return S05, S06 to sensor lot A"},
    {"case": "F1", "build": 31, "test_id": "TS-2210",
     "pattern": "one measurement on one unit",
     "culprit": None, "fix": None},   # flaky: nothing in the changelog caused it
]

# Hard distractor: lands in the same build as R1 and *sounds* related to a cold failure.
THERMAL_DISTRACTOR = "firmware/thermal: update temperature compensation table"


def regression_shift(build, test_id, temp_c, sensor_id):
    """How far a planted problem pushes a measurement. The conditions ARE the pattern."""
    if build == 33 and test_id == "TS-4471" and temp_c == -20:
        return 1.1                                   # R1: cold-only offset jump
    if build == 35 and test_id == "TS-2210":
        return 5.0                                   # R2: latency up everywhere
    if build == 37 and test_id == "TS-6002" and sensor_id in ("S05", "S06"):
        return 25.0                                  # R3: two noisy units
    if (build, test_id, sensor_id, temp_c) == (31, "TS-2210", "S03", 25):
        return 5.0                                   # F1: one flaky measurement
    return 0.0


def make_changelog(rng, build):
    entries = rng.sample(DISTRACTORS, 2)
    for c in CASES:
        if c["culprit"] and build == c["build"]:
            entries.append(c["culprit"])
        if c["fix"] and build == c["build"] + 1:
            entries.append(c["fix"])
    if build == 33:
        entries.append(THERMAL_DISTRACTOR)
    rng.shuffle(entries)        # so the culprit isn't always in the same position
    return entries


def is_pass(value, low, high):
    return (low is None or value >= low) and (high is None or value <= high)


def generate(db_path="data/qa.db", schema_path="qa/schema.sql",
             truth_path="data/ground_truth.json"):
    rng = random.Random(SEED)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(db_path).unlink(missing_ok=True)          # start fresh every run
    con = sqlite3.connect(db_path)
    con.executescript(Path(schema_path).read_text())

    for tid, (name, unit, low, high, _, _) in TESTS.items():
        con.execute("INSERT INTO tests VALUES (?, ?, ?, ?, ?)", (tid, name, unit, low, high))

    start = date(2026, 3, 2)
    for i, build in enumerate(BUILDS):
        con.execute("INSERT INTO builds VALUES (?, ?, ?)",
                    (build, (start + timedelta(days=3 * i)).isoformat(),
                     json.dumps(make_changelog(rng, build))))
        for tid, (_, _, low, high, mean, sigma) in TESTS.items():
            for sensor in SENSORS:
                for temp in TEMPS:
                    value = rng.gauss(mean, sigma) + regression_shift(build, tid, temp, sensor)
                    con.execute(
                        "INSERT INTO test_results (build_id, test_id, sensor_id, temp_c, value, passed) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (build, tid, sensor, temp, round(value, 3), int(is_pass(value, low, high))))
    con.commit()
    verify(con)
    con.close()
    Path(truth_path).write_text(json.dumps(
        [{k: v for k, v in c.items() if k != "fix"} for c in CASES], indent=2))
    print(f"Wrote {db_path} and {truth_path}")


def verify(con):
    """Self-check: every failure in the table must come from a planted case.
    A random failure would silently corrupt the answer key."""
    failing = con.execute("SELECT build_id, test_id, sensor_id, temp_c FROM test_results "
                          "WHERE passed = 0").fetchall()
    unplanted = [r for r in failing if regression_shift(r[0], r[1], r[3], r[2]) == 0]
    assert not unplanted, f"Unplanted failures (change seed or sigma): {unplanted}"
    for c in CASES:
        n = con.execute("SELECT COUNT(*) FROM test_results WHERE build_id=? AND test_id=? "
                        "AND passed=0", (c["build"], c["test_id"])).fetchone()[0]
        assert n > 0, f"Case {c['case']} didn't produce any failure"
        print(f"  {c['case']}: build {c['build']} {c['test_id']} -> {n} failing measurements")


if __name__ == "__main__":
    generate()