CREATE TABLE builds (
    build_id    INTEGER PRIMARY KEY,   -- previous build = next-lower build_id
    build_date  TEXT NOT NULL,         -- ISO date, '2026-03-13'
    changelog   TEXT NOT NULL          -- JSON list of change entries
);

CREATE TABLE tests (
    test_id     TEXT PRIMARY KEY,      -- 'TS-4471': same ID as in Radars and reports
    name        TEXT NOT NULL,         -- 'Calibration offset'
    unit        TEXT NOT NULL,         -- 'mV'
    limit_low   REAL,                  -- NULL = no lower limit
    limit_high  REAL                   -- NULL = no upper limit
);

CREATE TABLE test_results (
    id          INTEGER PRIMARY KEY,
    build_id    INTEGER NOT NULL REFERENCES builds(build_id),
    test_id     TEXT    NOT NULL REFERENCES tests(test_id),
    sensor_id   TEXT    NOT NULL,      -- 'S01'..'S08'
    temp_c      INTEGER NOT NULL,      -- -20, 25, 60
    value       REAL    NOT NULL,
    passed      INTEGER NOT NULL       -- 1 = pass, 0 = fail
);

CREATE INDEX idx_results_build_test ON test_results(build_id, test_id);