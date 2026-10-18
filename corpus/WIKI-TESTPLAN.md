---
doc_id: WIKI-TESTPLAN
title: Touch Sensor Validation Test Plan
type: wiki
test_ids: TS-4471, TS-2210, TS-6002
---
# Touch Sensor Validation Test Plan

## Purpose
Defines the automated tests CI runs on every firmware and hardware build of the touch
sensor stack. Each build is measured on eight units (S01-S08) in the thermal chamber
at three setpoints.

## Test definitions
| Test ID | Name | Unit | Lower limit | Upper limit | Temperatures (C) |
|---|---|---|---|---|---|
| TS-4471 | Calibration offset | mV | -1.0 | 1.0 | -20, 25, 60 |
| TS-2210 | Touch latency | ms | | 12.0 | -20, 25, 60 |
| TS-6002 | Noise floor | uV | | 40.0 | -20, 25, 60 |

## Procedure
Units soak for 20 minutes at each chamber setpoint before measurement. Results upload
automatically to the results database, and a build is flagged when any measurement
falls outside its limits.
