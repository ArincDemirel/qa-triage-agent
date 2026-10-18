---
doc_id: WIKI-TRIAGE
title: Regression Triage Guide
type: wiki
test_ids:
---
# Regression Triage Guide

## Read the failure pattern first
The shape of a failure narrows the cause quickly.
- Failing on a subset of units at every temperature usually points to hardware: a
  component lot, a supplier change, or a fixture problem on those units.
- Failing on every unit at every temperature usually points to firmware timing or
  configuration, such as sampling or refresh settings.
- Failing at one temperature on every unit points to a temperature-dependent
  algorithm, for example filtering or compensation behaviour in the cold or heat.
- One measurement on one unit with no matching change is usually a flaky fixture
  contact. Retest before escalating.

## Then check what changed
Compare the failing build with the last passing build and read every change between
them. Search past Radars for the same test and symptom before filing a new one. A
change that sounds related is not proof; look for evidence that ties it to the symptom.
