---
name: stage-execute
description: Execute and resume bounded research stages with preserved evidence.
---

Follow docs/automation.md. Check status, active lock/PID and authorization.
Run only the frozen plan through scripts/research.py run <ID>. One dispatcher;
no hidden parallel training. Inspect progress logs. Preserve failed/interrupted
attempts and every seed. On timeout/cap, report outcome and remaining design
uncertainty; do not silently enlarge budgets. Resume only a task checkpoint
whose full fingerprint is verified, otherwise use a new attempt. Verify all
registered outputs. Do not call an optimizer trajectory a capacity effect.
