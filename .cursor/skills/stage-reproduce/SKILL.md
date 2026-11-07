---
name: stage-reproduce
description: Restore, verify and replay research evidence without replacing history.
---

Read docs/automation.md. Verify portable archive before extraction into a new
empty directory. Check source/input snapshots, uv.lock, declared environment,
hashes, plan, report, review and closure. Run historical verification separately
from live source verification. Replay in a new attempt with explicit lineage;
state whether this is same-data replay, fresh-seed repeat, reanalysis or new-data
replication. Never overwrite original artifacts or claim statistical independence
from a same-code replay. Record numerical tolerances and hardware differences.
