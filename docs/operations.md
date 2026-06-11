# Operations

## Memory backend health checks

Memory backends (LanceDB, or any vector store) can fail silently in production. A broken index means agents lose semantic recall with no visible error. [`scripts/memory-healthcheck.sh`](../scripts/memory-healthcheck.sh) is a runnable check: it confirms the table exists, runs a real query against it, and verifies freshness, alerting through a webhook on failure.

```bash
LANCEDB_PATH=./memory/lancedb \
ALERT_ENDPOINT=https://hooks.example.com/... \
FRESHNESS_HOURS=48 \
  scripts/memory-healthcheck.sh
```

**Scheduling options by platform:**

| Platform | Method | Example |
|----------|--------|---------|
| macOS | launchd plist | `StartInterval: 1800` (every 30 min) |
| Linux | systemd timer | `OnUnitActiveSec=30min` |
| Any | cron | `*/30 * * * * /path/to/memory-healthcheck.sh` |

**Key principles:**
- **Test with a real query**, not just file existence. A corrupt Lance file passes `ls` but fails queries.
- **Check freshness.** A stale index is functionally equivalent to a broken one.
- **Alert, don't just log.** If the backend is down, agents run blind. Use a webhook for immediate notice.
- **Lock against concurrent runs.** Health checks that overlap with embedding jobs can corrupt state.
