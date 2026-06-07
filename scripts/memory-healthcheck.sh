#!/usr/bin/env bash
#
# memory-healthcheck.sh: verify the LanceDB semantic-memory backend is alive.
#
# A vector store can fail silently: a corrupt or empty index passes a file check
# but loses semantic recall with no visible error. Run this on a schedule (cron,
# launchd, or a systemd timer) and alert if it fails.
#
# Columns referenced (id, content, created_at) match SemanticMemory's schema.
#
# Usage:
#   LANCEDB_PATH=./memory/lancedb ALERT_ENDPOINT=https://hooks... ./memory-healthcheck.sh
#
set -euo pipefail

LANCEDB_PATH="${LANCEDB_PATH:-./memory/lancedb}"
ALERT_ENDPOINT="${ALERT_ENDPOINT:-}"
FRESHNESS_HOURS="${FRESHNESS_HOURS:-48}"

alert() {
  echo "ALERT: $1" >&2
  if [ -n "$ALERT_ENDPOINT" ]; then
    curl -fsS -X POST "$ALERT_ENDPOINT" \
      -H 'Content-Type: application/json' \
      -d "{\"text\": \"memory-healthcheck: $1\"}" || true
  fi
}

# 1. The table directory must exist (table name is "memories").
if [ ! -d "$LANCEDB_PATH/memories.lance" ]; then
  alert "memory DB missing at $LANCEDB_PATH"
  exit 1
fi

# 2. Run a real query. A corrupt index passes `ls` but fails here.
#    3. Then check freshness: a stale index is as bad as a broken one.
LANCEDB_PATH="$LANCEDB_PATH" FRESHNESS_HOURS="$FRESHNESS_HOURS" python3 - <<'PY' || { alert "memory DB query failed"; exit 1; }
import os
from datetime import datetime

import lancedb

path = os.environ["LANCEDB_PATH"]
limit = float(os.environ["FRESHNESS_HOURS"])

table = lancedb.connect(path).open_table("memories")
rows = table.count_rows()
assert rows > 0, "memories table is empty"

latest = max(r["created_at"] for r in table.search().limit(rows).to_list())
age_hours = (datetime.now() - datetime.fromisoformat(latest)).total_seconds() / 3600
print(f"OK: {rows} records, newest {age_hours:.1f}h old")
if age_hours > limit:
    raise SystemExit(f"stale: newest record is {age_hours:.0f}h old (limit {limit:.0f}h)")
PY
