#!/usr/bin/env bash
# Queue stress "as the jury": upload N videos at once, then report time-to-done for each (must stay ≤ 300 s).
# usage: BASE=https://host KEY=... bash scripts/smoke_queue.sh video1.mov video2.mov video3.mov
set -euo pipefail
: "${BASE:?}"; : "${KEY:?}"
t0=$(date +%s); jobs=()
for v in "$@"; do
  id=$(curl -sS -m 120 -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@$v" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
  jobs+=("$id:$(basename "$v")"); echo "queued $(basename "$v") -> $id"
done
pending=("${jobs[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  sleep 10; rest=()
  for j in "${pending[@]}"; do
    id=${j%%:*}; name=${j#*:}
    st=$(curl -sS -m 30 "$BASE/api/v1/jobs/$id" -H "X-API-Key: $KEY" | python3 -c 'import json,sys
try: d=json.load(sys.stdin); print(d["status"], d.get("verdict"))
except Exception: print("transient")')
    case "$st" in
      done*|failed*) echo "$name: $st after $(( $(date +%s) - t0 ))s $([ $(( $(date +%s) - t0 )) -le 300 ] && echo OK || echo 'OVER 5 MIN')";;
      *) rest+=("$j");;
    esac
  done
  pending=("${rest[@]+"${rest[@]}"}")
  [ $(( $(date +%s) - t0 )) -gt 400 ] && { echo "STOP: still pending: ${pending[*]}"; break; }
done
