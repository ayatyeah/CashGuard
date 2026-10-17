#!/usr/bin/env bash
# Smoke test "as the jury": upload videos through the public API (ТЗ §10), poll until done, check the contract and timing.
# usage: BASE=https://host KEY=... bash scripts/smoke_jury.sh video1.mov [video2.mov ...]
set -euo pipefail
: "${BASE:?BASE url required}"; : "${KEY:?KEY required}"
for v in "$@"; do
  t0=$(date +%s)
  resp=$(curl -sS -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@$v")
  job=$(printf '%s' "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["status"]=="processing", d; print(d["job_id"])')
  echo "[$(basename "$v")] job=$job"
  r=""
  while :; do
    sleep 5
    r=$(curl -sS -m 30 "$BASE/api/v1/jobs/$job" -H "X-API-Key: $KEY" || true)
    st=$(printf '%s' "$r" | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["status"])
except Exception: print("transient")')
    [ "$st" = "done" ] || [ "$st" = "failed" ] && break
    [ $(( $(date +%s) - t0 )) -gt 330 ] && { echo "  TIMEOUT > 330 s"; break; }
  done
  dt=$(( $(date +%s) - t0 ))
  printf '%s' "$r" | python3 - "$dt" <<'PY'
import json, sys, base64
r = json.load(sys.stdin); dt = int(sys.argv[1])
assert r["status"] in ("done", "failed"), r["status"]
assert r.get("verdict") in ("violation", "no_violation") or r["status"] == "failed", r.get("verdict")
vs = r.get("violations") or []
assert (r.get("verdict") == "violation") == bool(vs) or r["status"] == "failed"
for v in vs:
    assert v["type"] and v["description"] and v["timecode_sec"] >= 0 and 1 <= len(v["evidence_frames"]) <= 3
    for f in v["evidence_frames"]:
        b = base64.b64decode(f["image_base64"], validate=True); assert b[:2] == b"\xff\xd8", "not JPEG"
        assert not f["image_base64"].startswith("data:")
print(f"  status={r['status']} verdict={r.get('verdict')} events={[(v['type'][:40], v['timecode_sec']) for v in vs]}  {dt}s {'OK' if dt <= 300 else 'OVER 5 MIN'}")
PY
done
echo "unauthorized check:"; curl -sS -o /dev/null -w '  no key -> HTTP %{http_code}\n' -X POST "$BASE/api/v1/analyze" -F "file=@$1"
