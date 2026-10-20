#!/usr/bin/env bash
# Smoke-test a running API.  Usage: bash scripts/smoke_test_api.sh [BASE_URL]
# Exits non-zero on any failure (used by CI after `docker run`).
set -euo pipefail
BASE="${1:-http://localhost:8000}"
VALID='{"age":58,"sex":1,"cp":4,"trestbps":140,"chol":260,"fbs":0,"restecg":2,"thalach":130,"exang":1,"oldpeak":1.8,"slope":2,"ca":1,"thal":7}'
INVALID='{"age":58,"sex":5,"cp":4}'

for i in $(seq 1 30); do
  curl -sf "$BASE/ready" >/dev/null && break
  echo "waiting for API ($i)"; sleep 2
done

echo "--- GET /health";  curl -sf "$BASE/health"; echo
echo "--- GET /ready";   curl -sf "$BASE/ready"; echo
echo "--- POST /predict (valid)"
curl -sf -X POST "$BASE/predict" -H "Content-Type: application/json" -d "$VALID" | tee /tmp/pred.json; echo
grep -q '"prediction"' /tmp/pred.json
grep -q '"confidence"' /tmp/pred.json
echo "--- POST /predict (invalid, expect 422)"
code=$(curl -s -o /tmp/err.json -w "%{http_code}" -X POST "$BASE/predict" \
       -H "Content-Type: application/json" -d "$INVALID")
cat /tmp/err.json; echo
[ "$code" = "422" ] || { echo "expected 422, got $code"; exit 1; }
echo "--- GET /metrics (first lines)"
curl -sf "$BASE/metrics" | grep "^heart_api_requests_total" | head -5
echo "SMOKE TEST PASSED"
