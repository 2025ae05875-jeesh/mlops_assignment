"""Send synthetic traffic (mostly valid, some invalid) to the API so that
logs, Prometheus metrics and Grafana panels have data to show.

Usage:  python scripts/generate_traffic.py --url http://localhost:8080 -n 200
Uses only the standard library (works outside the venv too).
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng: random.Random) -> dict:
    return {
        "age": rng.randint(29, 77), "sex": rng.choice([0, 1]),
        "cp": rng.choice([1, 2, 3, 4]), "trestbps": rng.randint(94, 200),
        "chol": rng.randint(126, 564), "fbs": rng.choice([0, 1]),
        "restecg": rng.choice([0, 1, 2]), "thalach": rng.randint(71, 202),
        "exang": rng.choice([0, 1]), "oldpeak": round(rng.uniform(0, 6.2), 1),
        "slope": rng.choice([1, 2, 3]), "ca": rng.choice([0, 1, 2, 3, None]),
        "thal": rng.choice([3, 6, 7]),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--invalid-rate", type=float, default=0.15)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()
    rng = random.Random(7)
    counts: dict = {}
    for _ in range(args.n):
        p = random_patient(rng)
        if rng.random() < args.invalid_rate:
            p[rng.choice(["age", "sex", "chol"])] = rng.choice([-5, 999, "abc"])
        code = post(args.url.rstrip("/") + "/predict", p)
        counts[code] = counts.get(code, 0) + 1
        time.sleep(args.delay)
    print("status code counts:", counts)


if __name__ == "__main__":
    main()
