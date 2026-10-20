# Demo video checklist (target 5-8 minutes)

Record the lab desktop with any screen recorder (OBS / Windows Game Bar /
macOS Cmd+Shift+5). Speak over each step; show real output only.

1. (30 s) Problem, dataset, architecture diagram.
2. (45 s) `python scripts/prepare_data.py` output + one or two EDA figures.
3. (60 s) `python -m heartml.train` summary, then MLflow UI: experiment list,
   compare LR vs RF parent runs, open one run's artifacts (plots, model).
4. (30 s) `pytest -v` and `ruff check .` passing locally.
5. (60 s) GitHub Actions: a green run (both jobs), uploaded artifacts, the
   GHCR package; briefly show the deliberately failed run.
6. (45 s) `docker build`, `docker run`, smoke test: valid prediction and 422.
7. (60 s) Minikube: `kubectl get pods,svc -n heart-mlops`, scale to 3,
   curl `/predict` through port-forward or tunnel, `kubectl logs`.
8. (60 s) Run the traffic generator; Prometheus targets page (2-3 UP);
   Grafana dashboard panels updating.
9. (30 s) Limitations and conclusion.
