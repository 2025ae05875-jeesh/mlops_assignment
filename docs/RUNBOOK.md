# RUNBOOK - step-by-step execution (Prayogshala lab)

Project root: `/home/cloud/workspace/mlops_assignment` (persistent EFS).
Every phase ends with a CHECKPOINT. Do not continue past a failed checkpoint.
Screenshot names refer to `screenshots/README.md`.

---------------------------------------------------------------------------
## Phase 0 - Start of EVERY session (terminal 1)

```bash
cd /home/cloud/workspace/mlops_assignment
source scripts/session_setup.sh          # first time in a VM: builds venv (2-4 min)
```
CHECKPOINT: prints `[setup] heartml 0.1.0 | sklearn 1.5.2 | mlflow 2.17.2`.

kubectl (once; it lives on persistent storage):
```bash
mkdir -p /home/cloud/workspace/bin && cd /home/cloud/workspace/bin
curl -LO "https://dl.k8s.io/release/v1.32.0/bin/linux/amd64/kubectl" && chmod +x kubectl
cd - && kubectl version --client
```

---------------------------------------------------------------------------
## Phase 1 - Data acquisition, validation, EDA  (Task 1, 5 marks)

```bash
python scripts/prepare_data.py
python scripts/run_eda.py
ls reports/figures && cat data/data_manifest.json | head -30
```
CHECKPOINT: 303 rows x 14 columns, missing `ca`=4 / `thal`=2, 0 duplicates,
class counts {0: 164, 1: 139}, six PNGs in reports/figures.
Screenshots 01, 02. Write the EDA section of docs/decisions.md now.

```bash
git add . && git commit -m "Data acquisition, validation and EDA" && git push
```

---------------------------------------------------------------------------
## Phase 2 - Train, tune, track  (Tasks 2-4, 20 marks)

```bash
python -m heartml.train            # ~1-3 min on 4 CPUs
python scripts/predict_sample.py   # loads artifact in a NEW process
cat models/model_metadata.json
```
What happens: stratified 80/20 split (seed 42) -> 5-fold CV grid search for
Logistic Regression (24 candidates) and Random Forest (36 candidates), refit
on ROC-AUC -> each candidate logged as a nested MLflow child run -> each
tuned model evaluated once on the test set (confusion matrix, ROC curve,
feature importance logged) -> champion = best CV ROC-AUC -> saved to
`models/heart_model.joblib` + `models/model_metadata.json` -> quality gate
(test ROC-AUC >= 0.80, else exit code 1).

CHECKPOINT: two `[model] CV ROC-AUC=... test ROC-AUC=...` lines, `[champion]`,
`[gate] PASS`, and three JSON predictions from predict_sample.py.
Copy YOUR numbers into docs/decisions.md - never use example numbers.

MLflow UI (terminal 2, venv active):
```bash
cd /home/cloud/workspace/mlops_assignment && source scripts/session_setup.sh
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000 in the lab browser -> experiment `heart-disease-uci`.
* Runs list: expand the two parent runs to see the child (candidate) runs.
* Tick both parent runs -> Compare -> parallel coordinates / metric table.
* Open a parent run -> Artifacts -> plots/, cv/, model/.
Screenshots 03-06. Stop the UI with Ctrl+C when done.

Clean-environment reproducibility proof (optional but strong evidence):
```bash
python3 -m venv /tmp/cleanenv && source /tmp/cleanenv/bin/activate
pip install -q -r requirements.txt && pip install -q -e .
python scripts/predict_sample.py && deactivate && source scripts/session_setup.sh
```

Commit (mlruns/ is git-ignored; models and reports are committed):
```bash
git add . && git commit -m "Model training, MLflow tracking, packaged champion" && git push
```

---------------------------------------------------------------------------
## Phase 3 - Tests + CI/CD  (Task 5, 8 marks)

Local first:
```bash
ruff check .
pytest -v
```
CHECKPOINT: `All checks passed!` and all tests passed (29 tests). Screenshot 07.

The push above already triggered `.github/workflows/ci.yml`. On GitHub:
Repository -> Actions -> `ci-cd` -> latest run.
* Job 1 `Lint, test, train`: ruff -> pytest -> prepare data -> EDA -> train
  (quality gate) -> fresh-process inference -> uploads `model` and
  `reports-mlruns-logs` artifacts.
* Job 2 `Docker build, smoke test, push to GHCR`: builds image with the model
  trained in job 1, runs it, smoke test (valid 200 + invalid 422), uploads
  container logs, pushes `ghcr.io/2025ae05875-jeesh/heart-disease-api:<sha>`
  and `:latest` (only on main).
CHECKPOINT: both jobs green. Screenshots 08, 09, 11 (Profile -> Packages).
If GHCR push is denied: repo Settings -> Actions -> General -> Workflow
permissions -> "Read and write permissions" -> re-run.
Make the package public (Package -> Package settings -> Change visibility)
if you want Minikube to pull it.

Demonstrate a failing pipeline (screenshot 10):
```bash
git switch -c demo/failing-test
sed -i 's/assert len(cleaned) == 7/assert len(cleaned) == 999/' tests/test_data.py
git commit -am "DEMO: intentionally failing test" && git push -u origin demo/failing-test
```
Show the red run and the failing assertion in the pytest log, then clean up:
```bash
git switch main && git branch -D demo/failing-test && git push origin --delete demo/failing-test
```

---------------------------------------------------------------------------
## Phase 4 - Docker  (Task 6, 5 marks)

```bash
docker build -t heart-disease-api:1.0.0 .
docker images heart-disease-api
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0.0
docker ps                                   # STATUS becomes "(healthy)" after ~30 s
bash scripts/smoke_test_api.sh http://localhost:8000
docker logs heart-api | tail -20            # JSON request logs, no patient data
```
Swagger UI: http://localhost:8000/docs (screenshot 20).
CHECKPOINT: `SMOKE TEST PASSED`. Screenshots 12, 13.
Stop & remove:
```bash
docker stop heart-api && docker rm heart-api
```
Note: the image COPIES models/; rebuild after retraining.

---------------------------------------------------------------------------
## Phase 5 - Kubernetes on Minikube  (Task 7, 7 marks)

```bash
minikube start --driver=docker --cpus=2 --memory=4096
kubectl get nodes                                   # minikube Ready
minikube image load heart-disease-api:1.0.0         # copy local image into node
minikube image ls | grep heart
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-configmap.yaml -f k8s/02-deployment.yaml -f k8s/03-service.yaml
kubectl rollout status deployment/heart-api -n heart-mlops
kubectl get all -n heart-mlops -o wide
```
CHECKPOINT: 2/2 pods `Running` and `READY 1/1`. Screenshot 14.

Access (choose what works in the lab):
```bash
# A) port-forward (always works; routes to one pod)
kubectl port-forward -n heart-mlops svc/heart-api-svc 8080:80
#    new terminal:
bash scripts/smoke_test_api.sh http://localhost:8080

# B) LoadBalancer via tunnel (may ask for sudo); then EXTERNAL-IP appears
minikube tunnel            # keep running
kubectl get svc heart-api-svc -n heart-mlops
curl http://<EXTERNAL-IP>/health

# C) NodePort URL helper
minikube service heart-api-svc -n heart-mlops --url
```
Scale and observe (screenshot 15):
```bash
kubectl scale deployment heart-api -n heart-mlops --replicas=3
kubectl get pods -n heart-mlops -w        # Ctrl+C when 3/3 Running
kubectl logs -n heart-mlops -l app=heart-api --tail=5 --prefix
kubectl describe deployment heart-api -n heart-mlops | sed -n '1,40p'
```
Optional Ingress:
```bash
minikube addons enable ingress
kubectl apply -f k8s/04-ingress.yaml
curl -H "Host: heart-api.local" http://$(minikube ip)/health
```
Optional: use the GHCR image instead of the loaded one:
```bash
kubectl set image deployment/heart-api -n heart-mlops \
  heart-api=ghcr.io/2025ae05875-jeesh/heart-disease-api:latest
```
Screenshots 16, 17.

---------------------------------------------------------------------------
## Phase 6 - Monitoring  (Task 8, 3 marks)

```bash
docker pull prom/prometheus:v2.54.1 && docker pull grafana/grafana:11.2.0
minikube image load prom/prometheus:v2.54.1
minikube image load grafana/grafana:11.2.0

read -s -p "Grafana admin password: " GPW; echo
kubectl create secret generic grafana-admin -n heart-mlops --from-literal=password="$GPW"
kubectl create configmap grafana-dashboards -n heart-mlops \
  --from-file=monitoring/grafana/heart-api-dashboard.json
kubectl apply -f monitoring/k8s/prometheus.yaml -f monitoring/k8s/grafana.yaml
kubectl get pods -n heart-mlops
```
Port-forwards (each in its own terminal, or append `&`):
```bash
kubectl port-forward -n heart-mlops svc/heart-api-svc 8080:80 &
kubectl port-forward -n heart-mlops svc/prometheus 9090:9090 &
kubectl port-forward -n heart-mlops svc/grafana 3000:3000 &
python scripts/generate_traffic.py --url http://localhost:8080 -n 300
```
* Prometheus http://localhost:9090 -> Status -> Targets: `heart-api` shows one
  target per pod, all UP (screenshot 18). Try queries:
  `sum by (status) (rate(heart_api_requests_total[1m]))`,
  `heart_api_validation_errors_total`.
* Grafana http://localhost:3000 (admin / your password) -> Dashboards ->
  "Heart Disease MLOps" -> "Heart Disease API - Operations" (screenshot 19).
CHECKPOINT: panels show request rate, p95 latency, class counts, 422s.

Stop port-forwards: `kill %1 %2 %3` (or `pkill -f "kubectl port-forward"`).

---------------------------------------------------------------------------
## Phase 7 - Finish

```bash
git add . && git commit -m "Deployment, monitoring, docs and screenshots" && git push
```
* Report (10 pages): use your real numbers/screenshots + docs/decisions.md.
* Video: docs/video_checklist.md.
* Cleanup when finished: `kubectl delete namespace heart-mlops && minikube stop`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `externally-managed-environment` | you are outside the venv: `source scripts/session_setup.sh` |
| `ModuleNotFoundError: heartml` | `pip install -e .` inside the venv |
| push rejected for workflow file | token needs `workflow` scope |
| pod `ErrImageNeverPull`/`ImagePullBackOff` | run `minikube image load heart-disease-api:1.0.0`; tag must match manifest |
| pod `CrashLoopBackOff` | `kubectl logs <pod> -n heart-mlops`; usually missing model in image -> train, rebuild, reload |
| pod Running but not READY | `/ready` returns 503 -> model not loaded; check logs |
| Grafana pod `CreateContainerConfigError` | secret `grafana-admin` or configmap `grafana-dashboards` missing |
| Prometheus target DOWN | `kubectl get endpoints heart-api-headless -n heart-mlops` must list pod IPs |
| `InconsistentVersionWarning` on load | train and serve with the same scikit-learn (1.5.2 in both requirement files) |
| minikube `RSRC_INSUFFICIENT` | `--memory=3072` |
