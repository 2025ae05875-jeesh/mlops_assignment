# Engineering decisions log (write in YOUR OWN words)

Fill every section after you have run the step and looked at the output.
These explanations are what make the submission yours.

## Data
- Why the Cleveland subset / why not Kaggle `heart.csv`:
- Target encoding (num 0 vs 1-4) and its consequence:
- Why missing values are imputed inside the pipeline instead of before the split:
- Duplicates found / action:

## EDA observations (one or two sentences per figure, from YOUR plots)
- 01 class balance:
- 02 missing values:
- 03 numeric histograms:
- 04 boxplots / outliers:
- 05 categorical disease rate:
- 06 correlation heatmap (and why Pearson on nominal codes is only indicative):

## Features and models
- Feature grouping (numeric / binary / nominal) and why `ca` is numeric:
- Why scaling matters for LR but not RF:
- Grid chosen for each model and why:
- Selection rule (CV ROC-AUC, test set used once) and why:
- Champion and justification (use YOUR numbers):
- False negatives vs false positives in this context; threshold choice:

## MLOps
- MLflow file store vs SQLite on the EFS network drive:
- Why the whole pipeline (not only the estimator) is saved; pickle security note:
- CI design: which steps, what makes it fail, the quality gate:
- Image delivery: minikube image load vs GHCR:
- Probes, replicas, resources:
- What is logged and what is deliberately NOT logged (patient data):
- Prometheus metrics chosen and what each tells you:

## Limitations / future work
-
