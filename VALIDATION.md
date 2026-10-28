# Validation record

Verified on 24 October 2026 using Python 3.12.14 in the build environment.

## Automated tests

`python -m pytest -q`: **18 passed** in 36.61 seconds.

Covered: disjoint splits and target exclusion; tampered-data refusal; API input
validation; prediction/SHAP consistency; immutable feedback; delayed labels;
unique replay records; drift-window freshness; missing-label safeguards; class
coverage; promotion gates; real retraining; permanent evaluation reservations;
candidate rejection; deployment activation; manual rollback; automatic rollback
after corruption; post-activation failure recovery; refusal when no valid model
exists; MLflow artifacts and cross-process tracking; dashboard interaction.

Some deployment tests deliberately inject passing/failing gate results to
exercise recovery branches. They do not provide evidence of model accuracy.
Real gate decisions are tested independently and in the live exercise below.

## Live local exercise

- Started API, worker and Streamlit through `python run_local.py`.
- API readiness and Streamlit health endpoints responded successfully.
- Exercised the dashboard's prediction form using Streamlit AppTest.
- Replayed two distinct batches of 250 reserved high-wear records.
- Used 2-second worker/label intervals and a zero cooldown for this verification.
  Packaged defaults remain 15 seconds, 30 seconds and 600 seconds respectively.
- Worker received delayed labels, trained a candidate and automatically promoted
  it after all eight gates passed. No manual gate override was used.
- MLflow logged the baseline and candidate across separate processes without errors.
- The launcher shut down cleanly with exit code 0.

| Metric | Original model | Candidate |
|---|---:|---:|
| Fixed validation average precision | 0.8915 | 0.8837 |
| Fixed validation recall | 0.8627 | 0.8627 |
| Fixed validation precision | 0.8800 | 0.9362 |
| Recent holdout average precision | 0.6858 | 0.7173 |
| Recent holdout recall | 0.6667 | 0.6667 |
| Recent holdout precision | 0.7500 | 1.0000 |

The fixed validation set had 1,500 records / 51 failures. The recent holdout
had 175 records / 9 failures. Its small failure count does not establish a
statistically significant or generalizable improvement. Promotion followed the
documented prototype thresholds; fixed-validation AP fell slightly within its
allowed tolerance. The final test split was not evaluated for these decisions.

## Limits of verification

- Native Python services were executed in the build environment, not on the user's laptop.
- Windows commands are provided, but Windows itself was not available for execution.
- Docker and AWS deployment were not executed.
- Dashboard elements and interactions were checked with AppTest. Browser-level
  screenshot inspection was unavailable because the browser download failed.
- A non-failing upstream TestClient deprecation warning remains in the test output.

Dependencies are pinned. A fresh environment should install `requirements.txt`
before loading newly trained artifacts. Generated runtime/model files are not
included in the ZIP; the first launch creates them locally from the bundled data.
