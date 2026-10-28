# Run MaintAI on your laptop

This package contains the local application, the original UCI AI4I CSV, tests,
and Docker configuration. No AWS account, API key, paid service, or GPU is needed.
The first installation needs internet access to download Python packages.
The running application uses local data and local files.

## Windows: recommended first run

1. Install **Python 3.12 (64-bit)** if it is not installed. Select “Add Python to PATH”.
2. Extract the ZIP. Open the extracted `maintai` folder in VS Code.
3. Open a PowerShell terminal in that folder. Run these commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_local.py
```

You do not need to activate the environment or change PowerShell's execution policy.
If the `py` launcher is unavailable, use `python -m venv .venv`, after checking
that `python --version` reports 3.12.

Open **http://localhost:8501** for the dashboard.
Open **http://localhost:8000/docs** for interactive API documentation.
The first run trains the baseline model. Allow the services a few moments to start.

Keep this terminal open. Press **Ctrl+C** to stop the API, dashboard and worker.
To start again, run only:

```powershell
.\.venv\Scripts\python.exe run_local.py
```

## macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_local.py
```

## Try the automatic recovery loop

1. Open **Recovery lab** and choose **high-wear**, **250 readings**.
2. Click **Replay batch**. Go to Overview and wait for the worker to report one changed window.
3. Wait at least 15 seconds before replaying a second batch of 250 high-wear readings.
4. Labels arrive 30 seconds after each replay. The worker checks every 15 seconds.
5. Once there are enough labelled failures and two changed windows, it trains a candidate.
6. Inspect the recovery decision, gate results, model history and audit trail.

A **rejected** candidate is a successful demonstration of the validation gate:
the current model stays active. The system never forces promotion just to make
the demo look successful. Promotion and automatic rollback are separately
covered by fault-injection tests.

Readings used for replay are separate from baseline training and validation.
Higher-wear records are selected without modifying their values or inventing labels.
The dataset is finite: 1,500 unique replay records are available per runtime.

## A clean experiment

Stop all services first. **Rename** the `runtime` folder to an unused name such as
`runtime_previous`, then start again. This preserves your previous experiment.
Do not delete the `data` folder: it contains the original CSV and attribution.

## Where to look when something fails

- Dependency installation fails: confirm Python 3.12 and an internet connection.
- Port 8000 or 8501 is occupied: stop the previous application using that port.
- Dashboard reports API unavailable: inspect `runtime/logs/api.log`.
- Worker stops updating: inspect `runtime/logs/worker.log`.
- Waiting for labels: allow the 30-second demo delay and one worker cycle.
- Waiting for class coverage: replay more unused records; supervised training needs both classes.
- No previous model: rollback becomes available after a second model has been activated.
- Model checksum fails: a valid predecessor will be restored by the worker. With no predecessor,
  the API refuses readiness. Preserve the damaged runtime and start a fresh experiment.

## Docker, when you are ready

Install and start Docker Desktop (or Docker Engine + Compose on Linux), then run:

```bash
docker compose up --build
```

Use the same dashboard and API URLs. Initial training runs in the `init` service.
The named Docker volume keeps state between restarts. Stop the stack with:

```bash
docker compose down
```

Do not add `-v` unless you intend to delete all state in the Docker volume.
Docker and native Python use separate state locations by default.

The supplied Compose setup is prepared for local use. It has not been executed
in the build environment because that environment does not provide Docker.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use temporary directories and do not reset your working runtime.
See `README.md` for architecture, scope, commands and the later AWS migration.
