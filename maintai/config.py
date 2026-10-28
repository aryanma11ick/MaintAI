import os
from pathlib import Path

ROOT = Path(os.getenv("MAINTAI_HOME", "runtime")).resolve()
SEED = 42
THRESHOLD = 0.35
WINDOW = 500
MIN_WINDOW = 100
MIN_LABELS = 300
MIN_POSITIVES = 12
COOLDOWN_SECONDS = int(os.getenv("MAINTAI_COOLDOWN_SECONDS", "600"))
WORKER_SECONDS = int(os.getenv("MAINTAI_WORKER_SECONDS", "15"))
LABEL_DELAY_SECONDS = int(os.getenv("MAINTAI_LABEL_DELAY_SECONDS", "30"))
DEMO_ENABLED = os.getenv("MAINTAI_DEMO_ENABLED", "true").lower() == "true"
FEATURES = ["type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
NUMERIC = FEATURES[1:]
TARGETS = ["failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_NAMES = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power", "OSF": "Overstrain", "RNF": "Random failure"}

def ensure_dirs():
    for name in ("data", "models", "reports", "artifacts"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
