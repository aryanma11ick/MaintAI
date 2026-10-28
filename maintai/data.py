import hashlib
import io
import json
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config as c
from .store import sha256

SOURCE = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
RENAME = {"UDI":"source_id", "Type":"type", "Air temperature [K]":"air_temperature", "Process temperature [K]":"process_temperature", "Rotational speed [rpm]":"rotational_speed", "Torque [Nm]":"torque", "Tool wear [min]":"tool_wear", "Machine failure":"failure"}

def prepare(csv=None):
    c.ensure_dirs()
    manifest_path = c.ROOT / "data/manifest.json"
    if manifest_path.exists():
        verify_splits()
        return json.loads(manifest_path.read_text())
    if csv:
        raw = open(csv, "rb").read()
    else:
        bundled = __import__('pathlib').Path(__file__).resolve().parent.parent / 'data/ai4i2020.csv'
        if bundled.exists():
            raw = bundled.read_bytes()
        else:
            with urllib.request.urlopen(SOURCE, timeout=60) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    raw = z.read("ai4i2020.csv")
    frame = pd.read_csv(io.BytesIO(raw)).rename(columns=RENAME)
    needed = ['source_id'] + c.FEATURES + c.TARGETS
    if not set(needed).issubset(frame.columns):
        raise ValueError("Expected the original UCI AI4I CSV, including UDI and failure labels")
    frame = frame[needed]
    if frame.isna().any().any() or frame.source_id.duplicated().any():
        raise ValueError("Missing values or duplicate source IDs")
    if not frame['type'].isin(['L','M','H']).all():
        raise ValueError("Invalid product type")
    if not frame[c.TARGETS].isin([0,1]).all().all():
        raise ValueError("Failure labels must be binary")
    train, rest = train_test_split(frame,test_size=.4,stratify=frame.failure,random_state=c.SEED)
    validation, rest = train_test_split(rest,test_size=.625,stratify=rest.failure,random_state=c.SEED)
    stream, test = train_test_split(rest,test_size=.4,stratify=rest.failure,random_state=c.SEED)
    manifest = {'source':SOURCE,'license':'CC BY 4.0 (UCI distribution)','source_sha256':hashlib.sha256(raw).hexdigest(),'seed':c.SEED,'splits':{}}
    for name, part in [('train',train),('validation',validation),('stream',stream),('test',test)]:
        path=c.ROOT/'data'/f'{name}.csv'
        part.to_csv(path,index=False)
        manifest['splits'][name]={'rows':len(part),'positives':int(part.failure.sum()),'sha256':sha256(path)}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    return manifest

def verify_splits():
    manifest=json.loads((c.ROOT/'data/manifest.json').read_text())
    for name,meta in manifest['splits'].items():
        if sha256(c.ROOT/'data'/f'{name}.csv') != meta['sha256']:
            raise ValueError(f"Data integrity check failed: {name}")

def read_split(name):
    if name not in ('train','validation','stream','test'):
        raise ValueError('Unknown split')
    return pd.read_csv(c.ROOT/'data'/f'{name}.csv')

def matrix(frame):
    """Explicit allowlist: identifiers and all six targets are never predictors."""
    x=frame[c.NUMERIC].astype(float).copy()
    x['type_L']=(frame['type']=='L').astype(float)
    x['type_M']=(frame['type']=='M').astype(float)
    x['type_H']=(frame['type']=='H').astype(float)
    x['temperature_gap']=x.process_temperature-x.air_temperature
    x['power_watts']=x.torque*x.rotational_speed*(2*3.141592653589793/60)
    x['wear_torque']=x.tool_wear*x.torque
    return x
