import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from maintai import config as c, data, model, pipeline, store
from maintai.api import app

@pytest.fixture(scope='session')
def baseline(tmp_path_factory):
    previous=c.ROOT
    c.ROOT=tmp_path_factory.mktemp('baseline')
    pipeline.bootstrap()
    root=c.ROOT
    c.ROOT=previous
    return root

@pytest.fixture
def runtime(tmp_path,monkeypatch,baseline):
    root=tmp_path/'runtime'
    shutil.copytree(baseline,root)
    monkeypatch.setattr(c,'ROOT',root)
    monkeypatch.setattr(c,'LABEL_DELAY_SECONDS',0)
    monkeypatch.setattr(c,'COOLDOWN_SECONDS',0)
    return root

def seed_drift():
    pipeline.replay(250,'high-wear')
    first=pipeline.tick()
    pipeline.replay(250,'high-wear')
    return first

def test_split_disjointness_and_feature_allowlist(runtime):
    parts=[data.read_split(x) for x in ['train','validation','stream','test']]
    assert sum(map(len,parts))==10000
    for i,a in enumerate(parts):
        for b in parts[i+1:]: assert not set(a.source_id)&set(b.source_id)
    x=data.matrix(parts[0])
    assert not set(c.TARGETS+['source_id','Product ID'])&set(x.columns)
    changed=parts[0].copy()
    changed[c.TARGETS]=1-changed[c.TARGETS]
    pd.testing.assert_frame_equal(x,data.matrix(changed))

def test_tampered_data_is_refused(runtime):
    path=runtime/'data/train.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='integrity'): pipeline.tick()

def test_api_validation_prediction_and_explanation(runtime):
    with TestClient(app) as client:
        assert client.get('/health/ready').status_code==200
        assert client.post('/predict',json={'torque':-1}).status_code==422
        assert client.post('/predict',json={'failure':1}).status_code==422
        assert client.post('/predict',json={'type':'Z'}).status_code==422
        r=client.post('/predict',json={'torque':65,'tool_wear':230}).json()
        assert 0<=r['failure_score']<=1
        explanation=r['explanation']
        assert explanation['base_score']+sum(x['contribution'] for x in explanation['contributions'])==pytest.approx(r['failure_score'],abs=1e-6)
        assert client.get('/status').json()['counts']['predictions']==1

def test_feedback_is_immutable_and_idempotent(runtime):
    with TestClient(app) as client:
        pred=client.post('/predict?explain=false',json={}).json()
        url='/feedback/'+pred['prediction_id']
        assert client.post(url,json={'failure':1,'TWF':1}).status_code==200
        assert client.post(url,json={'failure':1,'TWF':1}).json()['status']=='already_recorded'
        assert client.post(url,json={'failure':0}).status_code==400
        assert client.post('/feedback/not-found',json={'failure':0}).status_code==400

def test_delayed_labels_and_unique_replay(runtime,monkeypatch):
    monkeypatch.setattr(c,'LABEL_DELAY_SECONDS',3600)
    pipeline.replay(250,'high-wear')
    assert pipeline.release_labels()==0
    assert pipeline.status()['counts']['labelled']==0
    pipeline.replay(250,'high-wear')
    with store.db() as con:
        assert con.execute('SELECT COUNT(DISTINCT source_id) FROM predictions').fetchone()[0]==500
        con.execute('UPDATE predictions SET label_due=?',(time.time()-1,))
    assert pipeline.release_labels()==500
    assert pipeline.release_labels()==0

def test_same_data_cannot_build_a_drift_streak(runtime):
    pipeline.replay(250,'high-wear')
    a=pipeline.monitor()
    b=pipeline.monitor()
    assert a['drift_streak']==b['drift_streak']==1
    assert b['trigger'] is False

def test_drift_without_labels_never_retrains(runtime,monkeypatch):
    monkeypatch.setattr(c,'LABEL_DELAY_SECONDS',3600)
    pipeline.replay(250,'high-wear');pipeline.tick()
    pipeline.replay(250,'high-wear')
    assert pipeline.tick()['status']=='waiting_for_labels'
    assert len(pipeline.status()['models'])==1

def test_label_coverage_required(runtime):
    pipeline.replay(100,'normal');pipeline.release_labels()
    assert pipeline.retrain()['status']=='waiting_for_labels'

def test_quality_gates_reject_regression():
    good={'recall':.9,'precision':.8,'average_precision':.8}
    better={'recall':.9,'precision':.8,'average_precision':.85}
    assert pipeline.gates(good,good,good,better)['passed']
    bad={**better,'recall':.5}
    assert not pipeline.gates(good,good,good,bad)['passed']
    assert not pipeline.gates(good,good,good,good)['passed']

def test_real_retraining_and_holdout_reservation(runtime):
    seed_drift()
    result=pipeline.tick()
    assert result['status'] in ('promoted','rejected')
    assert result['decision']['passed']==(result['status']=='promoted')
    trained=set(store.get('training_ids'))
    held=set(result['recent_evaluation_ids'])
    assert trained and held and not trained&held
    assert pipeline.status()['counts']['unused_labels']==0
    assert len(pipeline.status()['models'])==2
    assert pipeline.tick()['status']=='waiting_for_labels'
    assert not any(e['kind']=='tracking_error' for e in store.recent_events(100))

def test_rejected_candidate_does_not_replace_champion(runtime,monkeypatch):
    seed_drift()
    old=store.get('active_model')
    monkeypatch.setattr(pipeline,'gates',lambda *a:{'passed':False,'checks':{'injected_rejection':False}})
    assert pipeline.tick()['status']=='rejected'
    assert store.get('active_model')==old
    assert pipeline.status()['models'][0]['status']=='rejected'

def test_promotion_and_manual_rollback_mechanics(runtime,monkeypatch):
    seed_drift()
    old=store.get('active_model')
    # Isolate deployment mechanics; real gate outcomes are exercised above.
    monkeypatch.setattr(pipeline,'gates',lambda *a:{'passed':True,'checks':{'injected_pass':True}})
    result=pipeline.tick()
    assert result['status']=='promoted'
    assert store.get('active_model')!=old
    assert model.load()['version']==result['candidate']
    assert pipeline.rollback()['model']==old

def test_corrupt_candidate_automatically_rolls_back(runtime,monkeypatch):
    seed_drift()
    old=store.get('active_model')
    monkeypatch.setattr(pipeline,'gates',lambda *a:{'passed':True,'checks':{}})
    pipeline.tick()
    record=store.active_record()
    (runtime/record['path']).write_bytes(b'corrupted for fault injection')
    with TestClient(app) as client: assert client.get('/health/ready').status_code==503
    pipeline.tick()
    assert store.get('active_model')==old
    assert model.load()['version']==old
    with store.db() as con:
        assert con.execute('SELECT status FROM models WHERE version=?',(record['version'],)).fetchone()[0]=='broken'
    with pytest.raises(ValueError): pipeline.rollback()

def test_failed_post_activation_smoke_restores_champion(runtime,monkeypatch):
    seed_drift()
    old=store.get('active_model')
    original=model.smoke
    def fail_after_activation(bundle,frame):
        if store.get('active_model')!=old: raise RuntimeError('Injected deployment fault')
        return original(bundle,frame)
    monkeypatch.setattr(pipeline,'gates',lambda *a:{'passed':True,'checks':{}})
    monkeypatch.setattr(model,'smoke',fail_after_activation)
    result=pipeline.tick()
    assert result['status']=='rolled_back'
    assert store.get('active_model')==old

def test_recovery_refuses_to_invent_a_model(runtime):
    record=store.active_record()
    (runtime/record['path']).write_bytes(b'broken initial model')
    with pytest.raises(RuntimeError,match='no predecessor'): pipeline.tick()

def test_mlflow_run_and_artifact_are_present(runtime):
    import sqlite3
    with sqlite3.connect(runtime/'mlflow.db') as con:
        assert con.execute('SELECT COUNT(*) FROM runs').fetchone()[0]>=1
    assert store.active_record()['digest']==store.sha256(runtime/store.active_record()['path'])

def test_mlflow_survives_separate_worker_process(runtime):
    import os
    import subprocess
    import sys
    code='from maintai import data,model; model.train(data.read_split("train"),data.read_split("validation"),candidate=True)'
    result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'MAINTAI_HOME':str(runtime)},capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert not [e for e in store.recent_events(100) if e['kind']=='tracking_error']

def test_dashboard_renders_and_predicts(runtime,monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest
    with TestClient(app) as client:
        def local_request(method,url,**kwargs):
            return client.request(method,'/'+url.split('/',3)[3],json=kwargs.get('json'))
        monkeypatch.setattr(httpx,'request',local_request)
        view=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'maintai/dashboard.py'),default_timeout=40).run()
        assert not view.exception
        assert len(view.tabs)==4
        view.button[0].click().run()
        assert not view.exception
        assert any(m.label=='Failure score' for m in view.metric)
