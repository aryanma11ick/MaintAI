import json
import time
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from . import config as c
from .data import matrix
from . import store

def metrics(y, scores):
    y=np.asarray(y,dtype=int)
    scores=np.asarray(scores,dtype=float)
    pred=(scores>=c.THRESHOLD).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'average_precision':float(average_precision_score(y,scores)) if y.sum() else 0.0,
            'recall':float(recall_score(y,pred,zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'rows':len(y),'positives':int(y.sum())}

def scores(bundle, frame):
    model=bundle['binary']
    if 1 not in model.classes_:
        return np.zeros(len(frame))
    return model.predict_proba(matrix(frame))[:,list(model.classes_).index(1)]

def evaluate(bundle, frame):
    return metrics(frame.failure,scores(bundle,frame))

def train(frame, validation, candidate=False):
    x=matrix(frame)
    params={'n_estimators':160 if candidate else 100,'min_samples_leaf':1 if candidate else 2,'class_weight':'balanced_subsample','random_state':c.SEED,'n_jobs':2}
    binary=RandomForestClassifier(**params).fit(x,frame.failure)
    types=RandomForestClassifier(**params).fit(x,frame[c.TARGETS[1:]])
    version=time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    bundle={'binary':binary,'types':types,'version':version,'features':list(x.columns),'threshold':c.THRESHOLD}
    result=evaluate(bundle,validation)
    path=c.ROOT/'models'/f'{version}.joblib'
    joblib.dump(bundle,path)
    with store.db() as con:
        con.execute('INSERT INTO models VALUES (?,?,?,?,?,?)',(version,str(path.relative_to(c.ROOT)),store.sha256(path),json.dumps(result),time.time(),'candidate'))
    tracking='logged'
    try:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///'+str(c.ROOT/'mlflow.db'))
        if mlflow.get_experiment_by_name('MaintAI') is None:
            mlflow.create_experiment('MaintAI',artifact_location=(c.ROOT/'artifacts').as_uri())
        mlflow.set_experiment('MaintAI')
        with mlflow.start_run(run_name=version):
            mlflow.log_params({**params,'training_rows':len(frame),'threshold':c.THRESHOLD,'candidate':candidate})
            mlflow.log_metrics(result)
            mlflow.set_tag('dataset_manifest_sha256',store.sha256(c.ROOT/'data/manifest.json'))
            mlflow.log_dict({'training_source_ids':sorted(frame.source_id.astype(int).tolist())},'training_ids.json')
            mlflow.log_artifact(str(c.ROOT/'data/manifest.json'))
            mlflow.log_artifact(str(path),'models')
    except Exception as exc:
        tracking='failed'
        store.event('tracking_error',{'error':str(exc),'version':version})
    store.event('model_trained',{'version':version,'metrics':result,'training_rows':len(frame),'mlflow':tracking})
    return bundle,result

def load(version=None):
    if version is None:
        record=store.active_record()
    else:
        with store.db() as con:
            row=con.execute('SELECT * FROM models WHERE version=?',(version,)).fetchone()
        if row is None: raise ValueError('Unknown model')
        record=dict(row)
    path=c.ROOT/record['path']
    if store.sha256(path)!=record['digest']:
        raise RuntimeError('Model artifact failed its SHA-256 integrity check')
    # Only trusted, locally trained files are accepted; no pickle upload endpoint.
    return joblib.load(path)

def smoke(bundle, frame):
    p=scores(bundle,frame.head(10))
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise RuntimeError('Model smoke test failed')
    if list(matrix(frame).columns)!=bundle['features']:
        raise RuntimeError('Feature schema mismatch')

def explain(bundle, frame):
    import shap
    x=matrix(frame)
    ex=shap.TreeExplainer(bundle['binary'])(x)
    values=ex.values[0,:,1]
    base=float(ex.base_values[0,1])
    items=[{'feature':name,'value':float(x.iloc[0][name]),'contribution':float(v)} for name,v in zip(x.columns,values)]
    return {'base_score':base,'contributions':sorted(items,key=lambda r:abs(r['contribution']),reverse=True),'meaning':'SHAP contributions explain model output, not physical causes.'}

def describe(bundle, frame, with_explanation=True):
    p=float(scores(bundle,frame)[0])
    type_model=bundle['types']
    raw=type_model.predict_proba(matrix(frame))
    type_scores={}
    for name, classes, prob in zip(c.TARGETS[1:],type_model.classes_,raw):
        type_scores[name]=float(prob[0,list(classes).index(1)]) if 1 in classes else 0.0
    recommendations=[]
    if p>=c.THRESHOLD:
        recommendations.append('Review this machine with a maintenance engineer; this is a model flag, not a confirmed fault.')
    if type_scores['TWF']>=.2: recommendations.append('Inspect tool wear and replacement history.')
    if type_scores['HDF']>=.2: recommendations.append('Inspect cooling and heat dissipation.')
    if type_scores['PWF']>=.2: recommendations.append('Review power and torque readings against the equipment specification.')
    if type_scores['OSF']>=.2: recommendations.append('Inspect load and tool wear together for possible overstrain.')
    if not recommendations: recommendations.append('Continue routine monitoring; low model score does not guarantee normal equipment condition.')
    return {'model_version':bundle['version'],'failure_score':p,'failure_flag':p>=c.THRESHOLD,'threshold':c.THRESHOLD,
            'health_indicator':round(100*(1-p),1),'health_note':'Heuristic 100 × (1 − model score), not calibrated health or remaining life.',
            'failure_type_scores':type_scores,'recommendations':recommendations,
            'explanation':explain(bundle,frame) if with_explanation else None}
