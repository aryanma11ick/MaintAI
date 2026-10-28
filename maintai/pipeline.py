"""Drift -> labelled evidence -> candidate -> gates -> atomic activation/rollback."""
import json
import time
import uuid
import numpy as np
import pandas as pd
from filelock import FileLock, Timeout
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from . import config as c, data, model, store

def lock():
    c.ensure_dirs()
    return FileLock(str(c.ROOT/'pipeline.lock'),timeout=1)

def bootstrap(csv=None):
    store.init_db()
    with lock():
        manifest=data.prepare(csv)
        if store.get('active_model'):
            model.smoke(model.load(),data.read_split('validation'))
            return {'status':'already_initialized','model':store.get('active_model'),'dataset':manifest}
        bundle,metrics=model.train(data.read_split('train'),data.read_split('validation'))
        model.smoke(bundle,data.read_split('validation'))
        store.activate(bundle['version'],'Initial baseline')
        store.put('training_ids',[])
        store.put('worker_heartbeat',None)
        store.event('initialized',manifest)
        return {'status':'initialized','model':bundle['version'],'validation_metrics':metrics,'dataset':manifest}

def record_prediction(features, bundle=None, with_explanation=False):
    bundle=bundle or model.load()
    frame=pd.DataFrame([features])
    result=model.describe(bundle,frame,with_explanation)
    prediction_id=uuid.uuid4().hex
    with store.db() as con:
        con.execute('INSERT INTO predictions(id,ts,features,score,predicted,model,scenario) VALUES (?,?,?,?,?,?,?)',
                    (prediction_id,time.time(),json.dumps(features),result['failure_score'],int(result['failure_flag']),bundle['version'],'manual'))
    return {'prediction_id':prediction_id,**result}

def feedback(prediction_id, labels):
    with store.db() as con:
        row=con.execute('SELECT labels,scenario FROM predictions WHERE id=?',(prediction_id,)).fetchone()
        if row is None: raise ValueError('Prediction not found')
        if row['scenario']!='manual': raise ValueError('Demo labels come only from the reserved dataset')
        if row['labels'] is not None:
            if json.loads(row['labels'])==labels: return {'status':'already_recorded'}
            raise ValueError('Labels are immutable once submitted')
        con.execute('UPDATE predictions SET labels=? WHERE id=?',(json.dumps(labels),prediction_id))
    store.event('feedback_received',{'prediction_id':prediction_id})
    return {'status':'recorded'}

def replay(count=250, scenario='normal'):
    if not c.DEMO_ENABLED: raise ValueError('Demo endpoints are disabled')
    if scenario not in ('normal','high-wear'): raise ValueError('Unknown scenario')
    with lock():
        data.verify_splits()
        pool=data.read_split('stream')
        with store.db() as con:
            used={r[0] for r in con.execute('SELECT source_id FROM predictions WHERE source_id IS NOT NULL')}
        pool=pool[~pool.source_id.isin(used)]
        # Selection bias changes the input distribution without editing features or labels.
        if scenario=='high-wear':
            batch=pool.sort_values('tool_wear',ascending=False).head(count)
        else:
            batch=pool.sample(n=min(count,len(pool)),random_state=c.SEED+len(used))
        if batch.empty: return {'status':'exhausted','inserted':0,'remaining':0}
        bundle=model.load()
        predictions=model.scores(bundle,batch)
        now=time.time()
        with store.db() as con:
            for (_,row),p in zip(batch.iterrows(),predictions):
                features={k:row[k] for k in c.FEATURES}
                labels={k:int(row[k]) for k in c.TARGETS}
                con.execute('INSERT INTO predictions(id,ts,features,score,predicted,model,source_id,scenario,pending_labels,label_due) VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (uuid.uuid4().hex,now,json.dumps(features),float(p),int(p>=c.THRESHOLD),bundle['version'],int(row.source_id),scenario,json.dumps(labels),now+c.LABEL_DELAY_SECONDS))
        result={'status':'replayed','inserted':len(batch),'remaining':len(pool)-len(batch),'scenario':scenario,'labels_available_after_seconds':c.LABEL_DELAY_SECONDS}
        store.event('demo_replayed',result)
        return result

def release_labels():
    with store.db() as con:
        cursor=con.execute('UPDATE predictions SET labels=pending_labels,pending_labels=NULL WHERE labels IS NULL AND pending_labels IS NOT NULL AND label_due<=?',(time.time(),))
        count=cursor.rowcount
    if count: store.event('demo_labels_arrived',{'count':count})
    return count

def drift_report(reference, observed):
    details={}
    # Bonferroni correction + a minimum effect size; not p-values alone.
    for feature in c.NUMERIC:
        stat,p=ks_2samp(reference[feature],observed[feature])
        details[feature]={'statistic':float(stat),'p_value':float(p),'drifted':bool(stat>=.15 and p<.01/len(c.NUMERIC))}
    ref=reference['type'].value_counts(normalize=True)
    obs=observed['type'].value_counts(normalize=True)
    distance=float(sum(abs(ref.get(t,0)-obs.get(t,0)) for t in ['L','M','H'])/2)
    details['type']={'total_variation':distance,'drifted':distance>=.15}
    return {'drifted':any(x['drifted'] for x in details.values()),'features':details,'rows':len(observed)}

def monitor():
    record=store.active_record()
    with store.db() as con:
        rows=con.execute('SELECT rowid AS seq,* FROM predictions ORDER BY rowid DESC LIMIT ?',(c.WINDOW,)).fetchall()
    if len(rows)<c.MIN_WINDOW:
        result={'status':'waiting_for_readings','readings':len(rows),'required':c.MIN_WINDOW,'trigger':False}
        store.put('monitor',result)
        return result
    observed=pd.DataFrame([json.loads(r['features']) for r in rows])
    report=drift_report(data.read_split('train'),observed)
    state=store.get('drift_state',{'seq':0,'streak':0})
    seq=rows[0]['seq']
    if seq-state['seq']>=c.MIN_WINDOW:
        state={'seq':seq,'streak':state['streak']+1 if report['drifted'] else 0}
        store.put('drift_state',state)
    labelled=[r for r in rows if r['labels'] is not None and r['model']==record['version']]
    quality=None
    degradation=False
    if len(labelled)>=c.MIN_WINDOW:
        y=[json.loads(r['labels'])['failure'] for r in labelled]
        if sum(y)>=5 and len(y)-sum(y)>=5:
            quality=model.metrics(y,[r['score'] for r in labelled])
            baseline=json.loads(record['metrics'])
            degradation=quality['recall']<max(.5,baseline['recall']-.10)
    trigger=state['streak']>=2 or degradation
    result={'status':'attention' if trigger else 'monitoring','drift':report,'drift_streak':state['streak'],
            'quality':quality,'quality_degradation':bool(degradation),'trigger':bool(trigger),'model':record['version']}
    store.put('monitor',result)
    return result

def gates(champion_base, candidate_base, champion_recent, candidate_recent):
    """No claim of statistical significance: these are deterministic demo gates."""
    checks={
      'baseline_recall_floor':candidate_base['recall']>=.65,
      'baseline_precision_floor':candidate_base['precision']>=.30,
      'baseline_recall_non_regression':candidate_base['recall']>=champion_base['recall']-.03,
      'baseline_ap_non_regression':candidate_base['average_precision']>=champion_base['average_precision']-.02,
      'recent_recall_floor':candidate_recent['recall']>=.65,
      'recent_precision_floor':candidate_recent['precision']>=.30,
      'recent_recall_non_regression':candidate_recent['recall']>=champion_recent['recall']-.03,
      'recent_ap_improvement':candidate_recent['average_precision']>=champion_recent['average_precision']+.002,
    }
    return {'passed':all(checks.values()),'checks':checks}

def labelled_frame(rows):
    return pd.DataFrame([{**json.loads(r['features']),**json.loads(r['labels']),
        'prediction_id':r['id'],'source_id':r['source_id'] if r['source_id'] is not None else -int(r['id'][:12],16)} for r in rows])

def recover_active():
    """Roll back a corrupt/unloadable active artifact to a verified predecessor."""
    try:
        active=store.active_record()
        model.smoke(model.load(),data.read_split('validation'))
        return False
    except Exception as exc:
        previous=store.get('previous_model')
        if not previous:
            raise RuntimeError('No valid active model and no predecessor to restore') from exc
        prior=model.load(previous)
        model.smoke(prior,data.read_split('validation'))
        broken=store.get('active_model')
        with store.db() as con:
            con.execute("UPDATE models SET status='broken' WHERE version=?",(broken,))
        store.activate(previous,'Automatic rollback: '+str(exc),expected=broken)
        store.event('automatic_rollback',{'from':broken,'to':previous,'reason':str(exc)})
        return True

def rollback():
    with lock():
        previous=store.get('previous_model')
        if not previous: raise ValueError('No previous model is available')
        with store.db() as con:
            status=con.execute('SELECT status FROM models WHERE version=?',(previous,)).fetchone()
        if not status or status[0] in ('broken','rejected'): raise ValueError('Previous model is not usable')
        model.smoke(model.load(previous),data.read_split('validation'))
        store.activate(previous,'Manual rollback')
        return {'status':'rolled_back','model':previous}

def retrain():
    champion=model.load()
    with store.db() as con:
        rows=con.execute('SELECT * FROM predictions WHERE labels IS NOT NULL AND consumed=0 ORDER BY rowid').fetchall()
        prior_ids=set(store.get('training_ids',[]))
        prior_rows=con.execute('SELECT * FROM predictions WHERE labels IS NOT NULL AND consumed=1').fetchall()
    if len(rows)<c.MIN_LABELS:
        return {'status':'waiting_for_labels','available':len(rows),'required':c.MIN_LABELS}
    new=labelled_frame(rows)
    counts=new.failure.value_counts()
    if min(counts.get(0,0),counts.get(1,0))<c.MIN_POSITIVES:
        return {'status':'waiting_for_class_coverage','counts':{str(k):int(v) for k,v in counts.items()},'minimum_per_class':c.MIN_POSITIVES}
    if time.time()-store.get('last_training',0)<c.COOLDOWN_SECONDS:
        return {'status':'cooldown','seconds':c.COOLDOWN_SECONDS}
    added,recent=train_test_split(new,test_size=.35,stratify=new.failure,random_state=c.SEED)
    if min(recent.failure.value_counts().get(0,0),recent.failure.value_counts().get(1,0))<4:
        return {'status':'waiting_for_evaluation_coverage'}
    base=data.read_split('train')
    old=[r for r in prior_rows if r['id'] in prior_ids]
    pieces=[base,added]
    if old: pieces.append(labelled_frame(old))
    training=pd.concat(pieces,ignore_index=True).drop_duplicates('source_id')
    validation=data.read_split('validation')
    if set(training.source_id)&(set(validation.source_id)|set(recent.source_id)):
        raise RuntimeError('Training/evaluation overlap detected')
    # Reserve roles BEFORE training. If this process crashes after activation,
    # evaluation rows still cannot leak into a later candidate's training set.
    with store.db() as con:
        con.executemany('UPDATE predictions SET consumed=1 WHERE id=?',[(r['id'],) for r in rows])
        con.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',('training_ids',json.dumps(sorted(prior_ids|set(added.prediction_id)))))
        con.execute('INSERT OR REPLACE INTO kv VALUES (?,?)',('last_training',json.dumps(time.time())))
    store.event('retraining_started',{'new_training_rows':len(added),'recent_evaluation_rows':len(recent),'champion':champion['version']})
    candidate,cb=model.train(training,validation,candidate=True)
    hb=model.evaluate(champion,validation)
    hr=model.evaluate(champion,recent)
    cr=model.evaluate(candidate,recent)
    decision=gates(hb,cb,hr,cr)
    report={'candidate':candidate['version'],'champion':champion['version'],'decision':decision,
            'baseline':{'champion':hb,'candidate':cb},'recent':{'champion':hr,'candidate':cr},
            'new_training_ids':added.prediction_id.tolist(),'recent_evaluation_ids':recent.prediction_id.tolist()}
    (c.ROOT/'reports'/f"{candidate['version']}.json").write_text(json.dumps(report,indent=2))
    if decision['passed']:
        try:
            model.smoke(model.load(candidate['version']),validation)
            store.activate(candidate['version'],'Passed all automatic promotion gates',expected=champion['version'])
            # Deliberately reload the deployed artifact for a post-activation smoke test.
            model.smoke(model.load(),validation)
            status='promoted'
        except Exception as exc:
            if store.get('active_model')==candidate['version']:
                store.activate(champion['version'],'Post-activation smoke failure: '+str(exc),expected=candidate['version'])
            with store.db() as con:
                con.execute("UPDATE models SET status='broken' WHERE version=?",(candidate['version'],))
            store.event('promotion_failed',{'candidate':candidate['version'],'error':str(exc)})
            status='rolled_back'
    else:
        with store.db() as con:
            con.execute("UPDATE models SET status='rejected' WHERE version=?",(candidate['version'],))
        status='rejected'
    store.event('candidate_'+status,report)
    return {'status':status,**report}

def tick():
    store.init_db()
    try:
        with lock():
            data.verify_splits()
            recovered=recover_active()
            released=release_labels()
            report=monitor()
            result=retrain() if report['trigger'] else {'status':'no_retraining_needed'}
            store.put('pipeline',{'at':time.time(),'result':result,'labels_released':released,'recovered':recovered})
            store.put('worker_heartbeat',time.time())
            return result
    except Timeout:
        return {'status':'busy'}

def worker():
    while True:
        try:
            result=tick()
            print(json.dumps({'time':time.time(),**result}),flush=True)
        except Exception as exc:
            store.event('worker_error',{'error':str(exc)})
            print(json.dumps({'status':'error','error':str(exc)}),flush=True)
        time.sleep(c.WORKER_SECONDS)

def status():
    store.init_db()
    with store.db() as con:
        counts=dict(con.execute('SELECT COUNT(*) AS predictions, SUM(labels IS NOT NULL) AS labelled, SUM(labels IS NOT NULL AND consumed=0) AS unused_labels FROM predictions').fetchone())
        models=[dict(r) for r in con.execute('SELECT version,metrics,created,status FROM models ORDER BY created DESC')]
        used=con.execute('SELECT COUNT(*) FROM predictions WHERE source_id IS NOT NULL').fetchone()[0]
    for m in models: m['metrics']=json.loads(m['metrics'])
    heartbeat=store.get('worker_heartbeat')
    pool_size=len(data.read_split('stream')) if (c.ROOT/'data/stream.csv').exists() else 0
    return {'active_model':store.get('active_model'),'previous_model':store.get('previous_model'),'counts':counts,
            'models':models,'monitor':store.get('monitor'),'pipeline':store.get('pipeline'),
            'worker_heartbeat':heartbeat,'worker_healthy':bool(heartbeat and time.time()-heartbeat<max(90,c.WORKER_SECONDS*3)),
            'stream_remaining':pool_size-used,'demo_enabled':c.DEMO_ENABLED,'events':store.recent_events()}
