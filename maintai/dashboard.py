import os
from datetime import datetime,timezone
import httpx
import pandas as pd
import streamlit as st

API=os.getenv('MAINTAI_API_URL','http://127.0.0.1:8000')
st.set_page_config(page_title='MaintAI | Maintenance intelligence',page_icon='⚙',layout='wide')
st.markdown('''<style>
.block-container{max-width:1250px;padding-top:2rem}
[data-testid="stMetric"]{background:#132831;border:1px solid #28434b;padding:18px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#b7cbd0}
h1,h2,h3{letter-spacing:-.03em} .stCaption{color:#9aafb5}
</style>''',unsafe_allow_html=True)
st.caption('MAINTAI  /  LOCAL OPERATIONS CONSOLE')
st.title('Maintenance intelligence, with a recovery loop.')
st.write('Predict failures, inspect explanations, and watch each model decision.')
st.caption('AI4I is a synthetic benchmark. This is an engineering demonstration, not a validated industrial safety system.')

def request(method,path,payload=None):
    try:
        r=httpx.request(method,API+path,json=payload,timeout=90,trust_env=False)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        try: message=exc.response.json().get('detail',str(exc))
        except ValueError: message=str(exc)
        st.error(message)
    except httpx.RequestError:
        st.error('The API is unavailable. Start MaintAI with: python run_local.py')
    return None

overview,predict_tab,operations,experiments=st.tabs(['Overview','Inspect a machine','Recovery lab','Model history'])

@st.fragment(run_every='5s')
def live_overview():
    state=request('GET','/status')
    if state is None: return
    counts=state['counts']
    a,b,d,e=st.columns(4)
    a.metric('Readings received',counts['predictions'])
    b.metric('Outcome labels',counts['labelled'] or 0)
    d.metric('Worker','Running' if state['worker_healthy'] else 'Waiting / unavailable')
    e.metric('Replay records left',state['stream_remaining'])
    st.write('**Serving model:**',state['active_model'] or 'Not initialized')
    mon=state['monitor'] or {}
    run=state['pipeline'] or {}
    left,right=st.columns([1.4,1])
    with left:
        st.subheader('Monitoring')
        st.write('State:',mon.get('status','Waiting for the first worker cycle'))
        if 'drift' in mon:
            st.caption(f"Consecutive changed windows: {mon['drift_streak']} · At least two are needed for a drift trigger.")
            rows=[{'Feature':key,'Drift detected':v['drifted'],'Effect size':round(v.get('statistic',v.get('total_variation',0)),3)} for key,v in mon['drift']['features'].items()]
            st.dataframe(rows,hide_index=True,width='stretch')
        if mon.get('quality'):
            q=mon['quality']
            st.write(f"Recent labelled recall: **{q['recall']:.1%}** · Precision: **{q['precision']:.1%}** · {q['positives']} failures")
        else: st.caption('Quality monitoring waits for enough labelled outcomes from the active model.')
    with right:
        st.subheader('Recovery decision')
        result=run.get('result',{})
        st.info(result.get('status','Waiting for the worker').replace('_',' ').capitalize())
        if result.get('decision'):
            st.dataframe([{'Check':k.replace('_',' '),'Passed':v} for k,v in result['decision']['checks'].items()],hide_index=True)
        elif result.get('status')=='waiting_for_labels':
            st.write(f"{result['available']} / {result['required']} unused labels received.")
        st.caption('Keeping the current model after a failed evaluation is a valid recovery decision.')
    st.subheader('Audit trail')
    events=[{'Time (UTC)':datetime.fromtimestamp(e['ts'],timezone.utc).strftime('%H:%M:%S'),'Event':e['kind'],'Details':str(e['details'])[:600]} for e in state['events']]
    st.dataframe(events,hide_index=True,width='stretch')

with overview: live_overview()
with predict_tab:
    st.subheader('Inspect a reading')
    with st.form('reading'):
        l,r=st.columns(2)
        with l:
            product=st.selectbox('Product type',['L','M','H'],index=1)
            air=st.number_input('Air temperature (K)',200.,450.,300.,.1)
            process=st.number_input('Process temperature (K)',200.,600.,310.,.1)
        with r:
            rpm=st.number_input('Rotational speed (rpm)',1.,20000.,1500.,10.)
            torque=st.number_input('Torque (Nm)',0.,500.,40.,.1)
            wear=st.number_input('Tool wear (min)',0.,2000.,100.,1.)
        submit=st.form_submit_button('Analyze reading',type='primary')
    if submit:
        with st.spinner('Scoring and explaining the reading…'):
            result=request('POST','/predict',{'type':product,'air_temperature':air,'process_temperature':process,'rotational_speed':rpm,'torque':torque,'tool_wear':wear})
        if result: st.session_state['prediction']=result
    result=st.session_state.get('prediction')
    if result:
        a,b,d=st.columns(3)
        a.metric('Failure score',f"{result['failure_score']:.1%}")
        b.metric('Model flag','Review' if result['failure_flag'] else 'Below threshold')
        d.metric('Health indicator',f"{result['health_indicator']}/100")
        st.caption(result['health_note']+' Failure scores are not calibrated probabilities.')
        st.write('**Suggested checks**')
        for item in result['recommendations']: st.write('• '+item)
        st.subheader('What influenced this prediction?')
        explanation=result.get('explanation')
        if explanation:
            rows=pd.DataFrame(explanation['contributions']).set_index('feature')
            st.bar_chart(rows['contribution'],horizontal=True)
            st.caption('Positive contributions increase the failure score; negative values decrease it. '+explanation['meaning'])
        with st.expander('Auxiliary failure-type scores and raw response'):
            st.caption('Types can overlap. Rare types, especially random failures, may not be predictable; these scores are exploratory.')
            st.json(result)
with operations:
    st.subheader('Run a local recovery experiment')
    st.write('Replay reserved records, wait for delayed outcome labels, and let the worker decide whether a new model is justified.')
    st.info('Start with 250 high-wear readings. Wait at least 15 seconds, then send another 250. Labels arrive after 30 seconds. Promotion is earned by evaluation; it is not guaranteed.')
    scenario=st.selectbox('Replay distribution',['high-wear','normal'])
    count=st.slider('Readings in this batch',100,500,250,50)
    if st.button('Replay batch',type='primary'):
        outcome=request('POST','/demo/replay',{'count':count,'scenario':scenario})
        if outcome: st.success(str(outcome))
    st.caption('High-wear selects unused records with higher tool wear. It does not change sensor values or invent labels. A total of 1,500 unique records is available per fresh runtime.')
    st.divider()
    st.subheader('Restore a predecessor')
    st.write('Available only after another model has been activated. All changes are recorded in the audit trail.')
    if st.button('Roll back to previous model'):
        outcome=request('POST','/models/rollback')
        if outcome: st.success(str(outcome))
with experiments:
    state=request('GET','/status')
    st.subheader('Every candidate stays visible')
    if state:
        table=[{'Version':m['version'],'Status':m['status'],'Validation AP':round(m['metrics']['average_precision'],4),'Recall':round(m['metrics']['recall'],4),'Precision':round(m['metrics']['precision'],4)} for m in state['models']]
        st.dataframe(table,hide_index=True,width='stretch')
    st.caption('AP means average precision. Metrics above use the fixed validation split, not the untouched final test split. Complete runs and artifacts are also logged locally through MLflow.')
