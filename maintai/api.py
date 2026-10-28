from contextlib import asynccontextmanager
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from . import config as c, data, model, pipeline, store

@asynccontextmanager
async def lifespan(app):
    store.init_db()
    yield

app=FastAPI(title='MaintAI',version='0.1.0',lifespan=lifespan,
            description='Local predictive maintenance demo. Scores are not calibrated probabilities. Bind to localhost.')

class Reading(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    type: Literal['L','M','H']='M'
    air_temperature: float=Field(300,ge=200,le=450)
    process_temperature: float=Field(310,ge=200,le=600)
    rotational_speed: float=Field(1500,gt=0,le=20000)
    torque: float=Field(40,ge=0,le=500)
    tool_wear: float=Field(100,ge=0,le=2000)

class Feedback(BaseModel):
    model_config=ConfigDict(extra='forbid')
    failure: Literal[0,1]
    TWF: Literal[0,1]=0
    HDF: Literal[0,1]=0
    PWF: Literal[0,1]=0
    OSF: Literal[0,1]=0
    RNF: Literal[0,1]=0

class Replay(BaseModel):
    model_config=ConfigDict(extra='forbid')
    count: int=Field(250,ge=1,le=500)
    scenario: Literal['normal','high-wear']='normal'

@app.get('/health/live')
def live(): return {'status':'alive'}

@app.get('/health/ready')
def ready():
    try:
        bundle=model.load()
        model.smoke(bundle,data.read_split('validation'))
        return {'status':'ready','model':bundle['version']}
    except Exception as exc:
        raise HTTPException(503,str(exc)) from exc

@app.get('/status')
def status(): return pipeline.status()

@app.post('/predict')
def predict(reading: Reading, explain: bool=True):
    try:
        return pipeline.record_prediction(reading.model_dump(),with_explanation=explain)
    except (RuntimeError,FileNotFoundError) as exc:
        raise HTTPException(503,str(exc)) from exc

@app.post('/feedback/{prediction_id}')
def feedback(prediction_id: str, labels: Feedback):
    try: return pipeline.feedback(prediction_id,labels.model_dump())
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@app.post('/demo/replay')
def replay(request: Replay):
    if not c.DEMO_ENABLED: raise HTTPException(403,'Demo endpoints disabled')
    try: return pipeline.replay(request.count,request.scenario)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except __import__('filelock').Timeout as exc: raise HTTPException(409,'Pipeline busy; try again shortly') from exc

@app.post('/models/rollback')
def rollback():
    try: return pipeline.rollback()
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except __import__('filelock').Timeout as exc: raise HTTPException(409,'Pipeline busy; try again shortly') from exc
