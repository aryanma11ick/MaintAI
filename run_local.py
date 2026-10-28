"""One command starts all local services; Ctrl+C stops child processes."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parent
os.chdir(PROJECT)

def stop(proc):
    if proc.poll() is not None: return
    proc.terminate()
    try: proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()

def main():
    for port in (8000,8501):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0:
                sys.exit(f'Port {port} is already in use. Stop the other application before starting MaintAI.')
    subprocess.run([sys.executable,'-m','maintai','bootstrap'],check=True)
    from maintai.config import ROOT
    logs=ROOT/'logs'; logs.mkdir(exist_ok=True)
    commands={
        'api':[sys.executable,'-m','uvicorn','maintai.api:app','--host','127.0.0.1','--port','8000'],
        'worker':[sys.executable,'-m','maintai','worker'],
        'dashboard':[sys.executable,'-m','streamlit','run','maintai/dashboard.py','--server.address','127.0.0.1','--server.port','8501','--server.headless','true','--browser.gatherUsageStats','false']}
    processes={}; files={}; failures={name:0 for name in commands}
    def start(name):
        files[name]=open(logs/f'{name}.log','a',encoding='utf-8')
        processes[name]=subprocess.Popen(commands[name],stdout=files[name],stderr=subprocess.STDOUT)
    def interrupted(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for name in commands: start(name)
        print('\nMaintAI is starting.\nDashboard: http://localhost:8501\nAPI docs:  http://localhost:8000/docs\nLogs:      '+str(logs)+'\nPress Ctrl+C here to stop all services.\n',flush=True)
        while True:
            time.sleep(3)
            for name,proc in list(processes.items()):
                if proc.poll() is not None:
                    files[name].close()
                    failures[name]+=1
                    if failures[name]>3: raise RuntimeError(f'{name} repeatedly exited. Check {logs/name}.log')
                    print(f'{name} exited; restarting ({failures[name]}/3).',flush=True)
                    start(name)
    except KeyboardInterrupt:
        print('\nStopping MaintAI…')
    finally:
        for proc in processes.values(): stop(proc)
        for f in files.values(): f.close()

if __name__=='__main__': main()
