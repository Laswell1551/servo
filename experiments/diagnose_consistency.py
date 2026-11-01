"""Replay logged selections with recorded control times to diagnose event coverage.

This does not rerun or replace timing measurements. It reconstructs the actual
event-wise success outcomes and verifies every raw delivered-value observation.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import e2e_servo as b
from experiments import revision_consistency as r


def event_best(scene, selected, completed=None):
    best=np.zeros(len(scene.event_x))
    for j,k in enumerate(selected):
        if completed is not None and not completed[j]:continue
        for w,e in scene.flow_event_edges[k].items():best[w]=max(best[w],scene.q_true[e])
    return best


def main():
    b.seed_everything(2026)
    gnn,mlp,_=b.train_estimators(b.SimConfig(),False,False)
    actor=SAC.load(str(r.CKPT/'sac_coverage_36000'))
    output=[]
    for kind,filename,methods in [('nominal','raw_main.csv',['SERVO','MLP']),
                                  ('drift','raw_recovery.csv',['SERVO','NoAdapt'])]:
        data=pd.read_csv(r.OUT/filename)
        for method in methods:
            print('Replay',kind,method,flush=True)
            for seed,group in data[data.method==method].groupby('seed'):
                cal=b.OnlineCalibrator('none' if method=='NoAdapt' else 'ph')
                cfg=b.SimConfig();spec=b.Scenario(name=kind,concept_drift=kind=='drift')
                for row in group.sort_values('slot').itertuples():
                    scene=b.make_scene(int(seed),int(row.slot),cfg,spec)
                    qr,ir=b.predict_scene(mlp if method=='MLP' else gnn,scene)
                    q,imp=cal.apply(scene,qr,ir);chosen=json.loads(row.selected)
                    ctx=r.context(scene,chosen,q,imp)
                    action,_=actor.predict(r.observation(ctx),deterministic=True)
                    p,f=b.action_to_resources(action,ctx,r.MAX_B)
                    out=b.allocation_outcome(ctx,p,f)
                    completed=((out['latency']+row.control_ms/1000<=ctx['deadline'][:len(chosen)]) &
                               (out['sinr_db']>=ctx['rel_db'][:len(chosen)]))
                    delivered=event_best(scene,chosen,completed)
                    potential=event_best(scene,chosen)
                    available=event_best(scene,range(len(scene.flow_x)))
                    raw=float(delivered@scene.imp_true)
                    assert abs(raw-row.raw_value)<1e-6,(kind,method,seed,row.slot,raw,row.raw_value)
                    mass=float(scene.imp_true[available>0].sum())
                    denom=float(available@scene.imp_true)
                    output.append(dict(scenario=kind,method=method,seed=seed,slot=row.slot,
                                       selected_potential=float(potential@scene.imp_true)/denom,
                                       delivered=float(raw/denom),
                                       potential_missed_mass=float(scene.imp_true[(potential==0)&(available>0)].sum())/mass,
                                       delivered_missed_mass=float(scene.imp_true[(delivered==0)&(available>0)].sum())/mass,
                                       selected=json.dumps(chosen)))
                    cal.update(row.slot,scene,chosen,qr,ir)
    raw=pd.DataFrame(output);raw.to_csv(r.OUT/'raw_selection_diagnostics.csv',index=False)
    result=[]
    for kind,other in [('nominal','MLP'),('drift','NoAdapt')]:
        df=raw[raw.scenario==kind]
        if kind=='drift':df=df[df.slot>=60]
        a=df[df.method=='SERVO'];d=df[df.method==other]
        for metric in ['selected_potential','delivered','potential_missed_mass','delivered_missed_mass']:
            diff=a.groupby('seed')[metric].mean()-d.groupby('seed')[metric].mean()
            result.append(dict(scenario=kind,comparison='SERVO--'+other,metric=metric,
                               delta=diff.mean(),ci95=r.ci(diff),seeds=len(diff)))
    pd.DataFrame(result).to_csv(r.OUT/'selection_diagnostics.csv',index=False)
    print('Reconstructed and verified',len(raw),'logged decisions.',flush=True)


if __name__=='__main__':main()
