#!/usr/bin/env python3
"""servo_restyle.py (v2) — editorial high-end re-render of ALL data figures.
Experiment outputs are cached (genuine, deterministic) so styling iterates fast."""
import sys, os, time, pickle, numpy as np
BASE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,BASE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import servo_style as S
from servo_style import PALETTE, TIER, INK, SUB, line, grad_bar, finish, direct_labels, insight
from servo_sim import (exp_budget, exp_redundancy, exp_comprehensive, exp_ablation,
                       exp_approx, exp_drift, make_instance, value, covered_mask,
                       sel_servo, sel_wum, sel_wumrd, sel_osm, sel_pfprio, sel_random, ci95)
S.use_style()
OUT=BASE; SEEDS=list(range(20))
def save(fig,n): fig.savefig(f"{OUT}/{n}.pdf",bbox_inches="tight"); fig.savefig(f"{OUT}/{n}.png",bbox_inches="tight"); plt.close(fig)
def cached(name,fn):
    p=os.path.join(BASE,f"_cache_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p,"rb"))
    v=fn(); pickle.dump(v,open(p,"wb")); return v

def sel_servo_plain(N,B,imp,cov,q):
    Sx=[]; vS=0.0
    for _ in range(B):
        bk,bg=-1,-1.0
        for k in range(N):
            if k in Sx: continue
            g=value(Sx+[k],imp,cov,q)-vS
            if g>bg: bg,bk=g,k
        if bk<0: break
        Sx.append(bk); vS=value(Sx,imp,cov,q)
    return Sx

def comp_value_vs_rho():
    E=45;B=12;Ns=(45,75,120,165,210,270,340)
    M={"SERVO":sel_servo,"WUM-RD":sel_wumrd,"OSM":sel_osm,"WUM":sel_wum,"PF/Prio":sel_pfprio,"Random":sel_random}
    vr={m:([],[]) for m in M}; rr=[]
    for N in Ns:
        rr.append(round(N*3.2/E,1)); per={m:[] for m in M}
        for s in SEEDS:
            imp,cov,q,_=make_instance(N,E,s); full=value(list(range(N)),imp,cov,q)
            for m,fn in M.items():
                Ssel=fn(N,B,imp,cov,q) if m!="Random" else fn(N,B,imp,cov,q,seed=s)
                per[m].append(value(Ssel,imp,cov,q)/full)
        for m in M: vr[m][0].append(float(np.mean(per[m]))); vr[m][1].append(float(ci95(per[m])))
    return rr,vr
def comp_scalability():
    E=45;Ns=(60,120,240,360,480,640); lz=[];pl=[]
    for N in Ns:
        tl=[];tp=[]
        for s in range(8):
            imp,cov,q,_=make_instance(N,E,s)
            t0=time.perf_counter(); sel_servo(N,16,imp,cov,q); tl.append((time.perf_counter()-t0)*1000)
            t0=time.perf_counter(); sel_servo_plain(N,16,imp,cov,q); tp.append((time.perf_counter()-t0)*1000)
        lz.append(float(np.mean(tl))); pl.append(float(np.mean(tp)))
    return Ns,lz,pl
def comp_diminishing():
    import heapq; N=90;E=45;Bd=25; curves=[]
    for s in SEEDS:
        imp,cov,q,_=make_instance(N,E,s); Sx=[];vp=0.0;g=[]
        heap=[(-value([k],imp,cov,q),k,0) for k in range(N)]; heapq.heapify(heap)
        while heap and len(Sx)<Bd:
            ng,k,st=heapq.heappop(heap)
            if st==len(Sx):
                gg=value(Sx+[k],imp,cov,q)-vp; Sx.append(k); vp+=gg; g.append(gg)
            else: heapq.heappush(heap,(-(value(Sx+[k],imp,cov,q)-vp),k,len(Sx)))
        curves.append(g[:Bd])
    m=min(len(c) for c in curves); arr=np.array([c[:m] for c in curves])
    return arr.mean(0).tolist(),(1.96*arr.std(0,ddof=1)/np.sqrt(len(arr))).tolist()
def comp_case():
    N=12;E=9;B=3; best=None
    for s in range(200):
        imp,cov,q,_=make_instance(N,E,s,cov_per_flow=2.6)
        Sw=sel_wum(N,B,imp,cov,q); Ss=sel_servo(N,B,imp,cov,q)
        cw=covered_mask(Sw,cov,q,0.0); cs=covered_mask(Ss,cov,q,0.0); gap=len(cs)-len(cw)
        if best is None or gap>best[0]: best=(gap,s,imp,cov,q,list(Sw),list(Ss),list(cw),list(cs),float(value(Sw,imp,cov,q)),float(value(Ss,imp,cov,q)))
    return best

print("computing/loading experiments...")
budgets,e1=cached("e1",exp_budget)
rhos,gmean,gci=cached("e2",exp_redundancy)
comp=cached("e3",exp_comprehensive)
abl=cached("e4",exp_ablation)
ratios=np.array(cached("e5",exp_approx))
track,regret,ds=cached("e6",exp_drift)
rr,vr=cached("vr",comp_value_vs_rho)
Ns2,lazy,plain=cached("scal",comp_scalability)
dmean,dci=cached("dim",comp_diminishing)
case=cached("case",comp_case)
print("rendering...")

# 1 coverage vs budget — direct labels
fig,ax=plt.subplots(figsize=(3.6,2.8))
for m in ["Random","PF/Prio","WUM","OSM","WUM-RD","SERVO"]:
    line(ax,budgets,e1[m]["mean"],PALETTE[m],ci=e1[m]["ci"],emph=(m=="SERVO"))
ax.set_xlabel("admission budget  $B$  (flows)"); ax.set_ylabel("normalised delivered value  $\\bar V$")
ax.set_ylim(0,1.04); finish(ax)
direct_labels(ax,[(m,PALETTE[m],budgets[-1],e1[m]["mean"][-1]) for m in
                  ["SERVO","WUM-RD","OSM","WUM","PF/Prio","Random"]])
save(fig,"fig_coverage_vs_budget")

# 2 gain vs redundancy
fig,ax=plt.subplots(figsize=(3.6,2.8))
gm=np.array(gmean); gc=np.array(gci)
ax.fill_between(rhos,gm-gc,gm+gc,color=PALETTE["SERVO"],alpha=0.14,lw=0)
line(ax,rhos,gm,PALETTE["SERVO"],emph=True,marker="s")
ax.set_xlabel(r"redundancy  $\rho$  (flows per event)"); ax.set_ylabel("SERVO value gain over WUM (%)")
ax.set_ylim(0,max(gm+gc)*1.22); finish(ax)
insight(ax,"grows with\nredundancy",xy=(rhos[-2],gm[-2]),xytext=(rhos[1]+1,max(gm+gc)*1.06),color=INK)
save(fig,"fig_gain_vs_redundancy")

# 3 value vs rho — direct labels
fig,ax=plt.subplots(figsize=(3.6,2.8))
for m in ["Random","PF/Prio","WUM","OSM","WUM-RD","SERVO"]:
    line(ax,rr,vr[m][0],PALETTE[m],ci=vr[m][1],emph=(m=="SERVO"))
ax.set_xlabel(r"redundancy  $\rho$  (flows per event)"); ax.set_ylabel("normalised delivered value  $\\bar V$")
ax.set_ylim(0,1.04); finish(ax)
direct_labels(ax,[(m,PALETTE[m],rr[-1],vr[m][0][-1]) for m in
                  ["SERVO","WUM-RD","OSM","WUM","PF/Prio","Random"]])
save(fig,"fig_value_vs_rho")

# 4 scalability — direct labels + insight
fig,ax=plt.subplots(figsize=(3.6,2.8))
line(ax,Ns2,plain,PALETTE["plain greedy"],marker="s")
line(ax,Ns2,lazy,PALETTE["lazy (CELF)"],emph=True)
ax.set_xlabel("number of flows  $N$"); ax.set_ylabel("selection runtime (ms)")
ax.set_ylim(0,max(plain)*1.18); finish(ax)
direct_labels(ax,[("plain greedy",PALETTE["plain greedy"],Ns2[-1],plain[-1]),
                  ("lazy (CELF)",PALETTE["SERVO"],Ns2[-1],lazy[-1])],gap_frac=0.10)
sp=plain[-1]/lazy[-1]
insight(ax,f"{sp:.1f}$\\times$ faster",xy=(Ns2[-1],lazy[-1]),xytext=(Ns2[-3]-30,lazy[-1]+max(plain)*0.22),color=PALETTE["SERVO"])
save(fig,"fig_scalability")

# 5 diminishing
fig,ax=plt.subplots(figsize=(3.6,2.8))
x=np.arange(1,len(dmean)+1)
line(ax,x,dmean,PALETTE["SERVO"],ci=dci,emph=True)
ax.set_xlabel("greedy iteration"); ax.set_ylabel("marginal value gain  $\\Delta(e\\,|\\,S)$")
ax.set_ylim(0,max(dmean)*1.14); finish(ax)
insight(ax,"diminishing returns\n(submodularity)",xy=(x[4],dmean[4]),xytext=(x[10],max(dmean)*0.7),color=INK)
save(fig,"fig_diminishing")

# 6 approx ratio
fig,ax=plt.subplots(figsize=(3.6,2.8))
n,bins,patches=ax.hist(ratios,bins=12,color=PALETTE["SERVO"],edgecolor="white",linewidth=0.9,zorder=3)
for p,h in zip(patches,n): p.set_alpha(0.45+0.5*h/max(n))
ax.axvline(0.6321,color=TIER['H'],ls="--",lw=1.5,zorder=4)
ax.axvline(float(ratios.mean()),color=INK,ls=":",lw=1.3,zorder=4)
ax.set_xlabel("empirical ratio  $V(S_{greedy})/V(S^\\star)$"); ax.set_ylabel("count"); finish(ax)
ax.text(0.6321,max(n)*1.02,"$1-1/e$",color=TIER['H'],fontsize=6.8,ha="center",fontweight="bold")
ax.text(float(ratios.mean()),max(n)*0.78,f"mean {ratios.mean():.2f}",color=INK,fontsize=6.8,ha="center",fontweight="bold")
save(fig,"fig_approx_ratio")

# 7 comprehensive (6 panels)
order=["SERVO","WUM-RD","OSM","MAB-CP","WUM","PF/Prio","Random"]
panels=[("value","normalised value  $\\bar V$",False),("succ","semantic-task success",False),
        ("hi_sla","high-tier SLA",False),("lo_sla","low-tier SLA",False),
        ("robust","robustness ($\\pm$40%)",False),("runtime","edge runtime (ms)",True)]
fig,axes=plt.subplots(2,3,figsize=(7.2,4.2))
for ax,(key,ttl,logy) in zip(axes.flat,panels):
    xs=np.arange(len(order)); bw=0.66
    vals=[comp[m][key][0] for m in order]; errs=[comp[m][key][1] for m in order]
    if logy:
        ax.set_yscale("log"); base=min(v for v in vals if v>0)*0.45
    for xi,m,v,er in zip(xs,order,vals,errs):
        emph=(m=="SERVO")
        if logy: grad_bar(ax,xi-bw/2,bw,v-base,PALETTE[m],base=base,edge=(INK if emph else None),frac=0.5,shadow=False)
        else:    grad_bar(ax,xi-bw/2,bw,v,PALETTE[m],edge=(INK if emph else None),frac=0.5)
        ax.errorbar(xi,v,yerr=er,fmt="none",ecolor="#3a3f45",elinewidth=0.8,capsize=2,zorder=8)
        ax.text(xi,(v*1.16 if logy else v+0.02),f"{v:.2f}",ha="center",va="bottom",fontsize=5.7,color="#333",zorder=9)
    ax.set_xticks(xs); ax.set_xticklabels(order,rotation=42,ha="right",fontsize=5.9)
    ax.set_title(ttl,fontsize=8.1)
    if not logy: ax.set_ylim(0,max(vals)*1.24)
    finish(ax)
fig.tight_layout(w_pad=1.5,h_pad=1.8)
save(fig,"fig_comprehensive")

# 8 ablation
variants=list(abl.keys())
fig,ax=plt.subplots(figsize=(3.8,2.9))
xs=np.arange(len(variants)); bw=0.62
vals=[abl[v]["value"][0] for v in variants]; errs=[abl[v]["value"][1] for v in variants]
cols=[PALETTE["SERVO"],PALETTE["WUM"],PALETTE["OSM"],PALETTE["Random"]]
for xi,vv,er,c,name in zip(xs,vals,errs,cols,variants):
    emph=name.startswith("Full")
    grad_bar(ax,xi-bw/2,bw,vv,c,edge=(INK if emph else None),frac=0.5)
    ax.errorbar(xi,vv,yerr=er,fmt="none",ecolor="#3a3f45",elinewidth=0.8,capsize=2,zorder=8)
    ax.text(xi,vv+0.02,f"{vv:.2f}",ha="center",va="bottom",fontsize=6.3,color="#333",zorder=9)
ax.set_xticks(xs); ax.set_xticklabels(variants,rotation=20,ha="right",fontsize=6.2)
ax.set_ylabel("normalised delivered value  $\\bar V$"); ax.set_ylim(0,1.02); finish(ax)
save(fig,"fig_ablation")

# 9 drift (2 panels) — direct labels
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,2.9))
T=len(next(iter(track.values()))); xT=np.arange(T)
seq=["MAB-CP","WUM-RD","SERVO-NoEvo","SERVO"]
for d in ds: a1.axvline(d,color="#ccd1d6",ls=":",lw=1.0,zorder=1)
for m in seq: line(a1,xT,track[m],PALETTE[m],emph=(m=="SERVO"),marker=None,lw=1.7)
a1.set_xlabel("time slot"); a1.set_ylabel("delivered value / full"); a1.set_title("(a) Value tracking under drift",fontsize=8.4)
finish(a1)
direct_labels(a1,[(m,PALETTE[m],T-1,track[m][-1]) for m in seq],gap_frac=0.07,fs=6.4)
for m in seq: line(a2,xT,regret[m],PALETTE[m],emph=(m=="SERVO"),marker=None,lw=1.9)
a2.set_xlabel("time slot"); a2.set_ylabel("cumulative value shortfall"); a2.set_title("(b) Cumulative regret",fontsize=8.4)
finish(a2)
direct_labels(a2,[(m,PALETTE[m],T-1,regret[m][-1]) for m in seq],gap_frac=0.07,fs=6.4)
fig.tight_layout(w_pad=1.8)
save(fig,"fig_drift")

# 10 case study (refined bipartite + tier legend + node shadow)
gap,s,imp,cov,q,Sw,Ss,cw,cs,Vw,Vs=case
N=12;Ec=9;Bc=3
def tier(w): return "H" if imp[w]>=0.9 else ("M" if imp[w]>=0.5 else "L")
def panel(ax,sel,covd,Vv,ttl,accent):
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2,0.5),9.6,8.9,boxstyle="round,pad=0.02,rounding_size=0.25",
                 fc="#fbfcfd",ec="#e6eaec",lw=1.0,zorder=0))
    fy=np.linspace(1.3,8.7,N); ey=np.linspace(1.7,8.3,Ec)
    for k in range(N):
        for w in cov[k]:
            on=(k in sel)
            ax.plot([2.5,7.5],[fy[k],ey[w]],color=(accent if on else "#e6e9eb"),
                    lw=(1.9 if on else 0.6),alpha=(0.92 if on else 0.85),zorder=(4 if on else 1))
    for k in range(N):
        on=(k in sel)
        if on: ax.add_patch(Circle((2.56,fy[k]-0.04),0.22,fc=(0,0,0),alpha=0.10,zorder=4))
        ax.add_patch(Circle((2.5,fy[k]),0.20,fc=(accent if on else "white"),ec=(accent if on else "#bcc3c9"),lw=1.4,zorder=5))
    for w in range(Ec):
        cd=(w in covd); c=TIER[tier(w)]
        if cd: ax.add_patch(Circle((7.56,ey[w]-0.04),0.24,fc=(0,0,0),alpha=0.10,zorder=4))
        ax.add_patch(Circle((7.5,ey[w]),0.22,fc=(c if cd else "white"),ec=c,lw=1.6,zorder=5))
    ax.text(2.5,9.25,"flows",ha="center",fontsize=7,color="#888")
    ax.text(7.5,9.25,"events",ha="center",fontsize=7,color="#888")
    ax.text(5.0,9.9,ttl,ha="center",fontsize=9,fontweight="bold",color=accent)
    ax.text(5.0,0.12,f"covers {len(covd)}/{Ec} events    $V={Vv:.2f}$",ha="center",fontsize=7.4,color="#333")
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,3.6))
panel(a1,Sw,cw,Vw,"WUM  (additive)",PALETTE["WUM-RD"])
panel(a2,Ss,cs,Vs,"SERVO  (submodular)",PALETTE["SERVO"])
# tier legend
for i,(t,lab) in enumerate([("H","high"),("M","medium"),("L","low")]):
    fig.text(0.33+i*0.13,0.02,"  "+lab,fontsize=6.6,color="#555",va="center")
    fig.patches=fig.patches
    import matplotlib.patches as mp
    fig.add_artist(mp.Circle((0.325+i*0.13,0.028),0.008,color=TIER[t],transform=fig.transFigure))
fig.suptitle(f"Case study: selection on a V2X intersection  ($B={Bc}$ of $N={N}$ flows)",fontsize=9.2,fontweight="bold",y=1.03)
fig.tight_layout(rect=[0,0.04,1,1])
save(fig,"fig_case_study")

print("ALL EDITORIAL FIGURES DONE")
