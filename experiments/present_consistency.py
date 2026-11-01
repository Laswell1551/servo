"""Create paper tables, macros, and vector figures from verified revision results."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import revision_consistency as r

OUT = r.OUT; FIG = ROOT / 'manuscript'
COLORS = {'SERVO':'#0072B2','WUM':'#D55E00','Equal':'#777777',
          'Equal-Match':'#999999','Reference':'#009E73','CostCov':'#5D497A',
          'NoAdapt':'#CC79A7','EMA':'#56B4E9','Periodic':'#E69F00'}
LABELS = {'SERVO':'SERVO','MLP':'MLP','WUM':'Additive','FI-1':'FI-1',
          'FI-CVX':'FI-CVX*','Equal':'Equal-full','Equal-Match':'Equal-match',
          'Reference':'Reference','CostCov':'CostCov+Ref.','Reward-Add':'Additive reward'}
ORDER = ['SERVO','MLP','WUM','FI-1','Equal','Equal-Match','Reference','FI-CVX','CostCov','Reward-Add']


def cell(row, key, digits=3, factor=1):
    v = float(row[key]) * factor; c = float(row[key + '_ci95']) * factor
    if not np.isfinite(v): return '--'
    return f'${v:.{digits}f}\\,\\pm\\,{c:.{digits}f}$'


def table(caption, label, cols, header, rows):
    return ('\\begin{table*}[!t]\n\\centering\\footnotesize\n'
            '\\setlength{\\tabcolsep}{3pt}\n\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}' + cols + '@{}}\n\\toprule\n'
            + header + '\\\\\n\\midrule\n' + '\n'.join(rows)
            + '\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n')


def numbers_and_tables():
    main = pd.read_csv(OUT/'summary.csv').set_index('method')
    pair = pd.read_csv(OUT/'paired.csv')
    rec = pd.read_csv(OUT/'recovery_post.csv').set_index('method')
    times = pd.read_csv(OUT/'recovery.csv').set_index('method')
    scale = pd.read_csv(OUT/'scalability.csv')
    stress = pd.read_csv(OUT/'robustness.csv')
    large = scale[(scale.vehicles == 224) & (scale.method == 'SERVO')].iloc[0]
    macros = {name: f'{main.loc[method,"value"]:.3f}' for name,method in
              [('MainValue','SERVO'),('AddValue','WUM'),('MLPValue','MLP'),('EqualValue','Equal'),
               ('MatchValue','Equal-Match'),('RefValue','Reference'),('FIValue','FI-1'),
               ('CVXValue','FI-CVX'),('CostValue','CostCov'),('AddRewardValue','Reward-Add')]}
    for name,key,digits,factor in [('MainEnergy','energy_total_j',2,1),('MainEC','energy_per_completion',3,1),
                                  ('MainOverhead','overhead_ms',2,1),('ResourceCU','cpu_util',1,100),
                                  ('ResourcePU','power_util',1,100),('MainMiss','sla_violation',2,100),
                                  ('MainRaw','raw_value',3,1),('MainReqPct','request_completion',1,100)]:
        macros[name]=f'{main.loc["SERVO",key]*factor:.{digits}f}'
    macros.update(CVXFeasible=f'{main.loc["FI-CVX","solver_feasible"]*100:.2f}',
                  CVXService=f'{main.loc["FI-CVX","service_only_value"]:.3f}',
                  CVXOverhead=f'{main.loc["FI-CVX","overhead_ms"]:.2f}',
                  LargeValue=f'{large.value:.3f}',LargeOverhead=f'{large.overhead_ms:.2f}',
                  LargeCompleted=f'{large.n_completed:.2f}',LargeReqPct=f'{large.request_completion*100:.1f}',
                  LargeRate=f'{large.completed_mbit*10:.1f}',LargeSelected=f'{large.n_selected:.1f}')
    small=scale[(scale.vehicles==36)&(scale.method=='SERVO')].iloc[0]
    macros['SmallCompleted']=f'{small.n_completed:.2f}'
    macros['CostEnergy']=f'{main.loc["CostCov","energy_total_j"]:.2f}'
    diagnostic=pd.read_csv(OUT/'selection_diagnostics.csv')
    for prefix,comparison in [('GNN','SERVO--MLP'),('PH','SERVO--NoAdapt')]:
        p=diagnostic[diagnostic.comparison==comparison].set_index('metric')
        macros['Diag'+prefix+'Potential']=f'{p.loc["selected_potential","delta"]:.4f}'
        macros['Diag'+prefix+'PotentialCI']=f'{p.loc["selected_potential","ci95"]:.4f}'
    p=diagnostic[diagnostic.comparison=='SERVO--NoAdapt'].set_index('metric')
    macros['DiagPHMissReduction']=f'{-p.loc["delivered_missed_mass","delta"]:.4f}'
    macros['DiagPHMissCI']=f'{p.loc["delivered_missed_mass","ci95"]:.4f}'
    for prefix,other in [('GNN','MLP'),('PH','NoAdapt'),('Ref','Reference'),('Matched','Equal-Match'),
                         ('Reward','Reward-Add'),('Cost','CostCov')]:
        row=pair[(pair.right==other)&(pair.metric=='value')].iloc[0]
        delta=f'{row.delta:.4f}'
        macros['Delta'+prefix]='0.0000' if float(delta)==0 else delta
        macros['CI'+prefix]=f'{row.ci95:.4f}'
        change=pair[(pair.right==other)&(pair.metric=='selection_jaccard_distance')]
        if len(change):macros['Change'+prefix]=f'{change.iloc[0].delta*100:.1f}'
    for prefix,method in [('PH','SERVO'),('EMA','EMA'),('Periodic','Periodic'),('Frozen','NoAdapt')]:
        macros[prefix+'Recovery']=f'{times.loc[method,"recovery_slots"]:.1f}'
        macros[prefix+'Success']=f'{times.loc[method,"recovery_success"]*100:.0f}'
        macros[prefix+'MAE']=f'{rec.loc[method,"prediction_mae"]:.4f}'
        macros[prefix+'Value']=f'{rec.loc[method,"value"]:.3f}'
    st=stress[(stress.method=='SERVO')&(stress.scenario!='nominal')]
    macros.update(StressLow=f'{st.value.min():.3f}',StressHigh=f'{st.value.max():.3f}',
                  WorstMiss=f'{st.sla_violation.max()*100:.1f}')
    meta=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))['gnn_metrics']
    for name,key in [('GNNMAE','gnn_q_mae'),('MLPMAE','mlp_q_mae'),
                     ('GNNImportanceMAE','gnn_importance_mae'),('MLPImportanceMAE','mlp_importance_mae')]:
        macros[name]=f'{meta[key]:.4f}'
    text='% Generated from experiments/results_consistency; do not edit by hand.\n'
    text+='\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n'
    (FIG/'generated_consistency_numbers.tex').write_text(text,encoding='utf-8')
    rows=[]
    for method in ORDER:
        x=main.loc[method];name=LABELS[method]
        if method=='SERVO':name=r'\textbf{SERVO}'
        entries=[name,cell(x,'value'),cell(x,'raw_value',2),cell(x,'latency_p95_ms',1),
                 cell(x,'sla_violation',2,100),cell(x,'energy_total_j',2),
                 cell(x,'energy_per_completion',3),cell(x,'overhead_ms',2)]
        rows.append(' & '.join(entries)+r'\\')
    (FIG/'generated_consistency_rows.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    main_table=table('Nominal outcomes: means $\\pm$ 95\\% trace-seed CIs (ten seeds). '
                     '$D$ is unnormalized delivered value. P95 is the mean snapshot-wise completed-service '
                     '95th percentile of serial latency. Full overhead includes post-feedback calibration; '
                     'serial latency includes pre-execution control. FI-CVX* includes recorded Reference fallback.',
                     'tab:e2e','lrrrrrrr',r'Method & $\bar V$ & $D$ & P95 ms & Miss\% & J/snapshot & J/completion & Full ms',rows)
    main_table=main_table.replace(r'\tabcolsep}{3pt}',r'\tabcolsep}{1.5pt}')
    (FIG/'generated_consistency_table.tex').write_text(main_table,encoding='utf-8')
    response_table=main_table.replace(r'\begin{table*}[!t]',r'\begin{table}[H]').replace(r'\end{table*}',r'\end{table}')
    response_table=response_table.replace('tab:e2e','tab:response-results')
    (FIG/'generated_consistency_response_table.tex').write_text(response_table,encoding='utf-8')
    tables=[]
    rows=[]
    for method in ORDER:
        x=main.loc[method]
        rows.append(' & '.join([LABELS[method],cell(x,'cpu_util',1,100),cell(x,'power_util',1,100),
                               cell(x,'service_only_value'),cell(x,'request_completion',2,100),
                               cell(x,'n_completed',2)])+r'\\')
    tables.append(table('Additional nominal outcomes (ten trace seeds; means and 95\\% CIs). Service-only value omits control delay; the main table uses serial control-plus-service success.',
                        'tab:supp-util','lrrrrr',r'Method & CPU\% & Power\% & Service-only $\bar V$ & Requests completed\% & Completed/snapshot',rows))
    rows=[]
    for other in ['MLP','WUM','Equal','Equal-Match','Reference','Reward-Add','CostCov','NoAdapt','EMA','Periodic']:
        pp=pair[pair.right==other].set_index('metric')
        scope='post-drift' if other in ['NoAdapt','EMA','Periodic'] else 'nominal'
        def fmt(metric):
            delta=f'{pp.loc[metric,"delta"]:.4f}'
            if float(delta)==0:delta='0.0000'
            return f'${delta}\\pm{pp.loc[metric,"ci95"]:.4f}$'
        rows.append(' & '.join([f'SERVO--{other}',scope,fmt('value'),fmt('selected_value'),fmt('prediction_mae'),
                               f'{pp.loc["selection_jaccard_distance","delta"]*100:.1f}'])+r'\\')
    tables.append(table('Paired trace-seed differences (SERVO minus comparator; 95\\% CIs). Potential means selected reference coverage before delivery losses. Jaccard distance measures selection change, not prediction error. Intervals are conditional on one trained-model seed.',
                        'tab:supp-paired','llrrrr',r'Comparison & Scope & $\Delta\bar V$ & $\Delta$potential & $\Delta$MAE & Jaccard\%',rows))
    rows=[]
    for x in scale.sort_values(['vehicles','method']).itertuples():
        row=x._asdict(); conf=f'({int(x.vehicles)},{int(x.events)},{int(x.nodes)};{int(x.budget)})'
        rows.append(' & '.join([conf,LABELS.get(x.method,x.method),cell(row,'value'),cell(row,'n_completed',2),
                               cell(row,'request_completion',1,100),cell(row,'completed_mbit',1,10),cell(row,'overhead_ms',2)])+r'\\')
    tables.append(table('Joint scaling of (vehicles, events, nodes; admission budget), five seeds. Payload rate divides completed Mbit/snapshot by the 100-ms indexing interval; it is snapshot-normalized output, not steady-state queued throughput.',
                        'tab:supp-scale','llrrrrr',r'Configuration & Method & $\bar V$ & Completed/snapshot & Requests\% & Payload Mbit/s & Overhead ms',rows))
    rows=[]
    for method in ['SERVO','EMA','Periodic','NoAdapt']:
        x=rec.loc[method]
        rows.append(' & '.join([method,f'{times.loc[method,"recovery_slots"]:.1f}',
                               f'{times.loc[method,"recovery_success"]*100:.0f}',cell(x,'prediction_mae',4),
                               cell(x,'value'),cell(x,'sla_violation',2,100)])+r'\\')
    tables.append(table('Recovery and post-drift outcomes on six common traces. Recovery times are censored means in snapshots; other outcomes show means and 95\\% CIs.',
                        'tab:supp-recovery','lrrrrr',r'Method & Recovery & Success\% & MAE & $\bar V$ & Miss\%',rows))
    oracle=pd.read_csv(OUT/'oracle.csv');rows=[]
    for x in oracle.itertuples():
        d=x._asdict();name={'SERVO':'Coverage','WUM':'Additive','CostCov':'CostCov'}[x.method]
        rows.append(' & '.join([name,cell(d,'value'),cell(d,'oracle_value'),cell(d,'oracle_ratio')])+r'\\')
    tables.append(table('Small exact reference-coverage upper bound (up to 12 candidates, eight events, three nodes, budget four; ten trace seeds). All tested selectors use Reference allocation. The upper bound ignores delivery and is not a resource-coupled optimum.',
                        'tab:supp-oracle','lrrr',r'Selector & Delivered $\bar V$ & Exact coverage upper bound & Delivered/oracle',rows))
    rows=[]
    for comparison in ['SERVO--MLP','SERVO--NoAdapt']:
        p=diagnostic[diagnostic.comparison==comparison].set_index('metric')
        cells=[f'${p.loc[m,"delta"]:.4f}\\pm{p.loc[m,"ci95"]:.4f}$'
               for m in ['selected_potential','delivered','potential_missed_mass','delivered_missed_mass']]
        rows.append(' & '.join([comparison,'nominal' if comparison.endswith('MLP') else 'post-drift']+cells)+r'\\')
    tables.append(table('Event-level replay of 3,840 logged decisions, using recorded control times. Differences are SERVO minus comparator, with paired 95\\% trace-seed CIs. Potential/delivered coverage uses normalized quality-weighted value; missed mass uses importance-weighted event omission, for which a negative difference is favorable.',
                        'tab:supp-diagnostics','llrrrr',r'Comparison & Scope & $\Delta$potential & $\Delta\bar V$ & $\Delta M_{\rm potential}$ & $\Delta M_{\rm delivered}$',rows))
    (FIG/'generated_consistency_supp_tables.tex').write_text('\n'.join(tables),encoding='utf-8')
    (OUT/'paper_numbers.json').write_text(json.dumps(macros,indent=2),encoding='utf-8')


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':8.2,'axes.titlesize':8.6,'axes.labelsize':8.2,
                         'xtick.labelsize':7.2,'ytick.labelsize':7.2,'legend.fontsize':7,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320})


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',linestyle=':',alpha=.4);ax.set_axisbelow(True)


def teaser():
    fig,axes=plt.subplots(1,2,figsize=(3.55,1.88))
    for ax,title,selected in zip(axes,['Additive','Coverage-aware'],[[0,1],[0,2]]):
        ax.set_xlim(-.1,1.1);ax.set_ylim(-.05,1.16);ax.axis('off')
        ax.text(.5,1.10,title,ha='center',fontweight='bold',fontsize=9)
        flowx=np.linspace(.06,.94,4);eventx=[.12,.5,.88]
        for k,x in enumerate(flowx):
            w=[0,0,1,2][k];active=k in selected
            color='#0072B2' if active else '#8D969D'
            ax.add_patch(FancyArrowPatch((x,.78),(eventx[w],.36),arrowstyle='-|>',mutation_scale=8,
                                         color=color,linewidth=1.6 if active else .8,alpha=1 if active else .65))
            ax.add_patch(Circle((x,.85),.075,fc=color if active else 'white',ec=color,lw=1.2))
            ax.text(x,.85,f'{k+1}',ha='center',va='center',color='white' if active else '#333333',fontsize=8.5)
        covered={ [0,0,1,2][k] for k in selected }
        for w,x in enumerate(eventx):
            color='#0072B2' if w in covered else '#8D969D'
            ax.add_patch(Circle((x,.25),.09,fc='#DDECF4' if w in covered else 'white',ec=color,lw=1.2))
            ax.text(x,.25,rf'$\omega_{{{w+1}}}$',ha='center',va='center',fontsize=9)
        ax.text(.5,.02,f'2 admissions; {len(covered)} event'+('s' if len(covered)>1 else ''),ha='center',fontsize=8.4)
    fig.subplots_adjust(wspace=.23,left=.02,right=.98,top=.95,bottom=.04)
    save(fig,'fig_teaser_v2')


def nominal():
    data=pd.read_csv(OUT/'summary.csv').set_index('method')
    methods=['SERVO','WUM','Equal','Equal-Match','Reference','CostCov']
    labels=['Full','Add.','Equal','Match','Ref.','CostCov']
    specs=[('value','Delivered value $\\uparrow$',1),('latency_p95_ms','Serial P95 latency (ms) $\\downarrow$',1),
           ('sla_violation','SLA miss rate $\\downarrow$',1),('energy_per_completion','Energy / completion (J) $\\downarrow$',1),
           ('energy_total_j','Energy / snapshot (J) $\\downarrow$',1),('overhead_ms','Full overhead (ms) $\\downarrow$',1)]
    fig,axes=plt.subplots(2,3,figsize=(7.15,3.55))
    for ax,(key,title,factor) in zip(axes.flat,specs):
        vals=[data.loc[m,key]*factor for m in methods];err=[data.loc[m,key+'_ci95']*factor for m in methods]
        bars=ax.bar(np.arange(len(methods)),vals,yerr=err,capsize=2,color=[COLORS[m] for m in methods],edgecolor='black',linewidth=.45)
        bars[1].set_hatch('//');ax.set_xticks(range(len(methods)),labels,rotation=30,ha='right')
        ax.set_title(title,fontweight='bold');clean(ax)
        if key=='value':ax.set_ylim(0,1)
        if key=='sla_violation':ax.set_ylim(0,max(.04,max(np.array(vals)+np.array(err))*1.2))
    fig.subplots_adjust(wspace=.38,hspace=.82,left=.07,right=.99,bottom=.13,top=.91)
    save(fig,'fig_e2e_main')


def robustness():
    data=pd.read_csv(OUT/'robustness.csv');names=['nominal','burst','mobility','heterogeneity','failure','congestion','drift']
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.55));x=np.arange(len(names));width=.24
    for j,m in enumerate(['SERVO','NoAdapt','CostCov']):
        rows=data[data.method==m].set_index('scenario').loc[names]
        for ax,key in zip(axes,['value','sla_violation']):
            ax.bar(x+(j-1)*width,rows[key],width,yerr=rows[key+'_ci95'],capsize=1.5,
                   label=m,color=COLORS[m],edgecolor='black',linewidth=.35)
    for ax,title in zip(axes,['(a) Delivered value','(b) Serial SLA miss rate']):
        ax.set_xticks(x,['Nom.','Burst','Mob.','Hetero.','Failure','Congest.','Drift'],rotation=28,ha='right')
        ax.set_title(title,fontweight='bold');clean(ax)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'fig_e2e_robustness')


def scale_recovery():
    data=pd.read_csv(OUT/'scalability.csv');raw=pd.read_csv(OUT/'raw_recovery.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.55))
    for m in ['SERVO','WUM','CostCov']:
        d=data[data.method==m].sort_values('vehicles');x=np.arange(len(d))
        for ax,key in zip(axes[:2],['value','overhead_ms']):
            ax.errorbar(x,d[key],yerr=d[key+'_ci95'],marker='o',markersize=3,capsize=2,color=COLORS[m],label=m)
    labels=[f'{int(v)}/{int(e)}/{int(n)}\nB={int(bu)}' for v,e,n,bu in
            data[data.method=='SERVO'].sort_values('vehicles')[['vehicles','events','nodes','budget']].itertuples(index=False,name=None)]
    for ax,title in zip(axes[:2],['(a) Joint-scale value','(b) Full overhead']):
        ax.set_xticks(range(4),labels,fontsize=7.0);ax.set_xlabel('Config.: V / E / N; budget')
        ax.set_title(title,fontweight='bold');clean(ax)
    axes[0].set_ylabel('normalized value');axes[1].set_ylabel('ms / snapshot')
    axes[0].legend(frameon=False,fontsize=6)
    for m in ['SERVO','EMA','Periodic','NoAdapt']:
        d=raw[(raw.method==m)&(raw.seed==22000)].sort_values('slot')
        axes[2].plot(d.slot,d.prediction_mae,label=m,color=COLORS[m],lw=1.1)
    axes[2].axvline(60,color='black',ls=':',lw=.8);axes[2].set_xlabel('snapshot')
    axes[2].set_ylabel('candidate-graph MAE');axes[2].set_title('(c) Common drift trace',fontweight='bold')
    axes[2].legend(frameon=False,fontsize=6);clean(axes[2])
    fig.tight_layout(w_pad=.9);save(fig,'fig_e2e_scale_recovery')


def extra():
    data=pd.read_csv(OUT/'resource_curve.csv');fig,axes=plt.subplots(1,2,figsize=(7.15,2.7))
    for mode,label,color in [('sac','Coverage-SAC','#0072B2'),('reference','Reference','#009E73'),('equal','Equal','#777777')]:
        d=data[data.method==mode].sort_values('resource_fraction')
        for ax,key in zip(axes,['value','sla_violation']):
            ax.errorbar(d.energy_total_j,d[key],xerr=d.energy_total_j_ci95,yerr=d[key+'_ci95'],
                        marker='o',markersize=4,capsize=2,label=label,color=color)
    for ax,title in zip(axes,['(a) Coverage--energy trade-off','(b) SLA--energy trade-off']):
        ax.set_xlabel('attempt energy / snapshot (J)');ax.set_title(title,fontweight='bold');clean(ax)
    axes[0].set_ylabel('normalized delivered value');axes[1].set_ylabel('SLA miss rate')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=3,loc='upper center')
    fig.tight_layout(rect=(0,0,1,.88));save(fig,'fig_consistency_resource')
    over=pd.read_csv(OUT/'overlap.csv');oracle=pd.read_csv(OUT/'oracle.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.7))
    for mode,color in [('Coverage','#0072B2'),('Additive','#D55E00')]:
        d=over[over.method==mode].sort_values('retention')
        axes[0].errorbar(d.mean_redundancy,d.value,yerr=d.value_ci95,marker='o',capsize=2,label=mode,color=color)
    axes[0].set_xlabel(r'mean event multiplicity $\bar\rho$');axes[0].set_ylabel('normalized delivered value')
    axes[0].set_title('(a) Controlled overlap',fontweight='bold');axes[0].legend(frameon=False);clean(axes[0])
    d=oracle.set_index('method').loc[['SERVO','WUM','CostCov']];x=np.arange(3)
    axes[1].bar(x,d.value,yerr=d.value_ci95,capsize=3,color=[COLORS[m] for m in d.index],edgecolor='black',linewidth=.5)
    axes[1].axhline(d.oracle_value.iloc[0],color='black',ls='--',label='exact coverage upper bound')
    axes[1].set_xticks(x,['Coverage','Additive','CostCov']);axes[1].set_ylabel('normalized value')
    axes[1].set_title('(b) Small-set exact upper bound',fontweight='bold');axes[1].legend(frameon=False,fontsize=6.5);clean(axes[1])
    fig.tight_layout(w_pad=2);save(fig,'fig_consistency_overlap_oracle')


def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    assert not manifest['smoke'] and manifest['train_steps']==36000
    assert manifest['script_sha256']==hashlib.sha256((ROOT/'experiments/revision_consistency.py').read_bytes()).hexdigest()
    assert (OUT/'verification.json').is_file()
    style();numbers_and_tables();teaser();nominal();robustness();scale_recovery();extra()
    print('Generated LaTeX tables, result macros, and six vector figures.')


if __name__=='__main__':main()
