#!/usr/bin/env python3
"""servo_style.py (v2) — editorial 'high-end' style: spotlight palette,
direct line labels, insight callouts, soft-shadow gradient bars."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, FancyBboxPatch

PALETTE={
    'SERVO':'#0E9C73','WUM-RD':'#E08A3C','OSM':'#6F7C8C','WUM':'#9AA4AF',
    'MAB-CP':'#B26E86','PF/Prio':'#B8C0C7','Random':'#D2D8DD',
    'SERVO-NoEvo':'#8FBEAF','plain greedy':'#AEB6BD','lazy (CELF)':'#0E9C73',
}
INK='#222a30'; SUB='#6a727a'; TICK='#69727a'; SPINE='#c2c8cd'; GRID='#edf0f2'
TIER={'H':'#D6402C','M':'#E0883C','L':'#0E9C73'}

def use_style():
    plt.rcParams.update({
        'font.family':'Times New Roman','mathtext.fontset':'stix',
        'font.size':8,'savefig.dpi':360,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.edgecolor':SPINE,'axes.linewidth':0.8,'axes.labelcolor':SUB,'axes.labelsize':8,
        'axes.titlesize':9.4,'axes.titleweight':'bold','axes.titlecolor':INK,'axes.titlepad':8,
        'xtick.color':TICK,'ytick.color':TICK,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'legend.fontsize':6.6,
        'figure.facecolor':'white','axes.facecolor':'white','axes.axisbelow':True,
        'lines.solid_capstyle':'round',
    })

def finish(ax,ygrid=True,xgrid=False):
    if ygrid: ax.yaxis.grid(True,color=GRID,lw=1.0)
    if xgrid: ax.xaxis.grid(True,color=GRID,lw=1.0)
    ax.tick_params(length=2.5,width=0.8)

def line(ax,x,y,color,label=None,ci=None,emph=False,marker='o',lw=2.0,ms=4.4,ls='-'):
    z=7 if emph else 4
    x=np.asarray(x,float); y=np.asarray(y,float)
    if ci is not None:
        ci=np.asarray(ci,float)
        ax.fill_between(x,y-ci,y+ci,color=color,alpha=0.12,zorder=z-1,lw=0)
    ax.plot(x,y,color=color,lw=(lw+0.6 if emph else lw),marker=marker,
            ms=(ms+0.8 if emph else ms),mfc=color,mec='white',mew=1.0,
            label=label,zorder=z,ls=ls)

def direct_labels(ax,entries,dx_frac=0.025,gap_frac=0.052,fs=7.0,right_pad=0.20):
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    dx=(x1-x0)*dx_frac; gap=(y1-y0)*gap_frac
    ents=sorted(entries,key=lambda e:e[3]); ys=[e[3] for e in ents]
    for i in range(1,len(ys)):
        if ys[i]-ys[i-1]<gap: ys[i]=ys[i-1]+gap
    if ys[-1]>y1-(y1-y0)*0.01:
        sh=ys[-1]-(y1-(y1-y0)*0.01); ys=[v-sh for v in ys]
    lx=max(e[2] for e in ents)+dx
    for (label,color,xe,ye),yy in zip(ents,ys):
        ax.plot([xe,lx-dx*0.2],[ye,yy],color=color,lw=0.8,alpha=0.55,zorder=5,clip_on=False)
        ax.text(lx,yy,label,color=color,fontsize=fs,fontweight='bold',
                va='center',ha='left',zorder=8,clip_on=False)
    ax.set_xlim(x0,lx+(x1-x0)*right_pad)

def insight(ax,text,xy,xytext,color=INK,fs=7.0):
    ax.annotate(text,xy=xy,xytext=xytext,fontsize=fs,color=color,fontweight='bold',
                ha='center',va='center',zorder=9,
                arrowprops=dict(arrowstyle='-|>',color=color,lw=1.0,
                                shrinkA=2,shrinkB=3,mutation_scale=9))

def grad_bar(ax,x,w,height,color,base=0.0,z=3,frac=0.55,edge=None,shadow=True,cap=True):
    c=np.array(to_rgb(color)); light=np.clip(c+(1-c)*frac,0,1)
    if shadow:
        y0,y1=ax.get_ylim(); off=(y1-y0)*0.012
        ax.add_patch(FancyBboxPatch((x+w*0.10,base-off),w,height,
            boxstyle="round,pad=0,rounding_size=0.0",fc=(0,0,0),alpha=0.08,ec='none',zorder=z-1))
    t=np.linspace(0,1,200)[:,None]
    img=np.clip(light*(1-t)+c*t,0,1).reshape(200,1,3)
    ax.imshow(img,extent=[x,x+w,base,base+height],origin='lower',aspect='auto',
              zorder=z,interpolation='bilinear')
    if cap:
        ax.plot([x,x+w],[base+height,base+height],color=np.clip(c*0.8,0,1),lw=1.1,zorder=z+1)
    if edge:
        ax.add_patch(Rectangle((x,base),w,height,fc='none',ec=edge,lw=1.1,zorder=z+1))
