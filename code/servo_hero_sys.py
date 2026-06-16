#!/usr/bin/env python3
"""servo_hero_figs.py (part 1) — isometric 3D-style SYSTEM MODEL figure.
Vector, IEEE-printable. Mirrors the reference's isometric system illustration."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path

import os
OUT=os.path.dirname(os.path.abspath(__file__))
TEAL="#159e76"; BLUE="#3b6ea5"; ORANGE="#e07b1a"; RED="#d6402c"; PURPLE="#7263b5"
INK="#222"; ROAD="#cfd3d6"; GRASS="#eef3ef"
plt.rcParams.update({"font.size":8,"font.family":"Times New Roman","mathtext.fontset":"stix","savefig.dpi":340,"pdf.fonttype":42,"ps.fonttype":42})

A=np.radians(30)
def iso(x,y,z=0.0):
    return ((x-y)*np.cos(A), (x+y)*np.sin(A)+z)
def shade(h,f):
    h=h.lstrip('#'); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return '#%02x%02x%02x'%(min(255,int(r*f)),min(255,int(g*f)),min(255,int(b*f)))

def poly(ax,pts3,color,z,ec="#2b2b2b",lw=0.4,alpha=1.0):
    ax.add_patch(Polygon([iso(*p) for p in pts3],closed=True,fc=color,ec=ec,lw=lw,zorder=z,alpha=alpha,joinstyle="round"))

def isobox(ax,x,y,w,d,h,color,z,topf=1.0,leftf=0.80,rightf=0.62,ec="#33333355"):
    # corners
    c=lambda X,Y,Z:(X,Y,Z)
    # right face (+x), left face (+y), top
    poly(ax,[c(x+w,y,0),c(x+w,y+d,0),c(x+w,y+d,h),c(x+w,y,h)],shade(color,rightf),z,ec=ec)
    poly(ax,[c(x,y+d,0),c(x+w,y+d,0),c(x+w,y+d,h),c(x,y+d,h)],shade(color,leftf),z,ec=ec)
    poly(ax,[c(x,y,h),c(x+w,y,h),c(x+w,y+d,h),c(x,y+d,h)],shade(color,topf),z+0.05,ec=ec)

def car(ax,x,y,color,z,facing="x"):
    # body + cabin as two low iso boxes; footprint ~1.3 x 0.72
    w,d=(1.35,0.72) if facing=="x" else (0.72,1.35)
    isobox(ax,x,y,w,d,0.34,color,z,topf=1.06,leftf=0.82,rightf=0.64)
    # cabin
    if facing=="x":
        isobox(ax,x+0.32,y+0.06,0.62,0.60,0.30,shade(color,1.12),z+0.2,topf=1.1,leftf=0.86,rightf=0.7)
    else:
        isobox(ax,x+0.06,y+0.32,0.60,0.62,0.30,shade(color,1.12),z+0.2,topf=1.1,leftf=0.86,rightf=0.7)

def signal(ax,x,y,z0,n=3,color=TEAL):
    sx,sy=iso(x,y,z0)
    for i in range(1,n+1):
        r=0.16*i
        th=np.linspace(np.radians(35),np.radians(145),40)
        ax.plot(sx+r*np.cos(th), sy+0.5*r*np.sin(th)+0.0, color=color, lw=1.1, alpha=0.55-0.10*i, zorder=40)

def beam(ax,x,y,zev,color):
    p0=iso(x,y,0.05); p1=iso(x,y,zev)
    ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=color,lw=1.0,ls=(0,(2,2)),alpha=0.6,zorder=30)

def hazard_icon(ax,x,y,z):
    sx,sy=iso(x,y,z)
    ax.add_patch(Circle((sx,sy),0.55,fc=RED,ec="none",alpha=0.16,zorder=44))
    tri=Polygon([(sx,sy+0.40),(sx-0.40,sy-0.30),(sx+0.40,sy-0.30)],closed=True,fc=RED,ec="white",lw=1.6,zorder=45)
    ax.add_patch(tri); ax.text(sx,sy-0.06,"!",ha="center",va="center",color="white",fontsize=11,fontweight="bold",zorder=46)

def ped_icon(ax,x,y,z,color=TEAL):
    sx,sy=iso(x,y,z)
    ax.add_patch(Circle((sx,sy),0.5,fc=color,ec="none",alpha=0.16,zorder=44))
    ax.add_patch(Circle((sx,sy+0.28),0.13,fc=color,ec="white",lw=1.0,zorder=45))
    ax.add_patch(Polygon([(sx,sy+0.16),(sx-0.16,sy-0.30),(sx+0.16,sy-0.30)],closed=True,fc=color,ec="white",lw=1.0,zorder=45))

def cloud(ax,cx,cy,scale=1.0,label="Cloud"):
    for dx,dy,r in [(-0.6,0,0.42),(-0.15,0.16,0.5),(0.4,0.05,0.46),(0.0,-0.12,0.55),(0.75,-0.05,0.34)]:
        ax.add_patch(Circle((cx+dx*scale,cy+dy*scale),r*scale,fc="#eef4fb",ec="#9bb8d6",lw=1.3,zorder=60))
    ax.text(cx+0.05*scale,cy-0.02*scale,label,ha="center",va="center",fontsize=7.5,color="#4a6f99",fontweight="bold",zorder=61)

# ---------------- build scene ----------------
fig,ax=plt.subplots(figsize=(7.4,4.7)); ax.axis("off")

# ground + road
G=14.0
poly(ax,[(0,0,0),(G,0,0),(G,G,0),(0,G,0)],GRASS,1,ec="#d4ddd6",lw=1.0)
poly(ax,[(0,6,0),(G,6,0),(G,8,0),(0,8,0)],ROAD,2,ec="none")     # horizontal road
poly(ax,[(6,0,0),(8,0,0),(8,G,0),(6,G,0)],ROAD,2,ec="none")     # vertical road
# lane dashes
for xx in np.arange(0.6,G,1.1):
    if not (6<xx<8):
        p0=iso(xx,7.0,0.02); p1=iso(xx+0.5,7.0,0.02)
        ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color="white",lw=1.0,zorder=3)
for yy in np.arange(0.6,G,1.1):
    if not (6<yy<8):
        p0=iso(7.0,yy,0.02); p1=iso(7.0,yy+0.5,0.02)
        ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color="white",lw=1.0,zorder=3)

# buildings (context) — a couple of grey blocks in corners (drawn back-first)
isobox(ax,1.0,9.5,2.2,2.2,2.2,"#c9cdd2",5)
isobox(ax,10.2,9.6,2.0,2.0,2.8,"#d2d6da",5)

# RSU + edge server (two), with signal
def rsu(ax,x,y,z):
    isobox(ax,x,y,0.22,0.22,3.0,"#7a7f85",z)            # pole
    isobox(ax,x-0.45,y-0.45,1.1,1.1,0.85,BLUE,z+1, topf=1.05,leftf=0.8,rightf=0.62)  # edge server cabinet
    # server lights
    sx,sy=iso(x+0.1,y+0.1,3.95); ax.add_patch(Circle((sx,sy),0.05,fc=TEAL,ec="none",zorder=z+3))
    signal(ax,x,y,4.2,color=TEAL)
rsu(ax,3.4,3.4,7)
rsu(ax,11.2,11.2,8)

# floating events with beams + coverage links from nearby cars
hazard_icon(ax,7.0,7.0,2.7); beam(ax,7.0,7.0,2.7,RED)
ped_icon(ax,11.0,7.0,2.3,TEAL); beam(ax,11.0,7.0,2.3,TEAL)
ped_icon(ax,3.0,7.0,2.3,ORANGE); beam(ax,3.0,7.0,2.3,ORANGE)   # second event (merge proxy)

# vehicles (front-most drawn last / higher zorder)
car(ax,4.6,6.25,TEAL,12,facing="x")
car(ax,2.4,6.95,BLUE,13,facing="x")
car(ax,9.4,6.30,ORANGE,12,facing="x")
car(ax,6.30,2.6,"#5a8f3a",14,facing="y")
car(ax,6.95,9.7,"#9aa0a6",11,facing="y")

# coverage links (vehicle -> event), some converging on hazard (redundancy hint)
def link(ax,x,y,ex,ey,ez,color,z=35):
    p0=iso(x+0.4,y+0.3,0.4); p1=iso(ex,ey,ez)
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-",color=color,lw=1.3,alpha=0.75,zorder=z))
link(ax,4.6,6.25,7.0,7.0,2.7,TEAL); link(ax,2.4,6.95,7.0,7.0,2.7,BLUE)   # two on hazard
link(ax,9.4,6.30,11.0,7.0,2.3,ORANGE)                                     # one on ped
link(ax,6.30,2.6,7.0,7.0,2.7,"#5a8f3a")

# cloud + uplink to RSU
cloud(ax,iso(13.5,13.5,3.0)[0],iso(13.5,13.5,3.0)[1]+1.4,scale=1.25,label="Cloud\n(meta-refit)")
p0=iso(11.2,11.2,4.2); p1=(iso(13.5,13.5,3.0)[0],iso(13.5,13.5,3.0)[1]+0.6)
ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",color="#9bb8d6",lw=1.4,ls=(0,(4,2)),mutation_scale=11,zorder=58))

# title
ax.text(0.5,0.995,"SERVO over the V2X sensing--communication--computing continuum",
        transform=ax.transAxes,ha="center",va="top",fontsize=9.5,fontweight="bold")

# legend chips (top-left, vertical)
chips=[("vehicle / service flow",TEAL),("RSU + edge server",BLUE),("semantic event",RED),("wireless / coverage","#9bb8d6")]
lx,ly=0.015,0.93
for i,(t,c) in enumerate(chips):
    ax.add_patch(Circle((lx,ly-i*0.045),0.011,transform=ax.transAxes,fc=c,ec="none",zorder=71))
    ax.text(lx+0.022,ly-i*0.045,t,transform=ax.transAxes,fontsize=6.6,va="center",color="#333",zorder=71)

# value-fair callout (horizontal strip across the bottom, 3 clean lines)
bx,by,bw,bh=0.05,0.015,0.90,0.215
ax.add_patch(FancyBboxPatch((bx,by),bw,bh,transform=ax.transAxes,
             boxstyle="round,pad=0.008,rounding_size=0.015",fc="#eafaf3",ec=TEAL,lw=1.4,zorder=70))
ax.text(0.5,by+bh-0.030,"Submodular value-fair problem",transform=ax.transAxes,
        ha="center",va="top",fontsize=8.5,fontweight="bold",color=TEAL,zorder=71)
ax.text(0.5,by+bh*0.50,
        r"$\max_{S\in\mathcal{I}}\ V(S)=\sum_{\omega\in\Omega} \mathrm{imp}(\omega)\,\max_{k\in S:\,\omega\in\mathrm{cov}(k)} q_k(\omega)$",
        transform=ax.transAxes,ha="center",va="center",fontsize=9,color=INK,zorder=71)
ax.text(0.5,by+0.038,
        r"$\mathcal{I}$: transversal matroid + budgets  $\cdot$  greedy $(1-1/e-\varepsilon)$-optimal  $\cdot$  $V$ learned online by GNN",
        transform=ax.transAxes,ha="center",va="center",fontsize=6.4,color="#333",zorder=71)

# set limits from iso extent (extra room at bottom for the strip)
pts=[iso(0,0,0),iso(G,0,0),iso(G,G,0),iso(0,G,0),iso(G,G,7)]
xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
ax.set_xlim(min(xs)-1.0,max(xs)+1.0); ax.set_ylim(min(ys)-6.0,max(ys)+2.6)
ax.set_aspect("equal")
fig.savefig(f"{OUT}/fig_system_model.pdf",bbox_inches="tight")
fig.savefig(f"{OUT}/fig_system_model.png",bbox_inches="tight")
print("system model (iso) saved")
