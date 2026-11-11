"""Generate five statistical figures from verified result records; no fitting occurs."""
from pathlib import Path
import argparse
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator
import numpy as np

from comparison_panel import ROOT, digest

VENUE=ROOT
LABEL={'literal':'Literal','family_factor':'Factor','isotonic':'Isotonic','context_ridge':'Ridge','context_forest':'Forest','lagged_factor':'Lagged'}
COLOR={'literal':'#be4b45','family_factor':'#31688e','isotonic':'#727d86','context_ridge':'#80579a','context_forest':'#d08327','lagged_factor':'#278470'}
# Redundant shapes/styles preserve method identity without relying on colour.
MARKER={'literal':'s','family_factor':'o','isotonic':'^','context_ridge':'X','context_forest':'D','lagged_factor':'v'}
LINE={'literal':':','family_factor':'-','isotonic':'-.','context_ridge':'--','context_forest':'-','lagged_factor':'--'}

def set_risk_axis_limits(ax):
    """Include every finite policy error; undefined risks remain plot gaps.

    A policy with no answers has an undefined risk. Its NaNs must not shrink
    the scale of other policies. Keep the existing positive floor for zero or
    entirely undefined risks; y values are percentages.
    """
    finite = [float(y) for line in ax.lines for y in line.get_ydata() if np.isfinite(y)]
    upper = max(finite, default=0.)
    ax.set_ylim(0, max(.06, upper*1.15))


def main():
    baseline=json.loads((ROOT/'results/comparison-panel/summary.json').read_text())
    robust=json.loads((ROOT/'results/comparison-robustness/summary.json').read_text())
    fig=ROOT/'figures';fig.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','pdf.fonttype':42,'savefig.bbox':'tight'})
    def save(f,name):
        p=fig/(name+'.pdf');f.savefig(p,metadata={'CreationDate':None,'ModDate':None});f.savefig(fig/(name+'.png'),dpi=160);plt.close(f)
    def get(group,method,policy):
        return next(r for r in robust['summaries'] if r['group']==group and r['method']==method and r['policy']==policy and r['family']=='all')
    from render_cutoff import render_introduction
    render_introduction()
    methods=list(LABEL)[:-1]
    f,axs=plt.subplots(2,1,figsize=(3.5,4.0))
    for ax,source,title in zip(axs,['DHSC','DfE'],['Adult geographic','Child geographic']):
        for m in methods:
            r=baseline['summaries'][source+':geographic']['methods'][m+':point']
            ax.scatter(r['mape'],100*r['risk'],s=40,marker=MARKER[m],color=COLOR[m],label=LABEL[m],edgecolors='white',linewidths=.5)
        ax.set(xscale='log',xlabel='People-count error (MAPE, %)',ylabel='Point-answer error (%)',title=title)
        ax.set_ylim(0,5.5 if source=='DHSC' else 1.5)
        ax.set_xticks([8,10,20,40] if source=='DHSC' else [2,5,10,20,40])
        ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.15)
    handles,labels=axs[0].get_legend_handles_labels()
    order=[0,3,1,4,2]  # Matplotlib fills columns; retain the method order across rows.
    f.legend([handles[i] for i in order],[labels[i] for i in order],ncol=3,frameon=False,loc='lower center',fontsize=8.5)
    f.subplots_adjust(left=.2,right=.98,bottom=.2,top=.92,hspace=.65);save(f,'reuse-fit')
    f,axs=plt.subplots(2,2,figsize=(7,4.1),sharex=True)
    for row,source in enumerate(['DHSC','DfE']):
        for col,kind in enumerate(['alpha','budget']):
            ax=axs[row,col]
            for m in methods:
                rr=[get(source+':original',m,kind+'='+str(v)) for v in ([.025,.05,.1] if kind=='alpha' else [.25,.5,.75])]
                ax.plot([100*r['coverage'] for r in rr],[100*r['risk'] if r['risk'] is not None else np.nan for r in rr],marker=MARKER[m],linestyle=LINE[m],ms=4,lw=1.1,color=COLOR[m],label=LABEL[m],clip_on=False)
            # Expose settings without shifting points; rings pick out the headline
            # factor/forest comparison even when their answer fractions nearly coincide.
            if kind == 'alpha':
                for method in ('family_factor', 'context_forest'):
                    middle = get(source+':original', method, 'alpha=0.05')
                    if middle['risk'] is not None:
                        ax.scatter(100*middle['coverage'], 100*middle['risk'],
                                   s=62, facecolors='none', edgecolors=COLOR[method],
                                   linewidths=1.1, zorder=5, clip_on=False)
            setting = 'α: .025 → .05 → .10' if kind == 'alpha' else 'Budget: .25 → .50 → .75'
            ax.text(.025, .96, setting, transform=ax.transAxes, va='top', fontsize=7)
            ax.set(title=('Adult' if source=='DHSC' else 'Child')+(' calibrated guard' if kind=='alpha' else ' matched answer budget'),xlim=(0,100),ylabel='Error among answers (%)')
            set_risk_axis_limits(ax)
            ax.grid(alpha=.15)
    axs[1,0].set_xlabel('Answered queries (%)');axs[1,1].set_xlabel('Answered queries (%)')
    f.legend(*axs[0,0].get_legend_handles_labels(),ncol=5,frameon=False,loc='upper center',fontsize=8)
    f.subplots_adjust(left=.1,right=.98,bottom=.13,top=.84,wspace=.4,hspace=.55);save(f,'reuse-guards')
    f,axs=plt.subplots(1,3,figsize=(7,2.1));years=[2020,2023,2025]
    for m in ['family_factor','context_forest','lagged_factor']:
        pp=[get('DfE:current:'+str(y),m,'point') for y in years];gg=[get('DfE:current:'+str(y),m,'alpha=0.05') for y in years]
        for ax,val in zip(axs,[[r['mape'] for r in pp],[100*r['risk'] for r in pp],[100*r['coverage'] for r in gg]]):ax.plot(years,val,marker=MARKER[m],linestyle=LINE[m],ms=4,color=COLOR[m],label=LABEL[m])
    for ax,title in zip(axs,['Count error (MAPE, %)','Point-answer error (%)','Guarded answers (%)']):ax.set(title=title,xticks=years);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    f.legend(*axs[0].get_legend_handles_labels(),frameon=False,ncol=3,loc='upper center',fontsize=8.5)
    f.subplots_adjust(left=.065,right=.99,bottom=.18,top=.67,wspace=.5);save(f,'reuse-temporal')
    f,axs=plt.subplots(2,1,figsize=(3.5,4.0));parts=['original','salt-20261104','salt-20261105']
    for ax,source in zip(axs,['DHSC','DfE']):
        for m in methods:
            rr=[get(source+':'+p,m,'alpha=0.05') for p in parts]
            ax.plot(range(3),[100*r['coverage'] for r in rr],marker=MARKER[m],linestyle=LINE[m],ms=4,lw=1.1,color=COLOR[m],label=LABEL[m])
        ax.set(title=('Adult' if source=='DHSC' else 'Child')+' refitted partitions',xticks=range(3),xticklabels=['Original','Partition 1','Partition 2'],ylabel='Guarded answers (%)',ylim=(0,100));ax.grid(alpha=.15)
    handles,labels=axs[0].get_legend_handles_labels()
    f.legend([handles[i] for i in order],[labels[i] for i in order],ncol=3,frameon=False,loc='lower center',fontsize=8.5)
    f.subplots_adjust(left=.2,right=.98,bottom=.2,top=.92,hspace=.65);save(f,'reuse-splits')

    from export_tables import main as tables
    tables()

if __name__=="__main__":main()
