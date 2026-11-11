"""Draw a fixed-query prediction/cutoff path without causal or coverage claims."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VENUE = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_introduction():
    """Render a controlled three-policy path; retain all four cells in provenance."""
    source = VENUE / 'results/comparison-mechanism/summary.json'
    evidence = json.loads(source.read_text())
    group = evidence['groups']['DfE:current:2023']
    methods = ['lagged_factor', 'context_forest']
    cells = {(r['margin'], r['threshold']): r for r in group['crossed_guards']}
    q = group['q_by_case']['DfE-current-2023']
    answers = [[cells[(m,t)]['answered'] for t in methods] for m in methods]
    errors = [[cells[(m,t)]['wrong'] for t in methods] for m in methods]
    # Invariants tie the conceptual contrast to identical, native-resolved queries.
    assert group['resolved'] == 302
    assert all(cells[(m,t)]['resolved'] == 302 for m in methods for t in methods)
    assert answers == [[154,190],[149,181]] and errors == [[0,0],[0,0]]
    assert q[methods[0]] > q[methods[1]]
    same_cutoff_differences = [answers[1][j]-answers[0][j] for j in range(2)]
    assert same_cutoff_differences == [-5,-9]
    assert answers[1][1]-answers[0][0] == 27
    assert (answers[1][0]-answers[0][0])+(answers[1][1]-answers[1][0]) == 27

    # The full renderer crops other assets tightly; this fixed canvas must not inherit it.
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8.5,
                         'pdf.fonttype':42,'savefig.bbox':None}):
        fig, ax = plt.subplots(figsize=(3.35, 2.12))
        fig.subplots_adjust(left=.16, right=.99, bottom=.22, top=.87)
        # Start/end are totals from zero; the interior bars are signed changes.
        # Connectors mark the actual intermediate level, not interpolated policies.
        levels = [answers[0][0], answers[1][0], answers[1][1]]
        ax.bar(0, levels[0], width=.58, color='#647d8d', zorder=3)
        ax.bar(1, levels[0]-levels[1], bottom=levels[1], width=.58,
               color='#b66b35', zorder=3)
        ax.bar(2, levels[2]-levels[1], bottom=levels[1], width=.58,
               color='#31688e', zorder=3)
        ax.bar(3, levels[2], width=.58, color='#647d8d', zorder=3)
        for left, right, level in [(.29,.71,154),(1.29,1.71,149),(2.29,2.71,181)]:
            ax.plot([left,right],[level,level],ls='--',lw=.7,color='#69747b',zorder=2)
        for x,y,label in [(0,160,'154'),(1,163,'-5'),(2,187,'+32'),(3,187,'181')]:
            ax.text(x,y,label,ha='center',va='bottom',fontsize=10,weight='bold')
        ax.text(1,136,'149',ha='center',va='top',fontsize=8,color='#555d63')
        ax.set(xlim=(-.55,3.55),ylim=(0,218),yticks=[0,50,100,150,200],
               xticks=[0,1,2,3],xticklabels=['Lagged\nown policy','Swap only\npredictions',
                                            'Use forest\ncutoff','Forest\nown policy'])
        ax.set_ylabel('Answered queries',fontsize=8)
        ax.tick_params(axis='both',labelsize=7.8,length=2,pad=3)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#8a9398')
        ax.grid(axis='y',color='#e7ebee',lw=.6,zorder=0)
        fig.text(.575,.975,'Shown path: -5 +32 = +27 answers',
                 ha='center',va='top',fontsize=8.3,weight='bold')
        (VENUE/'figures').mkdir(exist_ok=True)
        pdf=VENUE/'figures/reuse-example.pdf'; png=pdf.with_suffix('.png')
        # A fixed canvas preserves the intended publication size; no tight crop rescales text.
        fig.savefig(pdf,metadata={'Title':'Decomposing answer yield on fixed queries','Author':''})
        fig.savefig(png,dpi=180); plt.close(fig)
    record=dict(source_path=str(source.relative_to(VENUE)),source_sha256=digest(source),
                group='DfE:current:2023',resolved_queries=302,answers=answers,errors=errors,
                cutoff=q,shared_cutoff_forest_minus_factor=same_cutoff_differences,
                own_policy_forest_minus_factor=27,diagonal_path_changes=[-5,32],
                renderer_sha256=digest(Path(__file__)),asset='figures/reuse-example.pdf',
                asset_sha256=digest(pdf),size_inches=[3.35,2.12],
                displayed_path=[154,149,181],alternate_path=[154,190,181],
                encoding='Zero-based total bars and signed change bars; each step changes one policy component',
                scope='Empirical illustration of retained development diagnostics, not new validation or a coverage guarantee for crossed cells')
    (VENUE/'records/cutoff-figure.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=="__main__":render_introduction()
