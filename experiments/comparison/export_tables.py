"""Generate plain CSV comparisons and named metrics from recorded numeric results."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = {'literal':'Literal', 'family_factor':'Factor', 'isotonic':'Isotonic',
         'context_ridge':'Ridge', 'context_forest':'Forest', 'lagged_factor':'Lagged'}


def count(value):
    """Serialise nonnegative integral totals as counts, never rounded rates."""
    if isinstance(value,bool) or not math.isfinite(value) or value < 0 or not float(value).is_integer():
        raise ValueError('Expected a nonnegative integer count')
    return int(value)


def main():
    def load(group):
        return json.loads((ROOT/'results'/group/'summary.json').read_text())

    def write(name, fields, rows):
        with (ROOT/'tables'/name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    baseline, robust, rank, mechanism, scope = map(load, ['comparison-panel',
        'comparison-robustness', 'comparison-ranking-boundary', 'comparison-mechanism', 'calibration-scope'])
    verification = json.loads((ROOT/'results/comparison-robustness/verification.json').read_text())
    methods = list(LABEL)[:-1]
    rows, metrics = [], {}
    for source, label in [('DHSC','Adult'), ('DfE','Child')]:
        values = baseline['summaries'][source+':geographic']
        for method in methods:
            point, guard = (values['methods'][method+':'+p] for p in ['point','guard'])
            rows.append([label, LABEL[method], f"{point['mape']:.2f}", f"{point['wape']:.2f}",
                         f"{100*point['risk']:.2f}", f"{100*guard['coverage_resolved']:.2f}",
                         guard['wrong_on_resolved'], guard['answers_on_resolved']])
            if method in ['family_factor','context_forest']:
                prefix = label.lower() + '_' + ('factor' if method=='family_factor' else 'forest') + '_'
                metrics.update({prefix+'mape':point['mape'], prefix+'point_error':point['risk'],
                                prefix+'answer_coverage':guard['coverage_resolved'],
                                prefix+'wrong':guard['wrong_on_resolved'], prefix+'answered':guard['answers_on_resolved']})
        point = rank['summaries'][source+':original']['methods']['pairwise_logistic']['point']
        rows.append([label,'Pairwise rank','','',f"{100*point['risk']:.2f}",'','',''])
    columns = ['Source','Method','MAPE_percent','WAPE_percent','Point_error_percent',
               'Guarded_answers_percent','Wrong','Answered']
    write('geographic.csv', columns, rows)
    rows = []
    for year in [2020,2023,2025]:
        for method in LABEL:
            selected = [r for r in robust['summaries'] if r['group']=='DfE:current:'+str(year)
                        and r['method']==method and r['family']=='all']
            point = next(r for r in selected if r['policy']=='point')
            guard = next(r for r in selected if r['policy']=='alpha=0.05')
            rows.append([year,LABEL[method],f"{point['mape']:.2f}",f"{point['wape']:.2f}",
                         f"{100*point['risk']:.2f}",f"{100*guard['coverage']:.2f}",count(guard['wrong']),count(guard['answered'])])
    write('temporal.csv', ['Year',*columns[1:]], rows)
    n = verification['overlapping_2024_2025_authority_year_vectors']
    anchors = robust['vintage_anchor_changes']
    write('vintage.csv', ['Comparison','Checked','Changed'], [
        ['Historical authority-year vectors',n,n-verification['identical_selected_count_and_region_vectors']],
        ['2024 native anchors',len(anchors),sum(r['anchor_current']!=r['anchor_preceding'] for r in anchors)],
        ['Change reference directions',len(anchors),sum(r['direction_changed'] for r in anchors)],
        ['Matched test authorities',len(anchors)//2,0]])
    groups = [('Adult',p,'DHSC:'+k) for p,k in [('original','original'),('salt 1','salt-20261104'),('salt 2','salt-20261105')]]
    groups += [('Child',p,'DfE:'+k) for p,k in [('original','original'),('salt 1','salt-20261104'),('salt 2','salt-20261105')]]
    groups += [('Child temporal',str(y),'DfE:current:'+str(y)) for y in [2020,2023,2025]]
    rows = []
    for task, label, key in groups:
        group = mechanism['groups'][key]; transitions = group['transitions']; cross = group['crossed_guards']
        cells = [value for i in [0,2,1,3] for value in [cross[i]['answered'],cross[i]['wrong']]]
        rows.append([task,label,group['resolved'],transitions['repair'],transitions['damage'],transitions['point_tie'],*cells])
    write('mechanisms.csv', ['Task','Partition_or_year','Resolved','Repair','Damage','Tie',
          *[label+'_'+stat for label in ['B_B','F_B','B_F','F_F'] for stat in ['answered','wrong']]], rows)
    rows = []
    for source, label in [('DHSC','Adult geographic'),('DfE','Child geographic')]:
        for method in ['family_factor','context_forest']:
            values = []
            for choice in ['specified-family','Bonferroni-vector']:
                row = next(r for r in scope['summaries'] if r['group']==source+':original'
                           and r['method']==method and r['scope']==choice and r['family']=='all' and r['alpha']==.05)
                values.extend([row['wrong'],row['answered']])
            rows.append([label,LABEL[method],*values])
    write('calibration-scope.csv', ['Source','Converter','Family_wrong','Family_answered','Bonferroni_wrong','Bonferroni_answered'], rows)
    write('cohorts.csv', ['Source','Target','Areas','Publication_radius'], [
        [source, family, baseline['summaries'][source+':geographic']['authorities'],radius]
        for source, families, radius in [('DHSC',['Requests 18-64','Requests 65+'],2),
                                        ('DfE',['Episodes','Referrals'],0)] for family in families])
    (ROOT/'tables/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Generated six CSV tables and named numeric metrics')


if __name__ == '__main__':
    main()
