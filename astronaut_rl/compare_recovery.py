"""Paired physical-game outcomes: the delivered weak policies versus the recovery."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
from .experiment import evaluate,evaluation_summary
from .qlearning import QLearner

METRICS=('frames_mean','survival_rate','oxygen_mean','oxygen_per_1000_frames','combined_success_rate')
def stats(values):
    return {'mean':statistics.fmean(values),'sd':statistics.stdev(values) if len(values)>1 else 0.,'n':len(values)}
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def frozen(path,record,seeds):
    a=QLearner.load(path);q={s:v[:] for s,v in a.q.items()};visits={s:v[:] for s,v in a.visits.items()};rng=a.rng.getstate()
    rows=evaluate(a,record['variant'],seeds,repeat=a.metadata['repeat'])
    if a.q!=q or a.visits!=visits or a.rng.getstate()!=rng:raise ValueError('Comparison changed a frozen policy')
    return rows

def compare(results,models,baseline,out):
    results,models,baseline,out=map(Path,(results,models,baseline,out));out.mkdir(parents=True,exist_ok=True)
    new=json.loads((results/'run_manifest.json').read_text())
    old=json.loads((baseline/'run_manifest.json').read_text())
    before={(r['variant'],r['seed']):r for r in old['records']}
    episode_rows=[];pairs=[];summaries={};provenance=[]
    for record in new['records']:
        key=record['variant'],record['seed']; prior=before[key]
        seeds=record['protocol']['test_seeds']
        if record.get('qualified') is False:raise ValueError('Unqualified recovery policy')
        this={}
        for name,root,r in [('before',baseline/'models',prior),('after',models,record)]:
            path=root/r['model_path'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if sha!=r['model_sha256']:raise ValueError('Model hash changed: '+str(path))
            rows=frozen(path,r,seeds);this[name]=rows
            summaries.setdefault(key[0],{}).setdefault(name,[]).append(evaluation_summary(rows))
            episode_rows.extend({'policy':name,'variant':key[0],'training_seed':key[1],**x} for x in rows)
            provenance.append({'policy':name,'variant':key[0],'training_seed':key[1],'filename':path.name,'sha256':sha,'repeat':r['repeat']})
        for a,b in zip(this['before'],this['after']):
            if a['seed']!=b['seed']:raise ValueError('Unpaired episode seeds')
            pairs.append({'variant':key[0],'training_seed':key[1],'seed':a['seed'],'before_frames':a['frames'],'after_frames':b['frames'],'before_survived':a['survived'],'after_survived':b['survived'],'before_oxygen':a['oxygen'],'after_oxygen':b['oxygen']})
    aggregate={}
    for variant,groups in summaries.items():
        aggregate[variant]={name:{metric:stats([x[metric] for x in rows]) for metric in METRICS} for name,rows in groups.items()}
        aggregate[variant]['change']={metric:stats([b[metric]-a[metric] for a,b in zip(groups['before'],groups['after'])]) for metric in METRICS}
    result={'variants':aggregate,'training_seeds':new['training_seeds'],'models':provenance,'interpretation':['Same pinned official physics and default rewards, same final episode seeds.','Each policy is frozen; results are means/sample SD of policy means, not independent training replicates per episode.','This measures the complete recovery pipeline, not the isolated causal effect of each feature, control or initialization change.','The older custom-reward/different-physics project is excluded.']}
    write(out/'episodes.csv',episode_rows);write(out/'paired_episodes.csv',pairs)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Paired recovery comparison','',
        'Both policy sets use the same pinned official physics, default rewards and final episode seeds. Values below are means of five policy means. The JSON also records sample SD across policies. This compares the complete recovery pipeline, including different state/action settings, training exposure and skill transfer; it does not isolate the effect of one change.','',
        '| Task | Before survival | After survival | Before tanks | After tanks | Before frames | After frames |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for variant,entry in aggregate.items():
        before,after=entry['before'],entry['after']
        lines.append('| %s | %.1f%% | %.1f%% | %.2f | %.2f | %.1f | %.1f |' % (
            variant,100*before['survival_rate']['mean'],100*after['survival_rate']['mean'],
            before['oxygen_mean']['mean'],after['oxygen_mean']['mean'],
            before['frames_mean']['mean'],after['frames_mean']['mean']))
    lines += ['', 'Each row of `paired_episodes.csv` matches the same training-seed identifier and episode seed. Both policies are frozen during evaluation. Historical custom-reward/old-physics models are excluded.','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate),flush=True);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--models',type=Path,default=Path('models'));p.add_argument('--baseline',type=Path,default=Path('baselines/independent_before_recovery'));p.add_argument('--out',type=Path,default=Path('results/recovery_comparison'));a=p.parse_args();compare(a.results,a.models,a.baseline,a.out)
if __name__=='__main__':main()
