"""Verify model provenance and replay every saved test/validation evaluation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from .config import config_hash
from .environment import ENV_COMMIT, VARIANTS, check_environment
from .experiment import evaluate, evaluation_summary, coverage_summary, derive_gate, gate_passed
from .qlearning import QLearner
from .state import state_capacity
from .transfer import PARENTS, verify_initialization


def compare(actual, expected, context):
    if isinstance(expected, dict):
        if not isinstance(actual,dict):
            raise ValueError('Expected a dictionary: '+context)
        for key,value in expected.items():
            if key not in actual:
                raise ValueError('Missing field: '+context+'.'+key)
            compare(actual[key],value,context+'.'+key)
    elif isinstance(expected,(int,float)):
        if not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError(f'Mismatch {context}: {actual} != {expected}')
    elif actual!=expected:
        raise ValueError(f'Mismatch {context}: {actual!r} != {expected!r}')


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    return [{key:(value if key in ('collision','variant','model_identifier','model_sha256') else float(value))
             for key,value in row.items()} for row in rows]


def model_path(root,relative):
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute():
        raise ValueError('Model paths must be nonempty relative paths')
    root=Path(root).resolve(); path=(root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Model path escapes the supplied models directory')
    return path


def checked_agent(root,relative,checksum,record,updates,role=None):
    path=model_path(root,relative)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:
        raise ValueError('Model checksum changed: '+str(path))
    agent=QLearner.load(path)
    metadata=agent.metadata
    for key in ('variant','seed','profile','repeat','alpha','gamma','budget','state_capacity',
                'epsilon_start','epsilon_end','epsilon_decay_fraction','gamma_per_frame','initialization',
                'initialized_rows','transferred_rows','budget_unit','environment_commit','config_hash','state_spec','protocol'):
        compare(metadata[key],record[key],path.name+'.'+key)
    compare(metadata['config'],record['config'],path.name+'.config')
    compare(config_hash(metadata['config']),record['config_hash'],path.name+'.config_checksum')
    if metadata['profile']!='official' or metadata['environment_commit']!=ENV_COMMIT:
        raise ValueError('Model is not part of the pinned official-reward experiment')
    compare(metadata['updates'],updates,path.name+'.updates')
    compare(metadata['checkpoint_step'],updates,path.name+'.checkpoint_step')
    compare(agent.alpha,record['alpha'],path.name+'.actual_alpha')
    compare(agent.gamma,record['gamma'],path.name+'.actual_gamma')
    compare(agent.seed,record['seed'],path.name+'.actual_seed')
    compare(agent.actions,2,path.name+'.actual_actions')
    if role is not None:
        compare(metadata['model_role'],role,path.name+'.model_role')
        compare(metadata['selected_step'],record['selected_step'],path.name+'.selected_step')
        compare(metadata['qualified'],record['qualified'],path.name+'.qualified')
        compare(metadata['gate'],record['gate'],path.name+'.gate')
    compare(sum(sum(row) for row in agent.visits.values()),updates,path.name+'.visit_count')
    compare(state_capacity(record['variant'],metadata['state_spec']),record['state_capacity'],path.name+'.capacity')
    return agent


def frozen_evaluation(agent,record,seeds,random_policy=False):
    before_q={state:row[:] for state,row in agent.q.items()}
    before_visits={state:row[:] for state,row in agent.visits.items()}
    before_rng=agent.rng.getstate()
    rows=evaluate(agent,record['variant'],seeds,record['repeat'],random_policy=random_policy)
    if agent.q!=before_q or agent.visits!=before_visits or agent.rng.getstate()!=before_rng:
        raise ValueError('Frozen evaluation changed the agent')
    return rows


def compare_rows(actual,expected,label):
    if len(actual)!=len(expected):
        raise ValueError('Episode count differs: '+label)
    for index,(row,saved) in enumerate(zip(actual,expected)):
        compare(row,saved,f'{label}[{index}]')


def checked_records(manifest):
    records=manifest.get('records'); seeds=manifest.get('training_seeds')
    if not isinstance(records,list) or not records:
        raise ValueError('The manifest contains no completed model records')
    if not isinstance(seeds,list) or not seeds or len(seeds)!=len(set(seeds)) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('Invalid manifest training seed list')
    variants=manifest.get('variants')
    if not variants or variants!=list(VARIANTS[:len(variants)]):
        raise ValueError('Manifest must describe a prefix-ordered curriculum')
    seen={}; actual=[]
    for record in records:
        key=(record['variant'],record['seed'])
        if key[0] not in variants or type(key[1]) is not int or key[1] not in seeds:
            raise ValueError('Invalid variant or training seed in model record')
        if key in seen: raise ValueError('Duplicate independent model record')
        seen[key]=record; actual.append(key)
    expected=[]; failures=[]
    for seed in seeds:
        for index,variant in enumerate(variants):
            key=(variant,seed)
            if key not in seen: raise ValueError('Incomplete training seed set or curriculum chain')
            expected.append(key)
            if seen[key]['qualified'] is not True:
                failures.append((seed,variant))
                if any((v,seed) in seen for v in variants[index+1:]):
                    raise ValueError('Dependent stage advanced after a failed parent gate')
                break
    if actual!=expected:
        raise ValueError('Records are not ordered flight -> asteroids -> oxygen within each seed')
    if bool(manifest.get('curriculum_passed')) != (not failures):
        raise ValueError('Manifest curriculum qualification differs from stage gates')
    declared=[(row['seed'],row['variant']) for row in manifest.get('failed_stages',[])]
    if declared!=failures: raise ValueError('Failed-stage declarations differ from completed records')
    return records


def _selection_key(variant,row,gate):
    if variant!='oxygen': return row['survival_rate'],row['frames_mean'],row['return_mean']
    if row['survival_rate']>=gate['minimum_survival_rate']:
        return 1,row['oxygen_mean'],row['survival_rate'],row['frames_mean']
    return 0,row['survival_rate'],row['oxygen_mean'],row['frames_mean']


def checked_history(record,prefix):
    config=record['config']; protocol=record['protocol']; budget=record['budget']
    if type(budget) is not int or budget<1: raise ValueError('Invalid decision budget: '+prefix)
    if config_hash(config)!=record['config_hash']: raise ValueError('Configuration checksum changed: '+prefix)
    for key in ('variant','profile','repeat','alpha','gamma','gamma_per_frame','epsilon_start','epsilon_end',
                'epsilon_decay_fraction','state_spec','state_capacity','environment_commit','budget_unit'):
        compare(record[key],config[key],prefix+'.config.'+key)
    compare(config['steps'],budget,prefix+'.config.steps')
    compare(record['profile'],'official',prefix+'.official_profile')
    compare(record['environment_commit'],ENV_COMMIT,prefix+'.environment_commit')
    compare(record['budget_unit'],'decision_updates',prefix+'.budget_unit')
    if type(record['repeat']) is not int or record['repeat']<1: raise ValueError('Invalid action repeat')
    compare(record['gamma'],record['gamma_per_frame']**record['repeat'],prefix+'.physical_discount_units')
    compare(protocol['budget_unit'],'decision_updates',prefix+'.protocol.budget_unit')
    compare(protocol['repeat'],record['repeat'],prefix+'.protocol.repeat')
    compare(protocol['gamma_per_frame'],record['gamma_per_frame'],prefix+'.protocol.gamma_per_frame')
    compare(protocol['reward'],'unchanged_official',prefix+'.protocol.reward')
    for key in ('validation_seeds','test_seeds'):
        compare(protocol[key],config[key],prefix+'.protocol.'+key)
        seeds=protocol[key]
        if not seeds or len(seeds)!=len(set(seeds)) or any(type(s) is not int or s<0 for s in seeds):
            raise ValueError('Invalid '+key)
    if set(protocol['validation_seeds'])&set(protocol['test_seeds']) or record['seed'] in protocol['validation_seeds'] or record['seed'] in protocol['test_seeds']:
        raise ValueError('Training, validation and test seeds overlap')
    history=record['validation_history']; interval=config['checkpoint_interval']
    if type(interval) is not int or interval<1: raise ValueError('Invalid checkpoint interval')
    expected=[0]+list(range(interval,budget+1,interval))
    if expected[-1]!=budget: expected.append(budget)
    if not isinstance(history,list) or not history: raise ValueError('Missing validation checkpoints')
    compare([row['step'] for row in history],expected,prefix+'.checkpoint_schedule')
    gate=derive_gate(config,history[0])
    compare(protocol['gate'],gate,prefix+'.protocol.gate')
    compare(protocol['validation_target'],{'survival_rate':gate['minimum_survival_rate'],
        'oxygen_mean':gate['minimum_oxygen_mean']},prefix+'.protocol.validation_target')
    compare(protocol['stability_required_checkpoints'],config['required_consecutive_passes'],prefix+'.required_passes')
    compare(record['gate'],gate,prefix+'.gate_thresholds')
    selection='survival_constraint_then_collection' if record['variant']=='oxygen' else 'survival_first_lexicographic'
    compare(protocol['selection'],selection,prefix+'.selection')
    consecutive=0; first=stable=None
    for index,row in enumerate(history):
        step=row['step']; count=row['frames']
        compare(row['checkpoint_idx'],index+1,prefix+'.checkpoint_idx')
        compare(row['decision'],step,prefix+'.checkpoint.decision')
        if not step<=count<=step*record['repeat']: raise ValueError('Impossible checkpoint physical frame count')
        compare(row['variant'],record['variant'],prefix+'.checkpoint.variant')
        compare(row['seed'],record['seed'],prefix+'.checkpoint.seed')
        compare(row['selected'],int(step==record['selected_step']),prefix+'.selected_flag')
        passing=gate_passed(row,gate); consecutive=consecutive+1 if passing else 0
        compare(row['eligible'],int(row['survival_rate']>=gate['minimum_survival_rate']),prefix+'.eligible')
        for key in ('passing','target_met'): compare(row[key],int(passing),prefix+'.'+key)
        compare(row['consecutive_passes'],consecutive,prefix+'.consecutive_passes')
        if passing and first is None: first=step
        if consecutive>=gate['required_consecutive_passes'] and stable is None: stable=step
    selected=max(history,key=lambda row:(_selection_key(record['variant'],row,gate),-row['step']))
    compare(selected['step'],record['selected_step'],prefix+'.validation_selection')
    passed=gate_passed(selected,gate); qualified=bool(passed and stable is not None)
    compare(record['qualified'],qualified,prefix+'.qualified')
    compare(record['gate']['selected_passed'],passed,prefix+'.gate.selected_passed')
    compare(record['gate']['qualified'],qualified,prefix+'.gate.qualified')
    compare(record['gate']['stability_step'],stable,prefix+'.gate.stability_step')
    compare(record['first_target_step'],first,prefix+'.first_target_step')
    compare(record['stability_step'],stable,prefix+'.stability_step')
    keys=evaluation_summary([]).keys()
    compare(record['selected']['step'],record['selected_step'],prefix+'.selected.step')
    compare(record['selected']['checkpoint_path'],selected['model_identifier'],prefix+'.selected.checkpoint_path')
    compare(record['selected']['validation'],{key:selected[key] for key in keys},prefix+'.selected.validation')
    compare(record['selected']['greedy'],record['greedy'],prefix+'.selected.greedy')
    compare(record['final']['step'],budget,prefix+'.final.step')
    compare(record['final']['validation'],{key:history[-1][key] for key in keys},prefix+'.final.validation')
    compare(record['final']['model_path'],record['final_model_path'],prefix+'.final.model_path')
    compare(record['final']['model_sha256'],record['final_model_sha256'],prefix+'.final.model_sha256')
    compare(record['final']['coverage'],record['coverage'],prefix+'.final.coverage')
    return history


def verify(results,models,checkpoints=True):
    check_environment(); results,models=Path(results),Path(models)
    manifest=json.loads((results/'run_manifest.json').read_text())
    if manifest.get('complete') is not True: raise ValueError('The experiment manifest is incomplete')
    records=checked_records(manifest); checked=checkpoint_count=0
    seen_configs={}; previous_records={}; previous_agents={}
    for record in records:
        prefix=f"{record['variant']}_seed{record['seed']}"; history=checked_history(record,prefix)
        previous=seen_configs.setdefault(record['variant'],record['config_hash'])
        if previous!=record['config_hash']: raise ValueError('Settings differ between training seeds of one task')
        compare(record['updates'],record['budget'],prefix+'.updates')
        if not record['budget']<=record['frames']<=record['budget']*record['repeat']:
            raise ValueError('Impossible training physical frame count')
        compare(history[-1]['frames'],record['frames'],prefix+'.final_physical_frames')
        selected=checked_agent(models,record['model_path'],record['model_sha256'],record,record['selected_step'],'selected')
        final=checked_agent(models,record['final_model_path'],record['final_model_sha256'],record,record['budget'],'final')
        selected_row=next(row for row in history if row['selected'])
        compare(selected.metadata['physics_frames'],selected_row['frames'],prefix+'.selected_physical_frames')
        compare(final.metadata['physics_frames'],record['frames'],prefix+'.final_physical_frames')
        compare(coverage_summary(final,record['state_capacity']),record['coverage'],prefix+'.coverage')
        compare(coverage_summary(selected,record['state_capacity']),record['selected']['coverage'],prefix+'.selected_coverage')
        zero=checked_agent(models,history[0]['model_identifier'],history[0]['model_sha256'],record,0)
        initialization=record['initialization']; parent_variant=PARENTS[record['variant']]
        if parent_variant is None:
            compare(initialization['type'],'zeros',prefix+'.initialization')
            if zero.q or any(sum(row) for row in zero.visits.values()): raise ValueError('Flight did not start from zero')
            ancestor={'decision_updates':0,'physics_frames':0,'train_seconds':0.0}
        else:
            parent_key=(parent_variant,record['seed'])
            if parent_key not in previous_records or previous_records[parent_key]['qualified'] is not True:
                raise ValueError('Child started before a qualified same-seed parent')
            parent_record=previous_records[parent_key]; parent=previous_agents[parent_key]
            compare(initialization['type'],'parent_q',prefix+'.initialization')
            for key,expected in (('parent_variant',parent_variant),('parent_seed',record['seed']),
                ('parent_model_path',parent_record['model_path']),('parent_model_sha256',parent_record['model_sha256']),
                ('parent_updates',parent.metadata['updates']),('parent_physics_frames',parent.metadata['physics_frames'])):
                compare(initialization[key],expected,prefix+'.initialization.'+key)
            model_path(models,initialization['parent_model_path'])
            verify_initialization(zero,parent,record['variant'],record['state_spec'])
            ancestor=parent_record['cumulative']
        compare(record['initialized_rows'],len(zero.q),prefix+'.initialized_rows')
        compare(record['transferred_rows'],len(zero.q),prefix+'.transferred_rows')
        compare(initialization['transferred_rows'],len(zero.q),prefix+'.initialization.rows')
        cost={'decision_updates':record['budget'],'physics_frames':record['frames'],'train_seconds':record['train_seconds']}
        compare(record['stage_cost'],cost,prefix+'.stage_cost')
        compare(record['cumulative'],{key:value+ancestor[key] for key,value in cost.items()},prefix+'.cumulative')
        for policy,suffix in (('greedy','evaluation'),('random','random')):
            stored=read_csv(results/f'{prefix}_{suffix}.csv'); seeds=[int(row['seed']) for row in stored]
            if seeds!=record['protocol']['test_seeds']: raise ValueError('Final test seed list changed')
            actual=frozen_evaluation(selected,record,seeds,policy=='random')
            compare_rows(actual,stored,prefix+'.'+policy)
            compare(evaluation_summary(actual),record[policy],prefix+'.'+policy+'.summary')
            checked+=len(actual)
        for row in history: model_path(models,row['model_identifier'])
        compare_rows(history,read_csv(results/f'{prefix}_validation.csv'),prefix+'.validation_history_csv')
        episodes=read_csv(results/f'{prefix}_episodes.csv')
        compare(sum(row['frames'] for row in episodes),record['frames'],prefix+'.episode_frame_sum')
        compare(sum(row['decisions'] for row in episodes),record['budget'],prefix+'.episode_decision_sum')
        compare(sum(row['completed'] for row in episodes),record['episodes'],prefix+'.completed_episodes')
        for index,row in enumerate(episodes):
            if row['budget_interrupted'] and (index!=len(episodes)-1 or row['terminated'] or row['truncated'] or row['completed']):
                raise ValueError('Invalid interrupted episode')
        training=read_csv(results/f'{prefix}_training.csv')
        if not training: raise ValueError('Missing training loss rows')
        compare(training[-1]['step'],record['budget'],prefix+'.loss_last_step')
        compare(training[0]['td_mse'],record['first_td_mse'],prefix+'.first_loss')
        compare(training[-1]['td_mse'],record['last_td_mse'],prefix+'.last_loss')
        if checkpoints:
            stored=read_csv(results/f'{prefix}_validation_episodes.csv'); groups={}
            for row in stored:
                step=int(row.pop('checkpoint_step')); groups.setdefault(step,[]).append(row)
            if set(groups)!={int(row['step']) for row in history}: raise ValueError('Validation episode groups differ from checkpoint history')
            for row in history:
                step=int(row['step']); agent=checked_agent(models,row['model_identifier'],row['model_sha256'],record,step)
                compare(agent.metadata['physics_frames'],row['frames'],prefix+'.checkpoint.physical_frames')
                actual=frozen_evaluation(agent,record,record['protocol']['validation_seeds'])
                compare_rows(actual,groups[step],prefix+f'.validation{step}')
                compare(row,evaluation_summary(actual),prefix+'.validation_summary')
                compare(row,{'coverage_'+k:v for k,v in coverage_summary(agent,record['state_capacity']).items()},prefix+'.checkpoint_coverage')
                if step==record['selected_step'] and (selected.q!=agent.q or selected.visits!=agent.visits):
                    raise ValueError('Selected table differs from selected checkpoint')
                if step==record['budget'] and (final.q!=agent.q or final.visits!=agent.visits):
                    raise ValueError('Final table differs from final checkpoint')
                checked+=len(actual); checkpoint_count+=1
        previous_records[(record['variant'],record['seed'])]=record
        previous_agents[(record['variant'],record['seed'])]=selected
        print(f"PASS {prefix}: Q transfer, gates, macro discounts, hashes and frozen evaluations match (qualified={record['qualified']})",flush=True)
    print(f'Verified {checked} episodes, {checkpoint_count} validation checkpoints and {len(records)} stage models.',flush=True)
    return {'episodes':checked,'checkpoints':checkpoint_count,'models':len(records)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--models',type=Path,default=Path('models'))
    parser.add_argument('--skip-checkpoints',action='store_true')
    args=parser.parse_args(); verify(args.results,args.models,not args.skip_checkpoints)


if __name__=='__main__': main()
