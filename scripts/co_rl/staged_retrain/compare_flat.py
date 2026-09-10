"""Two sequential scratch trials; only PPO clip differs. Use launch.sh."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TASK = 'Isaac-Velocity-Flat-Flamingo-Light-v1-ppo'


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--followup-clip',type=float)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((ROOT/'scripts/co_rl/staged_retrain/flat_baseline.json').read_text())
    alternative=json.loads((ROOT/'scripts/co_rl/staged_retrain/flat_clip01.json').read_text())
    differences={k for k in baseline.keys()|alternative.keys() if baseline.get(k)!=alternative.get(k)}
    if differences != {'agent.algorithm.clip_param'}:
        raise ValueError(f'Unexpected comparison differences: {differences}')
    if any(k.startswith('env.rewards') for k in baseline) or baseline['env.seed']!=42:
        raise ValueError('Invalid controlled trial config')
    state={'status':'running','trials':[], 'created':time.time(),
           'iterations_per_model':1000,'num_envs':1024,'rollout_steps':24,
           'training_seed':42,'only_difference':'agent.algorithm.clip_param'}
    planned=[('baseline',baseline),('clip01',alternative)]
    if (args.followup_clip is None) != (args.reference is None):
        raise ValueError('Follow-up clip and reference must be provided together')
    if args.reference is not None:
        if not 0 < args.followup_clip < 1:
            raise ValueError('Clip must be between zero and one')
        reference=json.loads((args.reference/'status.json').read_text())
        if reference['status']!='completed' or not reference['trials']:
            raise ValueError('Reference comparison must be completed')
        for trial in reference['trials']:
            delta={k for k in baseline.keys()|trial['config'].keys()
                   if baseline.get(k)!=trial['config'].get(k)}
            if trial['status']!='completed' or delta-{'agent.algorithm.clip_param'}:
                raise ValueError('Reference configuration is not comparable')
        state['trials']=reference['trials']
        state['reference_comparison']=str(args.reference.resolve())
        config=dict(baseline, **{'agent.algorithm.clip_param':args.followup_clip})
        planned=[('clip'+str(args.followup_clip).replace('.',''),config)]
    write(out/'status.json',state)
    try:
        for name,config in planned:
            spec=out/f'{name}.json';write(spec,config)
            run_name=f'{out.name}_{name}'
            logs=ROOT/'logs/co_rl/Flamingo_Light_Flat_Stand_Drive/ppo'
            if list(logs.glob('*_'+run_name)):
                raise FileExistsError(f'Run name already exists: {run_name}')
            trial={'name':name,'status':'training','config':config}
            state['trials'].append(trial);write(out/'status.json',state)
            train=[sys.executable,'-u','scripts/co_rl/train.py','--task',TASK,'--algo','ppo',
                   '--headless','--batch_exit','--num_envs','1024','--seed','42',
                   '--max_iterations','1000','--run_name',run_name,'--param_overrides',str(spec)]
            trial['train_command']=train;write(out/'status.json',state)
            print(f'Starting {name}: 1000 scratch updates',flush=True)
            with (out/f'{name}_train.log').open('w') as f:
                subprocess.run(train,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            matches=list(logs.glob('*_'+run_name))
            if len(matches)!=1:
                raise RuntimeError(f'Expected one completed run, found {matches}')
            log=matches[0]
            completion=json.loads((log/'batch_complete.json').read_text())
            if completion['iterations_requested']!=1000 or completion['last_iteration']!=999:
                raise RuntimeError(f'Unexpected training budget: {completion}')
            params=json.loads((log/'params.json').read_text())
            if params.get('warmstart_ckpt') or params.get('resume_ckpt'):
                raise RuntimeError('Expected scratch training')
            trial.update(status='evaluating',log_dir=str(log),checkpoint=str(log/completion['checkpoint']))
            write(out/'status.json',state)
            evaluate=[sys.executable,'-u','scripts/co_rl/staged_retrain/benchmark.py','--headless',
                      '--mode','flat','--checkpoint',trial['checkpoint'],'--output',str(out/f'{name}_eval')]
            trial['eval_command']=evaluate;write(out/'status.json',state)
            with (out/f'{name}_eval.log').open('w') as f:
                subprocess.run(evaluate,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            trial.update(status='completed',evaluation=json.loads((out/f'{name}_eval/summary.json').read_text()))
            write(out/'status.json',state)
            print(f'Completed {name}',flush=True)
        def score(trial):
            cases=trial['evaluation']['cases']
            failure=sum(c['failure_rate'] for c in cases)/len(cases)
            tracking=max((c['vx_mae_mps'] if c['vx_mae_mps'] is not None else 1e6)/.7
                         +(c['wz_mae_radps'] if c['wz_mae_radps'] is not None else 1e6)/.5 for c in cases)
            return failure,tracking
        # Reward definitions/weights must be identical across the controlled trials.
        params=[json.loads((Path(t['log_dir'])/'params.json').read_text()) for t in state['trials']]
        for key in ('reward_weights','reward_params'):
            if any(params[0][key]!=p[key] for p in params[1:]):
                raise RuntimeError(f'Reward mismatch: {key}')
        ranked=sorted(state['trials'],key=score)
        write(out/'ranking.json',{'selected':ranked[0]['checkpoint'],
              'criterion':'mean failure rate, then worst-case normalized vx+yaw MAE',
              'scope':f'{len(state["trials"])} clip trials, one training seed, three reset evaluation seeds; not a global optimum.',
              'scores':[{ 'name':t['name'],'score':score(t)} for t in ranked]})
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=str(error))
        raise
    finally:
        write(out/'status.json',state)


if __name__=='__main__':
    main()
