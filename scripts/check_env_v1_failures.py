"""Deterministic fault/retry integration probes, explicitly NOT model baseline results."""
from _common import ROOT
import json
from src.env_v1 import Environment, generate_kb
from src.env_v1.tasks import generate_tasks
from src.env_v1.evaluation import evaluate, summarize
from src.tasks import public_task


def run():
    kb=generate_kb()
    tasks=generate_tasks(kb)[:5]
    summaries=[]
    for rate in (0.0,0.2,0.5):
        rows=[]
        for seed in range(10):
            for task in tasks:
                env=Environment(kb,failure_rate=rate,seed=seed,max_tool_calls=8)
                env.reset(public_task(task),metadata={'scripted_fault_probe':True})
                answer=None
                for _ in range(8):
                    observation=env.step('lookup',entity=task['evaluation']['entity'])
                    if 'gpa' in observation.get('record',{}):
                        answer=observation['record']['gpa']
                        break
                env.step('direct_answer',final_answer=answer)
                rows.append(evaluate(task,env))
        summaries.append(dict(failure_rate=rate,**summarize(rows),
            failure_counts={kind:sum(e['type']==kind for r in rows for e in r['failure_events']) for kind in ('timeout','partial_result')}))
    result=dict(kind='scripted_fault_retry_probe_not_model_baseline',episodes_per_rate=50,results=summaries)
    (ROOT.parent/'results/env_v1_failure_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': run()
