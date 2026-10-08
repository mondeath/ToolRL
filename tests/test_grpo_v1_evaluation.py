"""Independent evaluator contracts without GPU or changes to frozen semantics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from src.evaluation.grpo_v1.data import prepare_selection, load_selection, leakage_audit, subset
from src.evaluation.grpo_v1.audit import audit_run
from src.evaluation.grpo_v1.archive import archived_pair, replay
from src.evaluation.grpo_v1.reporting import metrics, comparison
from src.training.grpo_v1.data import read_json
from src.env_v1 import Environment
from src.env_v1.tasks import initial_state
from src.tasks import public_task

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/grpo_v1_baseline/run_20261007_160206'

@pytest.fixture(scope='module')
def fixture():
    selection=prepare_selection(ROOT,RUN)
    train,items=load_selection(ROOT,RUN,selection)
    config=read_json(RUN/'config.json')
    return selection,train,items,config


def test_selection_and_explicit_leakage(fixture):
    selection,train,items,_=fixture
    assert selection==read_json(ROOT/'configs/grpo_v1_eval_tasks.json')
    assert len(items)==60 and len(subset(items,'pilot'))==3 and len(subset(items,'robustness'))==12
    audit=leakage_audit(train,items)
    assert all(audit['counts'][key]==0 for key in ('seed_overlap','task_id_overlap','exact_instance_overlap'))
    assert audit['counts']['template_overlap']==60
    assert audit['counts']['query_and_answer_overlap']>0
    bad=deepcopy(selection); bad['tasks'].reverse()
    with pytest.raises(ValueError): load_selection(ROOT,RUN,bad)


def test_checkpoint_missing_and_model_mismatch():
    audit=audit_run(RUN,expected_model='Qwen/Qwen2.5-1.5B-Instruct')
    assert not audit['restoration_ready'] and not audit['independently_restored']
    assert any('Expected' in reason for reason in audit['blockers'])
    assert not audit['cost_penalty_in_training'] and not audit['verifier_reward']
    assert audit['kl_beta']==0


def test_archive_replay_pilot_and_metrics(fixture):
    _,_,items,config=fixture
    pilot=subset(items,'pilot')
    before,after=archived_pair(pilot,RUN,config)
    result=comparison(before,after)
    assert result['base']['generated_tokens'] is None
    assert result['trained']['error_recovery_rate'] is None
    assert len(result['paired_tasks'])==3
    damaged=deepcopy(before[0]); damaged['trajectory']['tool_calls'][0]['observation']={'tampered':True}
    with pytest.raises(ValueError,match='observation'): replay(pilot[0],damaged,config)
    with pytest.raises(ValueError,match='Unpaired'): comparison(before,after[:-1])
    with pytest.raises(ValueError,match='Duplicate'): comparison(before+before[:1],after)
    failed=deepcopy(before)
    for row in failed: row['success']=row['task_success']=False
    assert metrics(failed)['cost_per_successful_task'] is None
    for row in failed: row['generated_tokens']=5
    assert metrics(failed)['generated_tokens']==15


def test_same_failure_schedule_and_nested_threshold(fixture):
    _,_,items,config=fixture
    item=items[0]
    def schedule(rate):
        env=Environment(initial_state(item['kb'],item['task']),failure_rate=rate,failure_type='timeout',seed=2026)
        env.reset(public_task(item['task']))
        for _ in range(8): env.step('lookup',entity='s00')
        return [bool(c['failure_type']) for c in env.trajectory.tool_calls]
    assert schedule(.2)==schedule(.2)
    assert all(not low or high for low,high in zip(schedule(.2),schedule(.5)))


def test_exact_capture_including_repair_and_eos(fixture,monkeypatch):
    import torch
    from src.policies import PolicyOutput
    from src.policies.huggingface import HuggingFacePolicy
    from src.evaluation.grpo_v1.inference import EvaluationPolicy,collect
    _,_,items,config=fixture
    item=next(x for x in items if x['task']['evaluation']['kind']=='lookup')
    outputs=[PolicyOutput('not json',[1,2,3]),
        PolicyOutput(json.dumps({'action':'direct_answer','arguments':{'final_answer':0}}),[4,5])]
    def act(self,state):
        result=outputs[self.call_index]; self.call_index+=1; return result
    monkeypatch.setattr(HuggingFacePolicy,'act',act)
    model=torch.nn.Linear(1,1)
    model.generation_config=SimpleNamespace(top_k=0)
    policy=EvaluationPolicy(model,None,config)
    policy.config['do_sample']=policy.sampling_config['do_sample']=False
    row=collect([item],policy,config)[0]
    assert row['generated_tokens']==5
    assert [s['token_ids'] for s in row['generation_segments']]==[[1,2,3],[4,5]]
    assert len(row['trajectory']['raw_outputs'])==2
    assert len(row['generation_segments'][1]['messages'])>0
