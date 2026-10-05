"""Independent fresh-source wrapper; all old native and cache guards remain."""
import sys
import behavior_qwenvision_two_host_policy_v1 as p
import run_behavior_qwenvision_two_host_v1 as host
import behavior_qwenvision_two_fresh_capture_v1 as fresh

BINDING=p.ROOT/'analysis/qwenvision_two_fresh_v1/implementation_lock.json'


def validate():
    binding=p.read(BINDING)
    if binding['source_contract_hash']!=p.canonical({k:v for k,v in binding.items() if k!='source_contract_hash'}):
        raise ValueError('Fresh-source execution identity differs')
    for name,sha in binding['code_sha256'].items():
        if p.digest(p.ROOT/name)!=sha:raise ValueError('Exact independent fresh-source code changed')
    plan=p.validate_plan(p.read(p.PLAN_PATH));original=p.validate_binding(plan)
    if (binding['original_plan_hash']!=plan['plan_hash'] or binding['original_host_implementation_hash']!=original['implementation_hash']
        or binding['original_manifest_hash']!=p.canonical({d:v['manifest'] for d,v in plan['datasets'].items()})
        or binding['images']!=17428 or binding['predictions']!=7674 or binding['fixed_width']!=256
        or binding['train_steps']!=1500 or binding['seed']!=73 or binding['numerical_tolerance_added'] is not False):
        raise ValueError('Original complete fixed scientific scope differs')
    return plan,binding


def main():
    plan,binding=validate()
    phase=sys.argv[sys.argv.index('--phase')+1] if '--phase' in sys.argv else 'preflight'
    if phase in ('capture','train','test'):
        p.gpu_guard(plan)
        auth=p.read(p.ROOT/'results'/p.OPERATION/'execution_authorization.json')
        if auth.get('source_contract_hash')!=binding['source_contract_hash']:raise ValueError('Native launch lacks this independent source authorization')
        science=p.ROOT/'results'/p.SCIENCE;science.mkdir(parents=True,exist_ok=True)
        receipt={'plan_hash':plan['plan_hash'],'source_contract_hash':binding['source_contract_hash'],
            'pod_id':auth['pod_id'],'launch_id':auth['launch_id'],'all_two_sources_newly_captured':True,
            'original_cache_used':False,'original_formula_preserved':True}
        p.immutable(science/'source_execution_contract.json',receipt)
    host.scientific.capture=fresh.capture
    # The host still verifies old model/image/cache provenance and every native,
    # runtime, successful backup and single GPU guard before each scientific call.
    host.main()


if __name__=='__main__':main()
