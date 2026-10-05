"""New host wrapper; frozen one-control science and all original guards intact."""
import sys
import behavior_qwenvision_two_host_policy_v1 as p
import run_behavior_qwenvision_two_control_v1 as scientific


def main():
    scientific.p = p
    scientific.shared.p = p
    plan = p.validate_plan(p.read(p.PLAN_PATH))
    p.validate_binding(plan)
    phase = sys.argv[sys.argv.index('--phase')+1] if '--phase' in sys.argv else 'preflight'
    if phase in ('capture', 'train', 'test'):
        p.gpu_guard(plan)
        # Exact narrowly repaired operational root-before-lock ordering. Does
        # not change source representations, training, prompts or predictions.
        (p.ROOT / 'results' / p.SCIENCE).mkdir(parents=True, exist_ok=True)
        p.validate_cache(plan, local=False)
    elif phase == 'verify':
        if not (p.ROOT / 'results' / p.SCIENCE).is_dir():
            raise ValueError('No complete scientific output to verify')
    scientific.main()


if __name__ == '__main__': main()
