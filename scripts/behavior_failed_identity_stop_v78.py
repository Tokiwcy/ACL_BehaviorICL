"""Retain exact pre-parent launch identity when closing a failed controller."""


def identity(pod, auth, deployment, accounting):
    expected = {'pod_id': deployment.get('pod_id'), 'launch_id': deployment.get('launch_id')}
    if (not pod or pod != expected['pod_id'] or not expected['launch_id']
            or any(accounting.get(k) != value for k, value in expected.items())
            or any(auth.get(k) not in (None, '', value) for k, value in expected.items())):
        raise ValueError('Exact original deployment and independent fee segment identity required')
    return {**auth, **expected}


def stop(backup, pod, auth, deployment, read, confirmed_stop, reason, capacity=False):
    bound = identity(pod, auth, deployment, read(backup / 'runtime_accounting.json'))
    result = confirmed_stop(backup, bound, reason, capacity)
    if result.get('pod_id') != bound['pod_id'] or result.get('disks_preserved') is not True:
        raise ValueError('Exact confirmed stop with preserved disk required')
    return bound, result
