import json
import numpy as np
import pytest
from decisionicl.cli import main


def test_offline_fit_retrieve_and_test_fit_rejection(tmp_path):
    train = np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]],np.float32)
    common = dict(train_ids=np.array(['t0','t1','t2','t3']),train_labels=np.array(['a','a','b','b']),
                  factor_count=np.array(1),train_probabilities_0=train)
    validation = tmp_path/'validation.npz'
    np.savez(validation,**common,query_split=np.array('validation'),query_ids=np.array([f'v{i}' for i in range(30)]),
        query_labels=np.array(['a','b']*15), parent_scores=np.tile([[.9,.8,.1,.2],[.1,.2,.9,.8]],(15,1)),
        query_probabilities_0=np.tile([[.9,.1],[.1,.9]],(15,1)))
    test = tmp_path/'test.npz'
    np.savez(test,**common,query_split=np.array('test'),query_ids=np.array(['q0']),
        parent_scores=np.array([[.1,.2,.8,.9]]),query_probabilities_0=np.array([[.1,.9]]))
    contract = tmp_path/'contract.json'
    contract.write_text(json.dumps(dict(target_model='fixed',dataset='test',bank_sha256='bank',evidence_encoder='fixed',parent='rices',seed=42)))
    calibration = tmp_path/'calibration.json'
    shared = ['--contract',str(contract),'--shots','1','--calibration',str(calibration)]
    main(['fit','--input',str(validation),*shared])
    plan = tmp_path/'plan.jsonl'
    main(['retrieve','--input',str(test),*shared,'--output',str(plan)])
    assert json.loads(plan.read_text())['demo_ids']==['t3']
    with pytest.raises(SystemExit):
        main(['fit','--input',str(test),*shared])

