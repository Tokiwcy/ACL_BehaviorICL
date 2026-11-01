import json
from pathlib import Path
import numpy as np
import pytest
from scipy.special import rel_entr
from decisionicl.baselines import (
    circles_features, circles_demonstrations, mmices_question_only,
    muier_scores, topk, rotated_factor_features,
)
from decisionicl.pipeline import DecisionICLRetriever
from decisionicl.schema import Sample
from decisionicl.models import exact_label, gemma_final_text, INVALID
from decisionicl.selection import factor_class_similarities, class_features
from decisionicl.analysis import repeated_summary

ROOT = Path(__file__).resolve().parents[1]


def test_real_paper_rescue_and_failure_distribution_scores():
    cases = json.loads((ROOT/'results/paper/algorithm_cases.json').read_text())['cases']
    assert {case['kind'] for case in cases} == {'rescue', 'harm'}
    for case in cases:
        q = [np.asarray([f['query_evidence']]) for f in case['factors']]
        p = [np.asarray([f['base_prototype'], f['decision_prototype']]) for f in case['factors']]
        scores = factor_class_similarities(q, p)[0]
        contributions = []
        for i, factor in enumerate(case['factors']):
            np.testing.assert_allclose(scores[:, i], [factor['base_agreement'], factor['decision_agreement']], atol=2e-7)
            middle = (q[i][:, None] + p[i][None])/2
            js = (rel_entr(q[i][:, None], middle).sum(-1)+rel_entr(p[i][None], middle).sum(-1))/2
            np.testing.assert_allclose(scores[:, i], 1-js[0]/np.log(2), atol=2e-7)
            contributions.append(factor['weighted_margin_contribution'])
        np.testing.assert_allclose(sum(contributions)+case['weighted_base_margin'], case['fused_margin'], atol=2e-6)


def test_circles_uses_independent_channel_maxima_and_last_tie():
    image = np.array([[.9, .4, .8, .2]], np.float32)
    caption = np.array([[.1, .7, .3, .9]], np.float32)
    factors = np.array([[[.1], [.9]]], np.float32)
    features, first, second = circles_features(image, caption, [0,0,1,1], factors)
    raw = np.array([[[.8,.1],[.85,.9]]], np.float32)
    expected = (raw-raw.mean(1,keepdims=True))/raw.std(1,keepdims=True)
    np.testing.assert_allclose(features, expected, atol=2e-6)
    np.testing.assert_array_equal(first, [[0,2]])
    np.testing.assert_array_equal(second, [[1,3]])
    np.testing.assert_array_equal(circles_demonstrations(np.ones((1,4)),np.ones((1,4))), [[3,3]])


def test_muier_and_mmices_follow_published_adaptations():
    q = np.array([[1.,0.]])
    t = np.array([[1.,0.],[0.,1.],[-1.,0.]])
    labels = np.array([[0.,1.],[1.,0.]])
    np.testing.assert_allclose(muier_scores(q,t,[0,1,1],labels), [[1,1,0]])
    image = np.array([[.2,.9,.5,.1]])
    np.testing.assert_array_equal(mmices_question_only(image,np.ones_like(image),2,4), topk(image,2))
    np.testing.assert_array_equal(rotated_factor_features(image,[0,0,1,1],np.array([[[.1],[.8]]])),
                                  class_features(image,[0,0,1,1],range(2),np.array([[[.8],[.1]]])))


def make_retriever(shots=1):
    return DecisionICLRetriever(['t0','t1','t2','t3'],['a','a','b','b'],
        [np.array([[.95,.05],[.85,.15],[.1,.9],[.2,.8]],np.float32)],
        contract=dict(target_model='test', dataset='test',bank_sha256='fixed', evidence_encoder='fixed', parent='rices',seed=42), shots=shots)


def fit(retriever):
    samples = [Sample(f'v{i}', '', 'a' if i%2==0 else 'b','validation') for i in range(30)]
    scores = np.array([[.9,.8,.3,.2] if s.label=='a' else [.1,.2,.8,.9] for s in samples],np.float32)
    probabilities = [np.array([[.9,.1] if s.label=='a' else [.1,.9] for s in samples],np.float32)]
    return retriever.fit(scores,probabilities,samples)


def test_validation_contract_roundtrip_and_no_test_label_access():
    retriever = make_retriever()
    calibration = json.loads(json.dumps(fit(retriever)))
    fresh = make_retriever()
    fresh.load_calibration(calibration)
    plan = fresh.retrieve(np.array([[.1,.2,.8,.9]]),[np.array([[.1,.9]])],['q0'])
    assert plan[0]['demo_ids']==['t3'] and plan[0]['test_label_used'] is False
    with pytest.raises(ValueError, match='different'):
        make_retriever(shots=2).load_calibration(calibration)
    with pytest.raises(ValueError, match='overlap'):
        fresh.retrieve(np.array([[.1,.2,.8,.9]]),[np.array([[.1,.9]])],['v0'])


def test_strict_parsing_and_sample_sd():
    assert exact_label(' A319 ', ['a319','a320'])=='a319'
    assert exact_label('The answer is a319.', ['a319','a320'])==INVALID
    assert gemma_final_text({'role':'assistant'})==''
    with pytest.raises(ValueError):
        gemma_final_text({'role':'assistant','content':None})
    values = repeated_summary([[1.,2.],[3.,4.],[5.,6.]])
    assert values['cell_sample_sd']==[2.,2.]
    assert values['equal_cell_sample_sd']==2.

