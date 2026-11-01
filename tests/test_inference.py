from types import SimpleNamespace
import numpy as np
import pytest
from decisionicl.inference import run_image_plan
from decisionicl.schema import Sample
from decisionicl.source_controls import matched_source_prompt
from decisionicl.discovery import balanced_discovery_subset
from decisionicl.prompts import vicl_one_shot, vicl_two_shot
from decisionicl.spatial import spatial_scores


def test_fixed_group_checkpoint_resume_and_partner_drift(tmp_path):
    image=tmp_path/'image.bin'
    image.write_bytes(b'fixed-image-identity')
    train=[Sample('t',str(image),'a','train')]
    query=[Sample('q0',str(image),'a','test'),Sample('q1',str(image),'b','test')]
    plan=[{'query_id':q.sample_id,'demo_ids':['t']} for q in query]
    class Backend:
        config=SimpleNamespace(model='fixed',revision='sha')
        calls=0
        def classify(self,conversations,labels):
            self.calls+=1
            return [{'prediction':'a','raw_output':'a'} for _ in conversations]
    backend=Backend()
    kwargs=dict(backend=backend,labels=['a','b'],question='Classify.',output_path=tmp_path/'answers.jsonl')
    first=run_image_plan(plan,train,query,**kwargs)
    assert len(first)==1 and backend.calls==1
    assert run_image_plan(plan,train,query,**kwargs)==first and backend.calls==1
    with pytest.raises(ValueError,match='different fixed-group'):
        run_image_plan(list(reversed(plan)),train,query,**kwargs)
    image.write_bytes(b'changed-image')
    with pytest.raises(ValueError,match='different fixed-group'):
        run_image_plan(plan,train,query,**kwargs)


def test_source_intervention_and_train_only_discovery():
    args=('Classify aircraft.',['a','b'],'shape')
    a=matched_source_prompt(args[0],'matched_decision',*args[1:])
    b=matched_source_prompt(args[0],'generic_visual',*args[1:])
    assert a.split('\n\n',1)[1]==b.split('\n\n',1)[1]
    with pytest.raises(ValueError,match='training samples'):
        balanced_discovery_subset([Sample('q','','a','test')],2,42)


def test_vicl_text_demonstrations_and_corresponding_patch_score():
    one=vicl_one_shot('query.png','summary','a','Classify.',['a','b'])
    two=vicl_two_shot('query.png',['summary1','summary2'],['a','b'],'Classify.',['a','b'])
    assert sum(x['type']=='image' for x in one[0]['content'])==1
    assert sum(x['type']=='image' for x in two[0]['content'])==1
    q=np.zeros((1,256,1024),np.float32);q[:,:,0]=1
    t=np.zeros((2,256,1024),np.float32);t[0,:,0]=1;t[1,:,1]=1
    np.testing.assert_allclose(spatial_scores(q,t),[[1,0]])

