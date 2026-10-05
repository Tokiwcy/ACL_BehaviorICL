"""CPU synthetic TAM API/target alignment tests; no pretrained model or CUDA."""
from pathlib import Path
import json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from scipy.optimize import minimize_scalar
import query_tam_adapter_v1 as adapter
class Processor:
    def __init__(self):self.tokenizer=self
    def tokenize(self,text):return ['alpha','beta']
    def encode(self,text,add_special_tokens=False):return [10,11]
    def batch_decode(self,rows,**kwargs):return ['alpha beta' if list(row)==[10,11] else str(row) for row in rows]
class TAMTests(unittest.TestCase):
    def test_actual_official_ECI_filter_and_first_generated_target_index(self):
        torch.manual_seed(73)
        namespace,receipt=adapter.load(ROOT/'.tools/tam_official_v1/tam.py',torch,np,minimize_scalar)
        tokens=[9,100,7,7,7,7,101,10,11,99,98,12,13]
        logits=[torch.rand(1,11,120,device='cpu'),torch.rand(1,1,120,device='cpu')]
        special={'img_id':[7],'prompt_id':[101,99],'answer_id':[98,-1]}
        rows=adapter.maps(namespace,tokens,(2,2),logits,special,np.zeros((5,5,3),dtype='uint8'),Processor(),2)
        self.assertEqual([r['token_id'] for r in rows],[12,13])
        np.testing.assert_array_equal(rows[0]['raw_CAM'],logits[0][0,2:6,12].numpy().reshape(2,2))
        np.testing.assert_array_equal(rows[1]['raw_CAM'],logits[0][0,2:6,13].numpy().reshape(2,2))
        self.assertTrue(all(np.isfinite(r['post_ECI_before_filter']).all() for r in rows))
        self.assertTrue(all(r['map'].shape==(2,2) for r in rows))
        self.assertFalse(receipt['author_source_bytes_changed'])
        # Changing only text token logits leaves fixed-target raw vision map
        # identical; the ECI context correction can still change the TAM.
        altered=[x.clone() for x in logits];altered[0][:,7:,:]*=3
        new,_=adapter.load(ROOT/'.tools/tam_official_v1/tam.py',torch,np,minimize_scalar)
        other=adapter.maps(new,tokens,(2,2),altered,special,np.zeros((5,5,3),dtype='uint8'),Processor(),2)
        np.testing.assert_array_equal(rows[0]['raw_CAM'],other[0]['raw_CAM'])
    def test_wrong_source_or_token_boundary_refused(self):
        namespace,_=adapter.load(ROOT/'.tools/tam_official_v1/tam.py',torch,np,minimize_scalar)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'changed.py';file.write_bytes(b'changed code')
            with self.assertRaises(ValueError):adapter.load(file,torch,np,minimize_scalar)
        with self.assertRaises(ValueError):
            adapter.maps(namespace,[1,2,3],(2,2),[torch.ones(1,3,20)],
                {'img_id':[7],'prompt_id':[101,99],'answer_id':[98,-1]},np.zeros((5,5,3)),Processor(),1)
    def test_frozen_selection_and_original_prompts_exact(self):
        plan=json.loads((ROOT/'analysis/query_tam_pilot_v1/protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(plan['model'],'Qwen/Qwen3-VL-4B-Instruct')
        self.assertEqual(len(plan['examples']),8)
        for d in ('cub','dtd'):
            self.assertEqual({v['historical_group'] for v in plan['examples'] if v['dataset']==d},{'task_only','generic_only','both_correct','both_wrong'})
        self.assertTrue(all(v['queries']['generic']=='Describe the image briefly.' for v in plan['examples']))
if __name__=='__main__':unittest.main()
