import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_behavior_qwenvision_match_control_v1 as m


class MatchControlTests(unittest.TestCase):
    def fixture(self):
        manifest={'bank':[{'sample_id':str(i),'label':'a','split':'bank'} for i in range(4)],'query':[{'sample_id':'q','label':'a','split':'query'}]}
        zero={r['sample_id']:{'true_label':r['label'],'split':r['split'],'zero_shot_pred':'b','zero_shot_valid':True,'zero_shot_correct':False} for r in manifest['bank']+manifest['query']}
        pred={'query_id':'q','target':'a','prediction':'a','correct':True,'demo_ids':['0','1','2','3'],'demo_labels':['a']*4}
        return manifest,zero,pred

    def test_all_errors_and_unknown_remain_in_denominator(self):
        manifest,zero,pred=self.fixture()
        rows=m.derived_rows('fixture',manifest,zero,[pred],'method')
        self.assertEqual(m.method_summary(rows)['strict_present_queries'],1)
        zero['q'].update(zero_shot_pred=None,zero_shot_valid=False)
        rows=m.derived_rows('fixture',manifest,zero,[pred],'method'); summary=m.method_summary(rows)
        self.assertEqual((summary['zero_errors'],summary['rescued'],summary['strict_unknown_queries']),(1,1,1))
        zero['q'].update(zero_shot_pred='b',zero_shot_valid=True)
        zero['0'].update(zero_shot_pred=None,zero_shot_valid=False)
        self.assertEqual(m.derived_rows('fixture',manifest,zero,[pred],'method')[0]['strict_status'],'present')

    def test_official_order_duplicate_bank_label_and_output_guard(self):
        manifest,zero,pred=self.fixture()
        for changed in ({**pred,'demo_ids':['q','1','2','3']},{**pred,'demo_ids':['0','0','2','3']},{**pred,'demo_labels':['b']*4},{**pred,'correct':False},{**pred,'query_id':'other'}):
            with self.assertRaises(ValueError):m.derived_rows('fixture',manifest,zero,[changed],'method')
        with self.assertRaises(ValueError):m.derived_rows('fixture',manifest,zero,[],'method')


if __name__=='__main__':unittest.main()
