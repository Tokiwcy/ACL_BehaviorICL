"""Offline regression for stale pending cells in a composite results audit."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import update_internvl35_qwen_results_v4 as updater


class CompositeTests(unittest.TestCase):
    def test_pending_list_uses_merged_states(self):
        audit = {'incomplete': ['cub/raw: missing_selection 0/5794'], 'datasets': {
            d: {'methods': {m: {'state': 'complete', 'completed': 5794, 'total': 5794}
                           for m in updater.METHODS}} for d in updater.DATASETS}}
        audit['datasets']['cub']['methods']['detriever'].update(state='partial', completed=2204)
        updater.refresh_incomplete(audit)
        self.assertEqual(audit['incomplete'], ['cub/detriever: partial 2204/5794'])
        self.assertEqual(audit['state'], 'incomplete')
        audit['datasets']['cub']['methods']['detriever'].update(state='complete', completed=5794)
        updater.refresh_incomplete(audit)
        self.assertEqual(audit['incomplete'], [])
        self.assertEqual(audit['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
