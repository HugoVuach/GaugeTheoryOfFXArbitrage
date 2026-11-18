"""Check chronological separation and selection-aware risk validation."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from fxresearch.data import ingest_csv
from fxresearch.protocol import split_plan, DAY_NS
from fxresearch.certification import validate_policy_blocks


class ProtocolTests(unittest.TestCase):
    def test_observed_dates_separate_phases_and_purge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'quotes.csv'
            source.write_text('time,bid,ask\n' + ''.join(f'2024-08-{day:02d}T12:00:00Z,1,1.01\n' for day in [1,2,5,6,7,8,9,12]))
            database = root / 'quotes.sqlite'
            ingest_csv(source, database, {'base':'USD','quote':'EUR','venue':'A',
                       'columns':{'market_ts':'time','bid':'bid','ask':'ask'}})
            plan = split_plan(database, root / 'plan.json', 2, 2, 2, 1, 5000)
            self.assertEqual(len(plan['windows']), 3)
            self.assertTrue(plan['test_windows_overlap'])
            first = plan['windows'][0]
            self.assertEqual(first['calibration_dates'], ['2024-08-05','2024-08-06'])
            self.assertEqual(first['last_decision_each_day_offset_ns'], DAY_NS - 5000000000)
            for window in plan['windows']:
                train, cal, test = (set(window[key]) for key in ['training_dates','calibration_dates','test_dates'])
                self.assertFalse(train & cal or train & test or cal & test)
            short = split_plan(database, root / 'short.json', 60, 99, 30)
            self.assertEqual(short['status'], 'insufficient_days_for_requested_plan')

    def test_frozen_family_validation_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / 'registry.json'
            registry.write_text(json.dumps({'policies':[{'policy_id':'safe'},{'policy_id':'bad'}]}))
            source = root / 'blocks.csv'
            with source.open('w',newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['block_id','policy_id','accepted','loss','pnl'])
                for day in range(120):
                    writer.writerow([day,'safe',1,0,1])
                    writer.writerow([day,'bad',1,1,100])
            result = validate_policy_blocks(source, registry, root / 'risk.json')
            self.assertEqual(result['selected_policy'],'safe')
            bounds = {row['policy_id']:row for row in result['policies']}
            self.assertLess(bounds['safe']['risk_upper'], .05)
            self.assertEqual(bounds['bad']['risk_upper'], 1)
            with source.open('a') as stream:
                stream.write('0,safe,1,0,1\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                validate_policy_blocks(source, registry, root / 'invalid.json')


if __name__ == '__main__':
    unittest.main()
