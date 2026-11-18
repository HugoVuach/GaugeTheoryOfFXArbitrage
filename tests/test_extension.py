"""Meaningful regression/property checks for the research additions."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fxresearch.graph import (enumerate_cycles, cycle_mean, maximum_cycle_mean,
    positive_cycle, best_witness, potential_diagnostic, triangle_maximum)
from fxresearch.certification import (conformal_quantile, minimum_blocks,
    structured_radius, box_radius, ReservedLeg, reserved_path_bound, adaptive_coverage, conditional_risk_upper)
from fxresearch.flow import CashEdge, funded_flow
from fxresearch.data import ingest_csv, iter_quotes, timestamp_ns, dataset_summary
from fxresearch.pipeline import run_diagnostics, audit_fills
from fxresearch.execution import MarketTape, ExecutionConfig, execute_route
from fxresearch.replay import replay_alerts


class GraphTests(unittest.TestCase):
    def test_sparse_random_against_enumeration_and_gauge(self):
        rng = np.random.default_rng(13061)
        for n in range(2, 8):
            cycles = list(enumerate_cycles(n))
            for _ in range(12):
                a = rng.normal(size=(n, n)); a[rng.random((n, n)) < .35] = -np.inf
                np.fill_diagonal(a, -np.inf)
                truth = max(cycle_mean(a, c) for c in cycles)
                found = maximum_cycle_mean(a)
                self.assertTrue((not np.isfinite(truth) and not np.isfinite(found)) or abs(truth - found) < 1e-10)
                witness = positive_cycle(a)
                self.assertEqual(bool(witness), truth > 1e-8)
                phi = rng.normal(size=n)
                shifted = a + phi[None, :] - phi[:, None]
                self.assertAlmostEqual(maximum_cycle_mean(shifted), found) if np.isfinite(found) else None
                if witness:
                    best, route = best_witness(a)
                    self.assertAlmostEqual(best, cycle_mean(a, route), places=7)

    def test_quad_and_sparse_hodge(self):
        a = np.full((4, 4), -3.); np.fill_diagonal(a, -np.inf)
        for i in range(4):a[i, (i+1)%4] = 1
        self.assertEqual(triangle_maximum(a), -1/3)
        self.assertAlmostEqual(maximum_cycle_mean(a), 1)
        h = np.array([0, 4, -3, 8.])
        mid = h[None, :] - h[:, None]; np.fill_diagonal(mid, -np.inf)
        mid[0, 2] = mid[2, 0] = mid[1, 3] = mid[3, 1] = -np.inf
        summary = potential_diagnostic(mid)
        self.assertEqual(summary.cycle_rank, 1)
        self.assertLess(summary.rms, 1e-12)


class CertificateTests(unittest.TestCase):
    def test_exact_rank_guarantee_and_small_samples(self):
        values = np.arange(10.)
        failures = 0
        for i, target in enumerate(values):
            failures += target > conformal_quantile(np.delete(values, i), .1)
        self.assertEqual(failures, 1)
        self.assertTrue(math.isinf(conformal_quantile([0,1,2,3,4], .01)))
        self.assertEqual(minimum_blocks(.01),99)

    def test_conditional_risk_binomial_bounds(self):
        self.assertEqual(conditional_risk_upper(0,0),1)
        self.assertEqual(conditional_risk_upper(4,4),1)
        expected=1-(.05/8)**(1/100)
        self.assertAlmostEqual(conditional_risk_upper(0,100,family_size=8),expected)
        self.assertGreater(conditional_risk_upper(1,100,family_size=8),expected)

    def test_support_function_attainment(self):
        rng = np.random.default_rng(13062)
        for _ in range(80):
            c = rng.normal(size=8); u=rng.normal(size=(8,3)); r=rng.uniform(0,.3,8)
            radius=structured_radius(c,u,r)
            z = -(u.T@c)/np.linalg.norm(u.T@c)
            error=u@z-r*np.sign(c)
            self.assertAlmostEqual(-c@error,radius)
            self.assertLessEqual(radius,box_radius(c,u,r)+1e-10)

    def test_capacity_requires_upper_not_lower_amount(self):
        legs=[ReservedLeg(1,2,100),ReservedLeg(1.1,1.2,15)]
        # Lower after first leg is 10, but upper is 20 > second capacity 15.
        self.assertFalse(reserved_path_bound(10,legs)['capacity_valid'])
        good=[ReservedLeg(1.1,1.2,100),ReservedLeg(1.1,1.2,100)]
        self.assertAlmostEqual(reserved_path_bound(10,good)['lower_profit'],2.1)

    def test_flow_dual_and_unit_transformation(self):
        edges=[CashEdge(0,1,1.1,100),CashEdge(1,2,1.05,110),CashEdge(0,2,1,100)]
        a=funded_flow(3,edges,0,2,100)
        self.assertLess(abs(a['duality_gap']),1e-8)
        self.assertAlmostEqual(a['terminal_cash'],115.5)
        g=np.array([3.,.2,7.])
        shifted=[CashEdge(e.source,e.destination,e.rate*g[e.destination]/g[e.source],e.capacity*g[e.source]) for e in edges]
        b=funded_flow(3,shifted,0,2,100*g[0],repayment=g[2]/g[0])
        self.assertAlmostEqual(b['net_profit'],g[2]*a['net_profit'])

    def test_aci_count_identity_arbitrary_drift(self):
        rng=np.random.default_rng(13063)
        scores=np.r_[rng.exponential(1,300),rng.exponential(4,300)]
        rows=adaptive_coverage(scores,rng.exponential(1,199),alpha=.1,gamma=.01)
        self.assertAlmostEqual(rows[:,1].mean()-.1,(.1-rows[-1,3])/(.01*len(rows)))
        self.assertGreaterEqual(rows[:,3].min(),-.009-1e-12)
        self.assertLessEqual(rows[:,3].max(),1.001+1e-12)


class DataTests(unittest.TestCase):
    def test_exact_ns_timezone_and_missing_time(self):
        self.assertEqual(timestamp_ns('2024-07-01T00:00:00.123456789Z')%1000000000,123456789)
        with self.assertRaises(ValueError):timestamp_ns('1700000000000')
        self.assertEqual(timestamp_ns('1700000000000',unit='ms'),1700000000000000000)
        with self.assertRaises(ValueError):timestamp_ns('2024-07-01T00:00:00')
        with self.assertRaises(ValueError):timestamp_ns('2024-10-27T02:30:00','Europe/Paris')
        self.assertEqual(timestamp_ns('2024-07-01T00:00:00','EST_FIXED'),timestamp_ns('2024-07-01T05:00:00Z'))

    def test_causal_full_day_multi_currency_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); src=p/'quotes.csv'; db=p/'quotes.sqlite'
            fields=['market','receive','venue','base','quote','bid','ask']
            with src.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(fields)
                # Six currencies, two venues, quote sent before midnight but
                # received after it. Detection must not read it at midnight.
                for venue in ['A','B']:
                    for c in ['EUR','GBP','JPY','CHF','AUD']:
                        w.writerow(['2024-08-01T00:00:00Z','2024-08-01T00:00:01Z',venue,'USD',c,1,1.001])
            cfg={'columns':dict(zip(['market_ts','receive_ts','venue','base','quote','bid','ask'],fields))}
            self.assertEqual(ingest_csv(src,db,cfg)['rows'],10)
            self.assertTrue(ingest_csv(src,db,cfg)['reused'])
            summary=dataset_summary(db);self.assertEqual(summary['rows_without_sizes'],10)
            self.assertEqual(len(list(iter_quotes(db,end=timestamp_ns('2024-08-01T00:00:00.5Z')))),0)
            run=run_diagnostics(db,{'start_utc':'2024-08-01T00:00:00Z','end_utc':'2024-08-02T00:00:00Z','step_ms':60000,'max_age_ms':1000},p/'out')
            self.assertEqual(run['grid_points'],1440)
            self.assertEqual(run['currency_count'],6)
            self.assertEqual(len(run['nodes']),12)
            self.assertEqual(run['data']['receive_times_assumed'],0)

    def test_fill_audit_does_not_hide_residual_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); src=p/'fills.csv'
            src.write_text('episode_id,initial_amount,status,source_currency,destination_currency,source_amount,destination_amount,fee_amount,fee_currency\na,100,filled,USD,EUR,100,90,0,USD\n')
            result=audit_fills(src,p/'audit.json')
            self.assertFalse(result[0]['closed'])
            self.assertIsNone(result[0]['realized_pnl'])

    def test_historical_replay_missing_size_and_overlapping_orders(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source=p/'quotes.csv';db=p/'book.sqlite';alert=p/'alerts.csv'
            source.write_text('time,base,quote,bid,ask,size\n2024-08-01T00:00:00Z,USD,EUR,1.1,1.1001,1000000\n2024-08-01T00:00:00Z,EUR,GBP,0.8,0.8001,1000000\n2024-08-01T00:00:00Z,GBP,USD,1.2,1.2001,1000000\n')
            cfg={'venue':'A','columns':{'market_ts':'time','base':'base','quote':'quote','bid':'bid','ask':'ask','bid_size':'size','ask_size':'size'}}
            ingest_csv(source,db,cfg)
            t=timestamp_ns('2024-08-01T00:00:00.020Z')
            with alert.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['decision_id','timestamp_ns','cycle_json'])
                for k in range(2):writer.writerow([k,t+k*100000000,json.dumps([['A','USD'],['A','EUR'],['A','GBP']])])
            manifest=replay_alerts(db,alert,{'latency_ms_per_leg':100},p/'out')
            self.assertEqual(manifest['status_counts']['closed_full'],1)
            self.assertEqual(manifest['status_counts']['overlap_skipped'],1)
            self.assertGreater(manifest['terminal_cash_by_venue']['A'],1000000)
            audit=audit_fills(p/'out/hypothetical_order_log.csv',p/'audit.json')
            self.assertTrue(audit[0]['closed'])
            # Missing source depth must cause abstention, not infinite liquidity.
            cfg['columns'].pop('bid_size');cfg['columns'].pop('ask_size')
            ingest_csv(source,p/'missing.sqlite',cfg)
            manifest=replay_alerts(p/'missing.sqlite',alert,{},p/'missing_out')
            self.assertEqual(manifest['orders'],0)
            self.assertEqual(manifest['status_counts']['unassessable_initial_quote_or_size'],2)


class ExecutionTests(unittest.TestCase):
    def test_reserved_profit_and_rejected_leg_cleanup(self):
        values=np.zeros((2000,3))
        tape=MarketTape(values,pulse_start_ms=np.array([0]),pulse_end_ms=np.array([18000]),pulse_amplitudes_bp=np.array([2.]))
        obs=np.zeros((3,3),dtype=int)
        good=execute_route(tape,(0,1,2),100,obs,ExecutionConfig(mode='reserved'),1,True)
        self.assertGreater(good['pnl'],0)
        self.assertTrue(good['complete'])
        bad=execute_route(tape,(0,1,2),100,obs,ExecutionConfig(rejection_probability=1),1,True)
        self.assertEqual(bad['pnl'],0)
        self.assertFalse(bad['complete'])
        for seed in range(20):
            result=execute_route(tape,(0,1,2),100,obs,ExecutionConfig(partial_probability=1),seed,True)
            self.assertLess(result['residual'],1e-5)
            self.assertTrue(any(row['leg']=='cleanup' for row in result['trace']))


if __name__=='__main__':unittest.main()
