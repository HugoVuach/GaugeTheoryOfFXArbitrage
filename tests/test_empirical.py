"""Exact reference comparisons for the accelerated real-data kernel."""
import itertools
import unittest
import numpy as np
from fxresearch.empirical import karp_mean,best_funded_cycle,full_hodge
from fxresearch.graph import enumerate_cycles,cycle_mean,potential_diagnostic

class EmpiricalKernelTests(unittest.TestCase):
    def test_sparse_karp_and_funded_dp_against_all_simple_cycles(self):
        rng=np.random.default_rng(8012024)
        for n in range(2,9):
            cycles=list(enumerate_cycles(n));funded=list(enumerate_cycles(n,anchor=0))
            for rep in range(6):
                a=rng.normal(size=(n,n));a[rng.random((n,n))<.2]=-np.inf;np.fill_diagonal(a,-np.inf)
                truth=max(cycle_mean(a,c) for c in cycles)
                value=karp_mean(a)
                self.assertTrue((not np.isfinite(truth) and not np.isfinite(value)) or abs(truth-value)<1e-10)
                expected=max(cycle_mean(a,c) for c in funded)
                value,route,length=best_funded_cycle(a)
                self.assertTrue((not np.isfinite(expected) and not np.isfinite(value)) or abs(expected-value)<1e-10)
                if length:
                    c=tuple(int(x) for x in route[:length]);self.assertEqual(c[0],0);self.assertEqual(len(set(c)),length)
                    self.assertAlmostEqual(cycle_mean(a,c),value)

    def test_hodge_and_gauge(self):
        rng=np.random.default_rng(9)
        for n in range(3,9):
            a=rng.normal(size=(n,n));np.fill_diagonal(a,-np.inf)
            value,curvature=full_hodge(a)
            self.assertAlmostEqual(value,potential_diagnostic(a).rms)
            phi=rng.normal(size=n)*10000
            shifted=a+phi[None,:]-phi[:,None]
            after,after_curvature=full_hodge(shifted)
            self.assertAlmostEqual(value,after,places=9);self.assertAlmostEqual(curvature,after_curvature,places=8)

if __name__=='__main__':unittest.main()
