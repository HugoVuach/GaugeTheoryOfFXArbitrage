import lzma,tempfile,unittest
from pathlib import Path
import numpy as np
from collect_dukascopy import TICK_DTYPE,decode
from study_dukascopy import DayTape,BASE,PAIRS,upper_quantile
from fxresearch.historical_replay import price_labels,replay_events

class HistoricalReplayTests(unittest.TestCase):
    def tape(self,root,small_final=False):
        day='2024-01-02';folder=root/day;folder.mkdir()
        values={'USDJPY':(1.1,1.101,.15,.15),'EURJPY':(1.249,1.25,2.,2.),'EURUSD':(1.2,1.2001,.00001 if small_final else 2.,2.)}
        for pair,(bid,ask,bv,av) in values.items():
            a=np.zeros(2,dtype=TICK_DTYPE);a['ms']=[0,1000];scale=1000 if pair.endswith('JPY') else 100000
            a['bid']=round(bid*scale);a['ask']=round(ask*scale);a['bid_volume']=bv;a['ask_volume']=av
            (folder/(pair+'.bi5')).write_bytes(lzma.compress(a.tobytes(),format=lzma.FORMAT_ALONE))
        return DayTape(root,day)

    def test_real_units_consumption_and_cash(self):
        with tempfile.TemporaryDirectory() as temp:
            tape=self.tape(Path(temp));t=tape.start
            routes=np.array([[0,3,1,-1,-1,-1,-1,-1]]*2,np.int8);lengths=np.array([3,3],np.int64)
            gross=np.full(2,10000*np.log(1.1*.8*1.2));events=np.array([t,t+1000],np.int64)
            records=replay_events(events,routes,lengths,gross,np.array([1,2]),np.zeros(2),0.,0.,100,0.,0,*tape.flat)
            self.assertEqual(len(records),2);self.assertAlmostEqual(records[0,4],5600.,places=6)
            # Identical quote updates must not replenish the first leg's 150k USD.
            self.assertEqual(records[1,5],0);self.assertEqual(records[1,9],0);self.assertAlmostEqual(records[1,4],0)

    def test_unclosed_inventory_not_a_zero_profit(self):
        with tempfile.TemporaryDirectory() as temp:
            tape=self.tape(Path(temp),True);t=tape.start
            routes=np.array([[0,3,1,-1,-1,-1,-1,-1]],np.int8)
            records=replay_events(np.array([t]),routes,np.array([3]),np.array([1000.]),np.array([1]),np.zeros(1),0.,0.,100,0.,0,*tape.flat)
            self.assertEqual(records[0,6],0);self.assertTrue(np.isnan(records[0,4]))
            self.assertEqual(records[0,12],1);self.assertGreater(records[0,13],0)
            self.assertAlmostEqual(records[0,14],12.,places=5)
            self.assertAlmostEqual(records[0,7],1000000.-100000.+12.,places=5)

    def test_archive_price_is_not_fresh_executable_price(self):
        with tempfile.TemporaryDirectory() as temp:
            tape=self.tape(Path(temp));t=tape.start
            route=np.array([[0,3,1,-1,-1,-1,-1,-1]],np.int8);delays=np.array([1000],np.int64)
            strict,_=price_labels(np.array([t]),route,np.array([3]),np.zeros(1),delays,*tape.flat)
            quoted,_=price_labels(np.array([t]),route,np.array([3]),np.zeros(1),delays,*tape.flat,np.inf)
            self.assertTrue(np.isnan(strict[0,0]));self.assertTrue(np.isfinite(quoted[0,0]))

    def test_unknown_labels_and_rank_resolution(self):
        self.assertTrue(np.isinf(upper_quantile([0.]*18)))
        self.assertEqual(upper_quantile([0.]*19),0)
        self.assertTrue(np.isinf(upper_quantile([0.]*20+[np.inf])))
        self.assertEqual(upper_quantile([0.]*99+[np.inf]),0)

    def test_histdata_cache_preserves_missing_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'202401'/'EURUSD';folder.mkdir(parents=True)
            t=1704153600123
            np.savez(folder/'2024-01-02.npz',times=np.array([t],np.int64),bid=[1.1],ask=[1.1001])
            tape=DayTape(root,'2024-01-02','histdata');index=PAIRS.index('EURUSD')
            self.assertEqual(tape.parts[index][0][0],t);self.assertTrue(np.isnan(tape.parts[index][3][0]));self.assertFalse(tape.volume_known)

if __name__=='__main__':unittest.main()
