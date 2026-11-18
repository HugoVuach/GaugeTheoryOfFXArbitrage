import json,tempfile,unittest
from pathlib import Path
import numpy as np
from analyse_long_period import fit_training_scales
from study_dukascopy import calibration

class TrainingScaleTests(unittest.TestCase):
    def test_scales_are_unchanged_when_test_labels_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);days=root/'days';days.mkdir()
            for date,phase,value in [('2024-01-02','training',3.),('2024-10-01','test',1e6)]:
                (days/(date+'.json')).write_text(json.dumps({'date':date,'phase':phase,'status':'processed'}))
                np.savez(days/(date+'_events.npz'),lengths=np.full(200,3),max_ages=np.full(200,50.),gross=np.full(200,value),archived_realized=np.zeros((200,3)))
            before=fit_training_scales(root)
            np.savez(days/'2024-10-01_events.npz',lengths=np.full(200,3),max_ages=np.full(200,999.),gross=np.full(200,-1e9),archived_realized=np.full((200,3),1e9))
            after=fit_training_scales(root)
            np.testing.assert_array_equal(before,after);self.assertEqual(before[0,1,0],1.)

    def test_previously_explored_blocks_do_not_enter_calibration(self):
        from datetime import date,timedelta
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);days=root/'days';days.mkdir();dates=[];d=date(2024,4,1)
            while len(dates)<19:
                if d.weekday()<5:dates.append(str(d))
                d+=timedelta(days=1)
            for day in dates+['2024-06-10']:
                value=float('inf') if day=='2024-06-10' else 0.
                scores={str(k):{'cycle_mean_error':value,'edge_error':value,'archived_cycle_mean_error':value,'archived_edge_error':value,'label_missing':0} for k in [50,100,500]}
                (days/(day+'.json')).write_text(json.dumps({'date':day,'phase':'calibration','status':'processed','scores':scores}))
                np.savez(days/(day+'_events.npz'),lengths=np.empty(0),gross=np.empty(0),realized=np.empty((0,3)),archived_realized=np.empty((0,3)))
            rows=calibration(root,save=False)
            joint=next(r for r in rows if r['method']=='joint_archived' and r['delay_ms']==100)
            self.assertEqual(joint['calibration_days'],19);self.assertEqual(joint['radius_bp'],0.)

if __name__=='__main__':unittest.main()
