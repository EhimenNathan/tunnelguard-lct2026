"""Контролируемая регрессия тонкого провода: появление, провал отражений на 0.5с, повторное появление.
Это синтетический тест, а не замена повторному проигрыванию исходного бэга с проводом.
"""
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'src/tunnel_guard'),str(HERE.parent/'src/tunnel_guard/test')]
from tunnel_sim import make_tunnel
from tunnel_guard.core.detector import ObstacleDetector,DetectorConfig


def main(out):
    rows=[]
    for hold in (0,10):
        det=ObstacleDetector(DetectorConfig(forward_axis='x',mount_calibration=False,hold_frames=hold,
                                           scorer_model=str(HERE.parent/'src/tunnel_guard/config/obstacle_scorer.json')))
        states=[]
        for k in range(80):
            p=make_tunnel(seed=k)
            visible=30<=k<50 or 55<=k<80
            if visible:
                rng=np.random.default_rng(k)
                wire=np.column_stack([rng.normal(60.,.015,70),np.linspace(-1.1,1.1,70),rng.normal(.65,.012,70)])
                p=np.vstack([p,wire]).astype(np.float32)
            res=det.process(p,k*.1)
            hit=any(abs(o['distance']-60)<3 for o in res.obstacles)
            stop=any(abs(o['distance']-60)<3 and o['zone']==2 for o in res.obstacles)
            states.append(dict(k=k,visible=visible,hit=hit,stop=stop))
        rows.append(dict(hold_frames=hold,states=states,
                         dropout_hit=sum(r['hit'] for r in states[50:55]),
                         dropout_stop=sum(r['stop'] for r in states[50:55])))
    Path(out).write_text(json.dumps(rows,indent=1))
    print([(r['hold_frames'],r['dropout_hit'],r['dropout_stop']) for r in rows],flush=True)


if __name__=='__main__':main(sys.argv[1])
