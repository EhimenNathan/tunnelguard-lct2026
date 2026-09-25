"""Website scenarios from the organisers' synthetic-obstacle bag (dataset 3), final model, real detector output:
  ds3_cube  - the 2×2 m cube in the centre of the gauge, from 100 m to the train (STOP from 98 m)
  ds3_row   - objects 3-7 in sequence: cube on the rail, at the gauge edge, just outside, 2×2 m at the edge, 2×2 m outside
usage (scratchpad):  python export_web_ds3.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'tunnel_guard'))

from export_web import Writer, MODEL
from ds3_stream import frames, nearest_object
from tunnel_guard.core.detector import ObstacleDetector, DetectorConfig
from tunnel_guard.core.geometry import warmup

WINDOWS = {'ds3_cube': (0.0, 23.5), 'ds3_row': (40.0, 58.5)}


def main():
    warmup()
    det = ObstacleDetector(DetectorConfig(scorer_model=MODEL))
    ws = {'ds3_cube': Writer('ds3_cube', 'Датасет 3 · куб 2×2 м в центре габарита'),
          'ds3_row': Writer('ds3_row', 'Датасет 3 · объекты 3–7 подряд')}
    for t, xyz in frames():
        res = det.process(xyz, t)
        for name, (a, b) in WINDOWS.items():
            if a <= t <= b:
                obj, dist = None, None
                x = det.calib.x
                o = nearest_object(x + 60.0)
                if o[0] - x > 0:
                    obj, dist = o[1], round(o[0] - x, 1)
                ws[name].add([res], det.cfg.gauge, t, note=f'{obj} · ≈{dist:.0f} м' if obj else None)
        if t > max(b for _, b in WINDOWS.values()):
            break
    about = ('Бэг организаторов: реальная запись тоннеля с их синтетическими объектами. Генератор ставит объекты на плоскость '
             'в системе лидара, а путь идёт под уклон — объекты «парят» над рельсами. Кадры — живой вывод финальной модели.')
    ws['ds3_cube'].save(dict(dataset=3, synthetic=True, models=['final'], about=about + ' Куб 2×2 м: STOP с 98 м.'))
    ws['ds3_row'].save(dict(dataset=3, synthetic=True, models=['final'],
                            about=about + ' Объекты 3–7: куб на рельсе, у края, за габаритом рядом, 2×2 м у края и за габаритом.'))


if __name__ == '__main__':
    main()
