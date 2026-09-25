"""Stream the organisers' synthetic-obstacle bag (dataset 3: zstd -> tar -> rosbag2 sqlite, 16-byte points) frame by frame,
without extracting it.  Shared by the video and website exporters.  The archive we received is corrupted after 438 of
1510 frames; iteration stops cleanly there."""
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import zstandard

from convert import iter_messages
from sqlpages import parse_pc2, split_record

DS3 = os.environ.get('DS3_ARCHIVE', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'dataset3',
                                                  'cloud_with_fake_obj.zst'))
# the ten objects, located in the recording by their tunnel coordinate (lidar odometry of the final detector, m)
OBJECTS = [(102, '1 · куб 2×2 м в центре габарита'), (307, '2 · куб 0.3 м в центре габарита'),
           (407, '3 · куб 0.3 м на рельсе'), (507, '4 · куб 0.3 м у края габарита'),
           (607, '5 · куб 0.3 м за габаритом, рядом'), (707, '6 · куб 2×2 м у края, в габарите'),
           (808, '7 · куб 2×2 м за габаритом'), (905, '8 · куб 2×2 м «сверху габарита»'),
           (1002, '9 · брус 2×0.2 м лежит на рельсах'), (1118, '10 · стержень 5 см свисает с потолка')]


def frames(path=DS3):
    """Yield (t since start [s], xyz float32 (N,3) in the sensor frame)."""
    raw = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(open(path, 'rb'), read_size=1 << 20)
    tf = tarfile.open(fileobj=raw, mode='r|')
    t0 = None
    for m in tf:
        if not m.name.endswith('.db3'):
            continue
        try:
            for _, payload in iter_messages(tf.extractfile(m)):
                vals = split_record(payload)
                if len(vals) < 4 or not isinstance(vals[3], (bytes, bytearray)) or len(vals[3]) < 1000:
                    continue
                pc = parse_pc2(vals[3])
                t0 = pc['stamp'] if t0 is None else t0
                a = np.frombuffer(pc['data'], np.float32).reshape(-1, pc['point_step'] // 4)[:, :3]
                yield pc['stamp'] - t0, np.ascontiguousarray(a[np.isfinite(a).all(1)])
        except zstandard.ZstdError:
            return          # corrupted tail of the archive
        return


def nearest_object(x_ahead):
    """(name, distance) of the object closest ahead of tunnel coordinate x_ahead, or None."""
    return min(OBJECTS, key=lambda o: abs(o[0] - x_ahead))
