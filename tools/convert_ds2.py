"""Convert the split rosbag2 recording in a (zip ->) zstd -> tar archive into per-piece range-image caches
(tools/convert.py format), streaming - nothing is extracted.  Resumable; pauses when the disk is nearly full.
usage (scratchpad):  python convert_ds2.py <bags.zip> <out_dir>"""
import os
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert import convert_bag
from list_archive import open_zst_stream


def wait_for_disk(min_free=700e6):
    while shutil.disk_usage(os.path.abspath(os.sep)).free < min_free:
        print('low disk space - waiting', flush=True)
        time.sleep(30)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    n = 0
    tf = tarfile.open(fileobj=open_zst_stream(src), mode='r|')
    for m in tf:
        if not m.name.endswith('.db3'):
            continue
        piece = os.path.basename(m.name)[:-4]
        prefix = os.path.join(out, piece)
        if os.path.exists(prefix + '_meta.json'):
            continue
        wait_for_disk()
        meta = convert_bag(tf.extractfile(m), prefix, keep_intensity=False)
        n += 1
        print(f'{piece}: {meta["n"]} frames, W={meta["W"]}  [{n} pieces, {time.time() - t0:.0f}s]', flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
