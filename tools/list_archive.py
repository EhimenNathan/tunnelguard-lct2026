"""Read-only listing of a (zip ->) zstd -> tar archive of rosbag2 recordings: member names, sizes and each bag's
metadata.yaml.  Streams everything; writes nothing.   usage: python list_archive.py <bags.zip | data.zst>"""
import sys
import tarfile
import time
import zipfile

import zstandard


def open_zst_stream(path):
    if path.lower().endswith('.zip'):
        z = zipfile.ZipFile(path)
        member = next(i for i in z.infolist() if i.filename.endswith('.zst'))
        raw = z.open(member)
    else:
        raw = open(path, 'rb')
    return zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw, read_size=1 << 20)


if __name__ == '__main__':
    t0 = time.time()
    tf = tarfile.open(fileobj=open_zst_stream(sys.argv[1]), mode='r|')
    total = 0
    for m in tf:
        total += m.size
        print(f'{m.size / 1e9:8.3f} GB  {m.name}   [{time.time() - t0:.0f}s]', flush=True)
        if m.name.endswith('metadata.yaml'):
            print(tf.extractfile(m).read().decode('utf-8', 'replace'), flush=True)
    print(f'total {total / 1e9:.2f} GB in {time.time() - t0:.0f}s')
