"""Потоковая проверка zstd без записи распакованного архива. Ошибка => exit 2.
python tools/finals_archive_check.py ARCHIVE OUTPUT.json
"""
import sys,json,zstandard
from pathlib import Path
def main(src,out):
 report=dict(archive=Path(src).name,bytes=Path(src).stat().st_size,decompressed_bytes=0,valid=False,error=None)
 try:
  with open(src,'rb') as raw,zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw) as z:
   while chunk:=z.read(8*1024*1024):report['decompressed_bytes']+=len(chunk)
   report['compressed_position']=raw.tell()
  report['valid']=True
 except zstandard.ZstdError as exc:report['error']=str(exc)
 Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report,flush=True)
 if not report['valid']:raise SystemExit(2)
if __name__=='__main__':main(*sys.argv[1:3])
