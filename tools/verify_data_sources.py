"""Проверить версии исходных архивов без распаковки и сторонних библиотек."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--file',choices=['falcon_test.zst','cloud_with_fake_obj.zst','new_synth_data.zst']);a=p.parse_args()
    sources=json.loads((Path(__file__).resolve().parents[1]/'docs/data_sources.json').read_text(encoding='utf-8'))['files']
    passed=True
    for source in sources:
        if a.file and source['name']!=a.file:continue
        path=Path(a.directory)/source['name']
        if not path.is_file():print('НЕТ ФАЙЛА:',path);passed=False;continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        ok=path.stat().st_size==source['bytes'] and h.hexdigest()==source['sha256']
        print('СОВПАДАЕТ' if ok else 'ОШИБКА ВЕРСИИ/ЦЕЛОСТНОСТИ',source['name'],h.hexdigest());passed &= ok
    raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
