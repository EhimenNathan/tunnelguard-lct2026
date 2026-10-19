"""Fetch an unpublished test archive with job credentials, verify, extract safely.
No credentials or signed URLs are written to logs/artifacts.
"""
import argparse,hashlib,json,os,shutil,tarfile
from pathlib import Path,PurePosixPath
import requests
import zstandard
import yaml
API='https://api.github.com/repos/EhimenNathan/tunnelguard-lct2026'

def download(release_id,target):
 session=requests.Session();session.headers['Authorization']='Bearer '+os.environ['TG_DATA_TOKEN']
 r=session.get(API+'/releases/'+release_id,timeout=45);r.raise_for_status();assert r.json()['draft'],'Expected unpublished data'
 r=session.get(API+'/releases/'+release_id+'/assets?per_page=100',timeout=45);r.raise_for_status();assets={a['name']:a for a in r.json()}
 def asset(asset_id):
  r=session.get(API+'/releases/assets/'+str(asset_id),headers={'Accept':'application/octet-stream'},stream=True,timeout=(30,180));r.raise_for_status();return r
 with asset(assets['manifest.json']['id']) as r:manifest=r.json()
 assert manifest['source_name']=='new_synth_data.zst'
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);whole=hashlib.sha256();offset=0
 with target.open('wb') as f:
  for part in manifest['parts']:
   assert part['offset']==offset and part['size']>0
   h=hashlib.sha256();size=0
   with asset(part['id']) as r:
    for b in r.iter_content(4*1024*1024):f.write(b);h.update(b);whole.update(b);size+=len(b)
   assert size==part['size'] and h.hexdigest()==part['sha256'],'Part integrity failed'
   offset+=size;print('verified',part['name'],size,flush=True)
 assert offset==manifest['archive_bytes'] and whole.hexdigest()==manifest['archive_sha256']
 (target.parent/'source.json').write_text(json.dumps(manifest,indent=2))

def public_download(public_key,target,remote_path):
 session=requests.Session();params={'public_key':public_key,'path':remote_path}
 info=session.get('https://cloud-api.yandex.net/v1/disk/public/resources',params=params,timeout=60);info.raise_for_status();info=info.json()
 assert info['type']=='file' and info.get('sha256')
 link=session.get('https://cloud-api.yandex.net/v1/disk/public/resources/download',params=params,timeout=60);link.raise_for_status()
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
 with session.get(link.json()['href'],stream=True,timeout=(45,180)) as response,target.open('wb') as f:
  response.raise_for_status()
  for chunk in response.iter_content(4*1024**2):f.write(chunk);h.update(chunk);size+=len(chunk)
 assert size==info['size'] and h.hexdigest()==info['sha256'], 'Public source integrity failed'
 (target.parent/(target.stem+'_source.json')).write_text(json.dumps(dict(source_name=info['name'],archive_bytes=size,archive_sha256=h.hexdigest(),public_key=public_key,remote_path=remote_path),indent=2))
 print('Verified public source',info['name'],size,h.hexdigest(),flush=True)

def extract(archive,root):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);members=[]
 with open(archive,'rb') as raw,zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw) as z,tarfile.open(fileobj=z,mode='r|') as tf:
  for m in tf:
   name=PurePosixPath(m.name)
   if name.is_absolute() or '..' in name.parts or not (m.isdir() or m.isfile()):raise ValueError('Unsafe member')
   dest=(root/str(name)).resolve()
   if not dest.is_relative_to(root):raise ValueError('Unsafe path')
   if m.isdir():dest.mkdir(parents=True,exist_ok=True);continue
   if m.size>shutil.disk_usage(root).free-256*1024*1024:raise OSError('Not enough cloud disk for this member')
   dest.parent.mkdir(parents=True,exist_ok=True)
   with tf.extractfile(m) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
   assert dest.stat().st_size==m.size
   members.append(dict(path=str(name),bytes=m.size))
 (root/'extraction.json').write_text(json.dumps(dict(expanded_bytes=sum(m['bytes'] for m in members),members=members),indent=2))
 bags=[]
 for p in sorted(root.rglob('metadata.yaml')):
  info=yaml.safe_load(p.read_text())['rosbag2_bagfile_information'];bags.append(dict(path=str(p.parent.relative_to(root)),frames=info['message_count']))
 assert sorted(b['frames'] for b in bags)==[386,545,736]
 (root/'bags.json').write_text(json.dumps(bags,indent=2));print(json.dumps(bags),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['download','extract','public-download']);ap.add_argument('source');ap.add_argument('target');ap.add_argument('--path',default='/new_synth_data.zst');a=ap.parse_args()
 if a.mode=='public-download':public_download(a.source,a.target,a.path)
 else:(download if a.mode=='download' else extract)(a.source,a.target)
