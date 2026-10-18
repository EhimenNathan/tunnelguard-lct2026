"""Русское видео из трёх записей организаторов, вид вперёд из кабины.

python tools/finals_cab_video.py web OUTPUT.mp4
Для 1× нужны все исходные кадры 10 Гц; облака/детекции не интерполируются.
"""
import json
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from finals_video import canvas, text, project, W, H, FPS

BG='#10243a';GREEN='#21e6a7';VIOLET='#ad91ff';RED='#ff526d';GOLD='#ffc247';INK='#e9f1f4';MUTED='#a9c3d6'
NAMES=['Платформа','Стрелка','Тоннель']

def colors(p):
    # Same height ramp as web/finals.js: HSL(.62-.48*h, .82, .55).
    import colorsys
    h=np.clip((p[:,2]+2.2)/5,0,1)
    return [tuple(round(c*255) for c in colorsys.hls_to_rgb(.62-.48*v,.55,.82)) for v in h]

def render(f,active,profile,rate=1.,step=.1):
    im=canvas('ТРИ ЗАПИСИ ОРГАНИЗАТОРОВ / ВИД ИЗ КАБИНЫ','Путь впереди. Препятствия в габарите.')
    d=ImageDraw.Draw(im)
    for i,name in enumerate(NAMES):
        x=70+i*470;selected=i==active
        d.rounded_rectangle((x,225,x+450,286),radius=12,fill='#174a47' if selected else '#162c43',outline=GREEN if selected else '#34536c',width=2)
        text(d,(x+20,238),f'0{i+1}  {name}',24,GREEN if selected else MUTED,selected)
    rect=(70,310,1450,917);d.rounded_rectangle(rect,radius=12,fill=BG,outline='#34536c',width=2)
    # Fixed forward-facing sensor-relative camera. No orbit or target-following
    # changes the main cab view; pitch aims slightly down the track.
    eye=[0,.9,0];look=[0,-.6,-70]
    p=np.asarray(f['p'],float).reshape(-1,3);xy,ok=project(p,eye,look,rect)
    rgb=colors(p)
    for j in np.flatnonzero(ok):
        x,y=xy[j];d.ellipse((x-1.4,y-1.4,x+1.4,y+1.4),fill=rgb[j])
    line=np.asarray(f['line'],float).reshape(-1,3)
    def poly(q,color,width=2):
        xx,v=project(q,eye,look,rect)
        for j in range(1,len(xx)):
            if v[j] and v[j-1]:d.line([tuple(xx[j-1]),tuple(xx[j])],fill=color,width=width)
    if len(line):
        # Actual configured gauge; start beyond the immediate cab foreground.
        rings=[]
        shape=np.array([[w,h] for h,w in profile]+[[-w,h] for h,w in profile[::-1]]+[[profile[0][1],profile[0][0]]])
        for x,y,z in line:
            if x<12:continue
            q=np.column_stack([np.full(len(shape),x),y+shape[:,0],z+shape[:,1]])
            rings.append((x,q))
            if round(x)%20==0:poly(q,GREEN if x<=f['measured'] else VIOLET,1)
        for j in [0,len(profile)-1,len(profile),2*len(profile)-1]:
            for i in range(1,len(rings)):
                poly([rings[i-1][1][j],rings[i][1][j]],GREEN if rings[i][0]<=f['measured'] else VIOLET,2)
    objects=f.get('boxes',[])
    labels=[]
    for obj in sorted(objects,key=lambda o:o['d']):
        c=np.array(obj['c']);sz=np.maximum(obj['size'],.18)
        q=np.array([c+sz*np.array([a,b,z])*.5 for a in (-1,1) for b in (-1,1) for z in (-1,1)])
        xx,v=project(q,eye,look,rect);color=RED if obj['zone']==2 else GOLD
        for i in range(8):
            for bit in (1,2,4):
                j=i^bit
                if j>i and v[i] and v[j]:d.line([tuple(xx[i]),tuple(xx[j])],fill=color,width=3)
        center,visible=project([c],eye,look,rect)
        if visible[0]:
            x,y=center[0]
            available=[]
            for dx in (35,-180,200,-345):
                for dy in (-65,-117,-13,-169,39):
                    tx=min(1210,max(95,x+dx));ty=min(810,max(390,y+dy))
                    bounds=(tx-8,ty-4,tx+135,ty+38)
                    if all(bounds[2]+8<a or b+8<bounds[0] or bounds[3]+8<c or e+8<bounds[1]
                           for a,c,b,e in labels):
                        available.append((tx,ty,bounds))
            if not available:continue  # keep the actual box; never overlap unreadable labels
            tx,ty,bounds=available[0];labels.append(bounds)
            d.line((x,y,tx,ty+22),fill=color,width=2)
            label=f"{obj['d']:.1f} м";d.rounded_rectangle(bounds,radius=7,fill='#0c1d2d',outline=color)
            text(d,(tx,ty),label,23,color,True)
    for zone in f.get('dust',[]):
        distance=zone.get('d',zone.get('distance'))
        if distance is None or not len(line):continue
        q=line[np.argmin(abs(line[:,0]-distance))].copy();q[2]+=1.2
        xx,v=project([q],eye,look,rect)
        if v[0]:
            x,y=xx[0];d.ellipse((x-9,y-9,x+9,y+9),outline=GOLD,width=3)
    d.rounded_rectangle((90,330,510,372),radius=7,fill='#0c1d2d')
    text(d,(105,338),'КАМЕРА ПО ХОДУ ПОЕЗДА / HESAI',17,MUTED)
    d.rounded_rectangle((90,856,590,898),radius=7,fill='#0c1d2d')
    text(d,(105,864),f"{NAMES[active]} · t = {f['t']:.1f} с · кадр {f['local']}",20,INK)
    sx=1490
    text(d,(sx,320),'РЕШЕНИЕ СИСТЕМЫ',16,MUTED)
    text(d,(sx,355),['СВОБОДНО','ВНИМАНИЕ','СТОП'][f['level']],34,[GREEN,GOLD,RED][f['level']],True)
    nearest=min((o['d'] for o in objects),default=None)
    vals=[('Препятствие',f'{nearest:.1f} м' if nearest is not None else 'нет детекции'),('Измеренный путь',f"{f['measured']:.0f} м"),('Габарит + прогноз',f"{f['env']:.0f} м"),('Зона аэрозоля', 'есть · не СТОП' if f.get('dust') else 'нет флага')]
    for i,(label,value) in enumerate(vals):
        y=425+i*105;text(d,(sx,y),label,19,MUTED);text(d,(sx,y+30),value,28,INK,True)
    for i,(color,label) in enumerate([(GREEN,'Габарит по измерениям'),(VIOLET,'Экстраполяция'),(RED,'Препятствие: СТОП'),(GOLD,'Внимание / зона пыли')]):
        y=856+i*25;d.ellipse((sx,y+6,sx+9,y+15),fill=color);text(d,(sx+20,y),label,16,MUTED)
    text(d,(70,952),'Цвет точек — высота. Рамки — сохранённый вывод детектора; классы объектов не размечены.',21,MUTED)
    text(d,(70,987),f'Синтетические записи организаторов · {rate:g}× (реальное время при 1×) · исходные облака {1/step:.0f} Гц.',19,MUTED)
    return im

def main(web,out,rate=1.):
    web=Path(web);out=Path(out);profile=json.loads((web/'data/gauge_profile.json').read_text())['profile']
    scenes=[json.loads((web/f'data/new_synth_{name}.json').read_text()) for name in ['platform','switch','tunnels']]
    durations=[p['duration_s'] for p in json.loads((web.parent/'docs/finals/new_synth/summary.json').read_text())['by_piece']]
    steps=[float(np.median(np.diff([f['t'] for f in frames]))) for frames in scenes]
    if rate<=0:raise ValueError('Скорость должна быть положительной')
    if any(s>.15 for s in steps):raise ValueError('Нужны все исходные облака 10 Гц. Сначала finals_organizer_bundle.py --clip-stride 1 и finals_prepare_video.py; клипы сайта прорежены.')
    writer=imageio.get_writer(str(out),fps=FPS,codec='libx264',macro_block_size=2,pixelformat='yuv420p',ffmpeg_params=['-crf','19','-maxrate','3M','-bufsize','6M','-preset','fast','-threads','2','-movflags','+faststart'])
    manifest={'camera':'fixed forward-facing cab','source':'all original organizer new_synth_data scans, native detector','playback_rate':rate,'fps':FPS,'resolution':[W,H],'scenes':[],'source_scan_step_s':steps}
    manifest['detector_commit']=json.loads((web.parent/'docs/finals/new_synth/summary.json').read_text())['commit']
    manifest['generator_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    manifest['generator_source_sha256']=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest['generator_hash_format']='UTF-8 LF'
    manifest['video_source_clip_sha256']={name:hashlib.sha256((web/f'data/new_synth_{name}.json').read_bytes()).hexdigest() for name in ['platform','switch','tunnels']}
    count=0
    try:
        for i,frames in enumerate(scenes):
            start=count
            # Cold-start frames remain visible, with their actual unconfirmed state.
            for j,f in enumerate(frames):
                next_t=frames[j+1]['t'] if j+1<len(frames) else durations[i]+steps[i]
                end=start+round((next_t-frames[0]['t'])/rate*FPS)
                im=render(f,i,profile,rate,steps[i]);a=np.asarray(im)
                for _ in range(max(0,end-count)):writer.append_data(a);count+=1
                if j==len(frames)//2:
                    im.save(out.parent/f'cab_{["platform","switch","tunnels"][i]}.png')
                    if i==0:im.save(web/'assets/video_poster.png')
            manifest['scenes'].append({'name':NAMES[i],'frames':len(frames),'source_start_s':frames[0]['t'],'last_saved_frame_s':frames[-1]['t'],'source_end_s':durations[i],'last_scan_hold_s':steps[i],'video_start_s':start/FPS,'video_end_s':count/FPS})
            print('Rendered',NAMES[i],count/FPS,'s',flush=True)
    finally:writer.close()
    manifest['duration_s']=count/FPS;manifest['output_bytes']=out.stat().st_size
    manifest['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    (web/'data/cab_video.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',out,manifest['output_bytes'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('web');ap.add_argument('out');ap.add_argument('--rate',type=float,default=1.);a=ap.parse_args();main(a.web,a.out,a.rate)
