"""Русское демонстрационное видео 1920×1080 из тех же реальных кадров, что использует сайт.
python tools/finals_video.py WEB_DIR OUTPUT.mp4
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

W,H,FPS=1920,1080,20
BG='#070d15';PANEL='#0c1723';INK='#e9f1f4';MUTED='#8b9da9';CYAN='#6cdeeb';RED='#ff7379';GOLD='#f1c37d'
fonts=Path('C:/Windows/Fonts')
def font(size,bold=False):
    local=fonts/('segoeuib.ttf' if bold else 'segoeui.ttf')
    return ImageFont.truetype(str(local if local.exists() else Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')),size)
def text(d,xy,s,size=25,color=INK,bold=False):d.text(xy,str(s),font=font(size,bold),fill=color)
def canvas(chapter,title,subtitle=''):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(70,40),'TUNNELGUARD',29,INK,True);text(d,(1370,44),'САХАЛИИН / ЛЦТ 2026',18,MUTED)
    d.line((70,100,1850,100),fill='#203141',width=2)
    text(d,(70,125),chapter,17,CYAN);text(d,(70,158),title,45,INK,True)
    if subtitle:text(d,(70,978),subtitle,23,MUTED)
    text(d,(70,1035),'РЕАЛЬНЫЕ ДАННЫЕ / ВОСПРОИЗВОДИМЫЕ ЭКСПЕРИМЕНТЫ / ОТКРЫТЫЙ КОД',14,MUTED)
    return im

def project(points,eye,target,rect):
    p=np.asarray(points,float).reshape(-1,3)
    p=np.column_stack([p[:,1],p[:,2],-p[:,0]])
    eye=np.array(eye);f=np.array(target)-eye;f=f/np.linalg.norm(f)
    right=np.cross(f,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,f)
    v=p-eye;depth=v@f
    x0,y0,x1,y1=rect;focal=.8*(x1-x0)
    with np.errstate(divide='ignore',invalid='ignore'):
        xx=(x0+x1)/2+focal*(v@right)/depth;yy=(y0+y1)/2-focal*(v@up)/depth
    ok=(depth>.5)&(xx>x0)&(xx<x1)&(yy>y0)&(yy<y1)&np.isfinite(xx)&np.isfinite(yy)
    return np.column_stack([xx,yy]),ok

def frame_image(f,title,chapter,focus=False):
    synthetic=f.get('synthetic',False)
    subtitle='СИНТЕТИКА: человек на 200 м и пыль на 120 м в реальном фоне Falcon.' if synthetic else 'Реальное облако Falcon. Показаны сохранённые решения детектора; воспроизведение ускорено.'
    im=canvas(chapter,title,subtitle);d=ImageDraw.Draw(im)
    rect=(70,245,1340,925);d.rectangle(rect,fill='#09121e',outline='#263b4b',width=2)
    target=f.get('truth_x') or 150
    if focus:
        eye=[8,5,-target+20];look=[.35,-.7,-target]
    else:eye=[24,26,16];look=[0,0,-85]
    p=np.asarray(f['p']).reshape(-1,3)
    xy,ok=project(p,eye,look,rect)
    for x,y in xy[ok]:d.ellipse((x-1,y-1,x+1,y+1),fill='#7e9aaf')
    line=np.asarray(f['line'])
    if len(line):
        for lat,h in [(-1.45,0),(1.45,0),(-1.45,3.5),(1.45,3.5)]:
            q=line+np.array([0,lat,h]);xx,v=project(q,eye,look,rect)
            for i in range(1,len(xx)):
                if v[i] and v[i-1]:d.line([tuple(xx[i-1]),tuple(xx[i])],fill=CYAN if q[i,0]<=f['measured'] else '#576986',width=2)
        for x,y,z in line[::2]:
            q=np.array([[x,y-1.45,z],[x,y+1.45,z],[x,y+1.45,z+3.5],[x,y-1.45,z+3.5],[x,y-1.45,z]])
            xx,v=project(q,eye,look,rect)
            if v.all():d.line([tuple(a) for a in xx],fill='#355c70',width=1)
    for o in f['boxes']:
        c=np.asarray(o['c']);sz=np.maximum(o['size'],[.25,.25,.25]);q=[]
        for a in (-1,1):
            for b in (-1,1):
                for cc in (-1,1):q.append(c+sz*np.array([a,b,cc])*.5)
        xx,v=project(q,eye,look,rect);color=RED if o['zone']==2 else GOLD
        for i in range(8):
            for bit in (1,2,4):
                j=i^bit
                if j>i and v[i] and v[j]:d.line([tuple(xx[i]),tuple(xx[j])],fill=color,width=3)
        center,valid=project([c],eye,look,rect)
        if valid[0]:text(d,(center[0,0]+10,center[0,1]-35),f"{o['d']:.1f} м",21,color,True)
    text(d,(95,265),'СИНТЕТИЧЕСКАЯ ИНЪЕКЦИЯ' if synthetic else 'FALCON / РЕАЛЬНАЯ ЗАПИСЬ',17,GOLD if synthetic else CYAN)
    text(d,(95,886),f"Кадр {f['k']:04d} · t = {f['t']:.1f} с",18,MUTED)
    sx=1390
    text(d,(sx,250),'РЕШЕНИЕ СИСТЕМЫ',17,MUTED)
    text(d,(sx,286),['СВОБОДНО','ВНИМАНИЕ','СТОП'][f['level']],44,[CYAN,GOLD,RED][f['level']],True)
    values=[('Цель по траектории',f'{target:.1f} м' if f.get('truth_x') is not None else 'нет разметки'),('Габарит / экстраполяция',f"{f['env']:.0f} м"),
            ('Измеренная геометрия',f"{f['measured']:.0f} м"),('Свободный путь',f"{f['clear']:.0f} м")]
    for i,(label,val) in enumerate(values):
        y=383+i*110;text(d,(sx,y),label,20,MUTED);text(d,(sx,y+35),val,34,INK,True)
        d.line((sx,y+92,1820,y+92),fill='#203141')
    text(d,(sx,855),'Точки прорежены только для показа.',17,MUTED)
    text(d,(sx,885),'Решения — на полном облаке.',17,MUTED)
    return im

def main(web,out):
    web=Path(web);data=web/'data'
    s=json.loads((data/'finals_summary.json').read_text());cloud=json.loads((data/'cloud_live.json').read_text())
    ladder=json.loads((data/'finals_ladder.json').read_text())
    writer=imageio.get_writer(out,fps=FPS,codec='libx264',macro_block_size=2,
                            ffmpeg_params=['-crf','22','-preset','fast'])
    def put(im,seconds):
        a=np.asarray(im)
        for _ in range(int(seconds*FPS)):writer.append_data(a)
    intro=canvas('ФИНАЛ / 2026','Видеть путь. Замечать раньше.');d=ImageDraw.Draw(intro)
    text(d,(80,325),'ОБНАРУЖЕНИЕ ПРЕПЯТСТВИЙ',64,INK,True)
    text(d,(80,410),'В ГАБАРИТЕ ПОЕЗДА МЕТРО',64,CYAN,True)
    text(d,(85,575),'3D-лидар → геометрия пути → гибридная модель → временное подтверждение',29,MUTED)
    text(d,(85,700),'Реальная запись Falcon Prime · 4 443 кадра · статичный поезд',30,INK)
    text(d,(85,760),'Синтетические тесты размечены отдельно.',25,MUTED)
    intro.save(web/'assets/video_poster.png');put(intro,3)
    specs=[('falcon_clip_0.json','Человек в дальнем поле','01 / РЕАЛЬНЫЙ FALCON',True),
           ('falcon_clip_1.json','От дальнего предупреждения — к устойчивому треку','02 / ПРИБЛИЖЕНИЕ',False),
           ('falcon_clip_2.json','Непрерывность решения в ближней зоне','03 / ПОДТВЕРЖДЕНИЕ',False),
           ('falcon_clip_3.json','Пыль и человек: разный физический отклик','04 / СИНТЕТИКА В РЕАЛЬНОМ ФОНЕ',False)]
    for filename,title,chapter,focus in specs:
        frames=json.loads((data/filename).read_text())
        if filename=='falcon_clip_0.json':frames=frames[18:]
        for f in frames:
            im=frame_image(f,title,chapter,focus)
            put(im,.15)
        if filename=='falcon_clip_0.json':
            im.save(web/'assets/falcon_scene.png')
            im.save(Path(out).parent/'falcon_scene.png')
    im=canvas('05 / ГИБРИДНАЯ МОДЕЛЬ','Физика + два специалиста + память');d=ImageDraw.Draw(im)
    stages=[('01','Оси и монтаж','Falcon / ROS, наклон, угловая плотность'),('02','Геометрия и σ','Рельсы, кривизна, шаблон тоннеля, C++'),
            ('03','LightGBM + CatBoost','½ + ½ в логитах; отдельный подвесной эксперт'),('04','Треки и дальняя память','Компенсация движения, M из N, удержание')]
    for i,(num,title,desc) in enumerate(stages):
        y=285+i*150;text(d,(85,y),num,27,CYAN);text(d,(165,y),title,35,INK,True);text(d,(165,y+55),desc,25,MUTED)
    put(im,5)
    for kind,title in [('person','Человек'),('dust','Пыль: цель — отсутствие ложного СТОП')]:
        im=canvas('06 / ЛЕСТНИЦА ДАЛЬНОСТЕЙ',title,'Синтетические инъекции в реальный фон. Один seed на дистанцию; 50 кадров с объектом.');d=ImageDraw.Draw(im)
        rr=[r for r in ladder if r['object']==kind]
        for i,row in enumerate(rr):
            x=95+i*102;y=835;v=row['detected'];stop=row['stop']
            d.rectangle((x,y-400*v,x+35,y),fill=CYAN);d.rectangle((x+38,y-400*stop,x+73,y),fill=RED)
            text(d,(x,y+20),row['distance'],18,MUTED)
            text(d,(x-2,y-400*v-35),f'{100*v:.0f}%',17,INK)
        text(d,(95,290),'Ложное препятствие' if kind=='dust' else 'Обнаружение',23,CYAN);text(d,(440,290),'Ложный СТОП' if kind=='dust' else 'СТОП',23,RED)
        if kind=='dust':
            text(d,(200,470),'0 ложных СТОП / 850 кадров с пылью',48,CYAN,True)
            text(d,(200,565),'Зона пыли измеряется отдельно; на 300 м флаг пропадает.',28,MUTED)
        put(im,4)
    im=canvas('07 / ИЗМЕРЕННЫЙ РЕЗУЛЬТАТ','Числа, которые можно проверить.');d=ImageDraw.Draw(im)
    vals=[(f"{s['first_detection']['truth_x']:.1f} м",'первое обнаружение человека по автоматической траектории'),
          (f"{s['envelope_median_m']:.0f} м",'медианная длина габарита, включая экстраполяцию'),
          (f"{cloud['output_hz']:.2f} Гц",f"живая ROS-выдача в облаке · {cloud['received']}/{cloud['expected']} сообщений"),
          (f"{cloud['processing_ms_p95']:.1f} мс",'p95 обработки в живом ROS-узле на облачном раннере')]
    for i,(val,label) in enumerate(vals):
        y=295+i*145;text(d,(85,y),val,49,CYAN,True);text(d,(500,y+14),label,25,INK)
    text(d,(85,915),'Автотраектория — proxy, не независимая ручная разметка. Дальний диапазон требует улучшения.',21,MUTED)
    put(im,5)
    for filename,title in [('ablation_current.png','Вклад компонентов: текущий этап'),('distance_ladder_current.png','Все 68 условий: включая пропуски')]:
        plot=Image.open(web/'assets'/filename).convert('RGB');plot.thumbnail((1800,780))
        im=canvas('08 / КОНТРОЛИРУЕМЫЕ ЭКСПЕРИМЕНТЫ',title)
        im.paste(plot,((W-plot.width)//2,245));put(im,5)
    im=canvas('08 / ПРОВЕРИТЬ САМОСТОЯТЕЛЬНО','Docker и RViz — в облаке.');d=ImageDraw.Draw(im)
    text(d,(90,330),'Codespaces → порт 6080 → рабочий стол → tg play',42,CYAN,True)
    for i,t in enumerate(['Пробел — пауза / продолжить','↑ / ↓ — скорость · → — следующий кадр','Ctrl+C → tg play другой_бэг · контейнер продолжает работать']):text(d,(90,450+i*80),t,30,INK)
    text(d,(90,780),'github.com/EhimenNathan/tunnelguard-lct2026',35,INK,True)
    text(d,(90,845),'Код · русская документация · 3D-сайт · протоколы тестов',27,MUTED)
    put(im,4);writer.close();print('Video saved',out,Path(out).stat().st_size,flush=True)

if __name__=='__main__':main(*sys.argv[1:3])
