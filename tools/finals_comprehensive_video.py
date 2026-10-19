"""Russian evidence film: real Falcon, qualification, organizer synth, 68 injections.
Every scan excerpt plays at 1x. Synthetic percentages use full scored trials.
"""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image,ImageDraw
from finals_video import canvas,text,project,W,H,FPS
from finals_cab_video import render,colors,GREEN,GOLD,RED,INK,MUTED

KINDS={'person':'Человек','box_0.5':'Ящик 0,5 м','box_0.3':'Ящик 0,3 м','dust':'Пылевой аэрозоль'}

def card(chapter,title,lines):
    im=canvas(chapter,title);d=ImageDraw.Draw(im)
    for i,line in enumerate(lines):text(d,(85,310+i*90),line,29,INK if i==0 else MUTED,i==0)
    d.rectangle((65,1026,1855,1072),fill='#070d15')
    text(d,(70,1035),'КАЖДАЯ ЗАПИСЬ И ИНЪЕКЦИЯ ПОМЕЧЕНЫ / СКОРОСТЬ ФРАГМЕНТОВ 1× / НЕТ ВЫДУМАННЫХ ДЕТЕКЦИЙ',14,MUTED)
    return im

def zoom(im,f):
    target=f.get('truth_x')
    if target is None:return
    p=np.asarray(f['p'],float).reshape(-1,3)
    q=p[np.abs(p[:,0]-target)<7]
    if not len(q):return
    rect=(1010,380,1425,640);d=ImageDraw.Draw(im)
    d.rounded_rectangle(rect,radius=8,fill='#091826',outline='#547086',width=2)
    text(d,(1022,390),'УВЕЛИЧЕНИЕ / ТЕ ЖЕ ВОЗВРАТЫ',15,MUTED)
    # Sensor-relative orthographic diagnostic inset; not a fabricated photograph.
    lat=float(f.get('truth_y') or .35);x=1217+(q[:,1]-lat)*110;y=608-(q[:,2]+1.8)*92
    for xx,yy,rgb in zip(x,y,colors(q)):
        if 1020<xx<1415 and 420<yy<625:d.ellipse((xx-2,yy-2,xx+2,yy+2),fill=rgb)
    text(d,(1022,615),f'Автотраектория ≈ {target:.1f} м',15,MUTED)

def main(quality,dense,web,out):
    quality,dense,web,out=map(Path,[quality,dense,web,out]);out.parent.mkdir(parents=True,exist_ok=True)
    profile=json.loads((web/'data/gauge_profile.json').read_text())['profile']
    summary=json.loads((quality/'candidate/summary.json').read_text())
    ladder=json.loads((quality/'candidate/ladder.json').read_text())
    qualification=json.loads((dense/'qualification/summary.json').read_text())
    organizer_summary=json.loads((dense/'offline/summary.json').read_text())
    assert summary['source_sha256']==qualification['source_sha256']==organizer_summary['source_sha256']
    assert summary['model_sha256']==qualification['model_sha256']==organizer_summary['model_sha256']
    assert summary['native'] and qualification['native'] and qualification['complete']
    assert qualification['frames']==1510 and len(ladder)==68
    manifest={'playback_rate':1,'fps':FPS,'resolution':[W,H],'chapters':[],'scored_injection_conditions':68,
      'injection_matching':'diagnostic ladder: longitudinal distance within 4m; 255 regression conditions also check lateral location',
      'injection_excerpt':'last 20 of 50 scored frames, predefined uniformly for every case; 30 warmup frames precede scoring',
      'limitations':'Automatic Falcon trajectory; no independent real obstacle/dust labels; shared-background synthetic diagnostics, not proof of unseen-route generalization',
      'detector_source_sha256':summary['source_sha256'],'source_label':summary['source_label']}
    runtime=summary['source_label'][:7];timeline=0.;samples=[]
    with tempfile.TemporaryDirectory(prefix='tg-film-',dir=out.parent) as scratch:
        scratch=Path(scratch);parts=[];current=[];writer=None;count=0
        def start(name):
            nonlocal writer,count,current
            p=scratch/(f'part_{len(parts):02d}.mp4');parts.append(p);current=[name,timeline];count=0
            writer=imageio.get_writer(str(p),fps=FPS,codec='libx264',pixelformat='yuv420p',macro_block_size=2,
              ffmpeg_params=['-crf','19','-maxrate','1700k','-bufsize','3400k','-preset','fast','-threads','2','-movflags','+faststart'])
        def put(im,n):
            nonlocal count
            a=np.asarray(im)
            for _ in range(n):writer.append_data(a);count+=1
        def close(source):
            nonlocal timeline
            writer.close();duration=count/FPS
            manifest['chapters'].append(dict(name=current[0],video_start_s=timeline,duration_s=duration,source=source));timeline+=duration
            print('Rendered',current[0],duration,'seconds',flush=True)
        start('Введение');put(card('ФИНАЛ / ПРОВЕРЯЕМЫЕ ДАННЫЕ','Четыре источника. Разные задачи проверки.',[
          'Реальная запись Falcon → человек на дальней дистанции.',
          'Квалификация → запись с искусственными объектами.',
          'Новые синтетические записи организаторов → три отдельные сцены.',
          'Наши инъекции → 17 дальностей × человек, два ящика, пыль.',
          'Показаны реальные выходы алгоритма. У каждого эксперимента есть ограничения.']),6*FPS);close('explanatory title')
        for ci,title in enumerate(['Дальний человек','Средняя дистанция','Ближняя зона']):
            frames=json.loads((quality/f'candidate/falcon_clip_{ci}.json').read_text());assert len(frames)>100
            assert .09<float(np.median(np.diff([f['t'] for f in frames])))<.11
            start('Falcon · '+title)
            for f in frames:
                f=dict(f,local=f['k']);im=render(f,ci,profile,names=['Дальний человек','Средняя дистанция','Ближняя зона'],
                  chapter='01 / FALCON_TEST.ZST / РЕАЛЬНАЯ ЗАПИСЬ',title='Реальный человек. Сохранённый вывод модели.',source_label='FALCON / ПОЕЗД НЕПОДВИЖЕН')
                zoom(im,f);d=ImageDraw.Draw(im)
                if 'dust' not in f:
                    d.rectangle((1485,766,1850,824),fill='#070d15');text(d,(1490,770),'не записано в клип',24,MUTED)
                note='В этом участке восстановлены 13 кадров СТОП в парной проверке.' if ci==1 else f'Runtime {runtime} · автотраектория — proxy, не независимая разметка.'
                text(d,(105,924),note,19,MUTED)
                first=summary['first_detection']
                if first['k']<=f['k']<first['k']+30:
                    text(d,(105,775),f"ПЕРВОЕ СОВПАДЕНИЕ: {first['truth_x']:.2f} м · t={first['t']:.1f} с",28,GREEN,True)
                    if f['k']==first['k']:
                        im.save(out.parent/'falcon_first_detection.png');samples.append('falcon_first_detection.png')
                put(im,2)
            close(dict(file='falcon_test.zst',type='real',first_scan=frames[0]['k'],last_scan=frames[-1]['k'],scans=len(frames)))
        old_frames=json.loads((quality/'baseline/falcon_clip_1.json').read_text());new_frames=json.loads((quality/'candidate/falcon_clip_1.json').read_text())
        old=next(f for f in old_frames if f['k']==2255);new=next(f for f in new_frames if f['k']==2255)
        assert not old['stop_hit'] and new['stop_hit']
        shared=np.unique(np.vstack([np.asarray(old['p']).reshape(-1,3),np.asarray(new['p']).reshape(-1,3)]),axis=0).ravel().tolist()
        comparison=canvas('ПАРНОЕ СРАВНЕНИЕ / ТОТ ЖЕ ВХОДНОЙ КАДР','Кадр 2255: восстановление СТОП.');draw=ImageDraw.Draw(comparison)
        for x,f,label,color in [(70,old,'ДО / 2e2cee7',GOLD),(970,new,'ПОСЛЕ / 404f3c8',GREEN)]:
            shot=render(dict(f,p=shared,local=f['k']),1,profile,names=['Falcon','Средняя дистанция','Ближняя зона'],chapter='РЕАЛЬНЫЙ FALCON / ОДИН КАДР',title='Сохранённый вывод',source_label='FALCON / ТОТ ЖЕ ВХОД')
            sd=ImageDraw.Draw(shot);sd.rectangle((1485,766,1850,824),fill='#070d15');text(sd,(1490,770),'не записано в клип',24,MUTED)
            comparison.paste(shot.resize((880,495),Image.Resampling.LANCZOS),(x,350));text(draw,(x,290),label,30,color,True)
        text(draw,(85,880),'Общая прореженная выборка исходных точек. Решения вычислены на полном входе.',25,MUTED)
        text(draw,(85,925),'Порог неопределённости СТОП не ослаблен. 14/14 условий приёмки пройдены.',25,MUTED)
        comparison.save(out.parent/'spatial_before_after.png');samples.append('spatial_before_after.png')
        start('Кадр 2255: до и после');put(comparison,8*FPS);close('same original Falcon scan 2255, two detector outputs, same visualization point subset')
        first=summary['first_detection'];first_stop=summary['first_stop']
        start('Что означает дальность');put(card('01 / РЕАЛЬНЫЙ FALCON / РЕЗУЛЬТАТ','Дальность — наблюдение, а не гарантия.',[
          f"Первое сопоставление: {first['truth_x']:.2f} м, t={first['t']:.1f} с.",
          f"Первый СТОП: {first_stop['truth_x']:.2f} м, t={first_stop['t']:.1f} с.",
          'Это один движущийся человек в одной записи и автоматическая траектория.',
          'Около 258 м в прежнем запуске — не гарантия для любого препятствия.',
          'Независимая разметка нужна для реальных precision / recall и FP / FN.']),5*FPS);close('automatic trajectory proxy')
        frames=json.loads((dense/'qualification/clip.json').read_text());assert len(frames)==200
        start('Квалификация');put(card('02 / CLOUD_WITH_FAKE_OBJ.ZST','Квалификация: запись с искусственными объектами.',[
          'Источник — общая папка организаторов, SHA-256 проверен в облаке.',
          'Проверена полная запись: 1510 кадров. Ниже фрагмент 3–23 с при 1×.',
          'Проверка совместимости и работы на прежнем наборе.',
          'Классы и истинность каждой рамки не подтверждены независимой разметкой.']),4*FPS)
        for f in frames:
            im=render(f,1,profile,names=['Falcon: реальные','Квалификация','Новые испытания'],chapter='02 / КВАЛИФИКАЦИЯ / ИСКУССТВЕННЫЕ ОБЪЕКТЫ',title='cloud_with_fake_obj.zst',source_label='КВАЛИФИКАЦИЯ / HESAI');put(im,2)
        im.save(out.parent/'qualification_scene.png');samples.append('qualification_scene.png');close(dict(file='cloud_with_fake_obj.zst',type='qualification with artificial objects',scans_shown=200,scans_evaluated=1510))
        start('Новые записи: происхождение');put(card('03 / NEW_SYNTH_DATA.ZST','Синтетические записи организаторов.',[
          'Платформа → стрелка → тоннель. Все 1667 сканов, исходные 10 Гц.',
          'Без дополнительных инъекций с нашей стороны.',
          'Цель: другая сцена и сенсор, устойчивость пути, фильтр и производительность.',
          'Рассеяние — информационный флаг; круги-мишени для него убраны.',
          'Без меток классов/пыли число выходов не равно числу верных объектов.']),5*FPS);close('organizer synthetic source explanation')
        organizer=dense/'tunnelguard_organizer.mp4';assert organizer.exists()
        parts.append(organizer.resolve());organizer_meta=json.loads((dense/'cab_video.json').read_text());subchapters=[dict(name=v['name'],video_start_s=timeline+v['video_start_s'],duration_s=v['video_end_s']-v['video_start_s'],scans=v['frames']) for v in organizer_meta['scenes']];manifest['chapters'].append(dict(name='Организаторы: платформа, стрелка, тоннель',video_start_s=timeline,duration_s=166.7,subchapters=subchapters,source='new_synth_data.zst, organizer synthetic, all 1667 scans'));timeline+=166.7
        start('Инъекции: цель');put(card('04 / НАШИ СИНТЕТИЧЕСКИЕ ИНЪЕКЦИИ','Меняем цель и дальность. Сохраняем реальный фон.',[
          '17 дистанций: 60, 90, 100, 120, 150, 175, 180, 190, 200,',
          '215, 220, 230, 250, 260, 270, 280, 300 м.',
          'Человек, ящики 0,5 и 0,3 м, модель пылевого аэрозоля.',
          'Цель: диагностировать дальность, задержку, пропуски и нежелательный СТОП.',
          '30 кадров прогрева + 50 с инъекцией; показываем последние 20 при 1×.',
          'Один seed; совпадение дальности ±4 м. Это диагностика, не реальная точность.']),7*FPS);close('synthetic diagnostic protocol, not real recording')
        for kind,kind_name in KINDS.items():
            start('Инъекции · '+kind_name)
            for row in [r for r in ladder if r['object']==kind]:
                d=row['distance'];clip=json.loads((quality/f'candidate/injection_clips/{kind}_{d}.json').read_text());assert len(clip)==20
                for f in clip:
                    im=render(f,1,profile,names=['Реальный фон','Наша инъекция','Метрики испытания'],chapter='04 / СИНТЕТИЧЕСКАЯ ИНЪЕКЦИЯ / НЕ РЕАЛЬНЫЙ ОБЪЕКТ',title=f'{kind_name} · {d} м',source_label='РЕАЛЬНЫЙ FALCON + ИНЪЕКЦИЯ')
                    draw=ImageDraw.Draw(im);draw.rectangle((1475,307,1850,850),fill='#070d15')
                    values=[('Дальность инъекции',f'{d} м'),('Ложное препятствие' if kind=='dust' else 'Детекция / 50 кадров',f"{row['detected']*100:.0f} %"),('Ложный СТОП' if kind=='dust' else 'СТОП / 50 кадров',f"{row['stop']*100:.0f} %"),('Возвраты цели / кадр',f"{row['returns_per_frame']:.1f}")]
                    for j,(label,value) in enumerate(values):
                        y=320+j*128;text(draw,(1490,y),label,19,MUTED);text(draw,(1490,y+32),value,34,INK,True)
                    draw.rectangle((65,919,1460,1018),fill='#070d15')
                    delay='не было' if row['first_frame'] is None else f"{row['first_frame']*.1:.1f} с"
                    text(draw,(105,933),f'Последние 20 из 50 кадров. Первое совпадение: {delay}. Метрики включают подтверждение.',19,MUTED)
                    text(draw,(105,974),'Решения фона и инъекции различаются; в реальном фоне может присутствовать другой человек.',18,MUTED)
                    put(im,2)
                if d in [60,150,300]:
                    name=f'injection_{kind}_{d}.png';im.save(out.parent/name);samples.append(name)
            close(dict(type='our synthetic injection in real Falcon background',object=kind,distances=[r['distance'] for r in ladder if r['object']==kind],frames_shown_per_condition=20,frames_scored_per_condition=50))
        regression=json.loads((quality/'holdout/holdout.json').read_text())
        trials=[r for r in regression['trials'] if r['object']=='person' and r['distance']==300]
        assert len(trials)==3
        n=sum(r['n'] for r in trials);hits=sum(r['hit'] for r in trials);stops=sum(r['stop'] for r in trials)
        start('Дополнительный регрессионный набор');put(card('05 / ДВА РАЗНЫХ ПРОТОКОЛА','68 диагностических и 255 регрессионных условий.',[
          'В предыдущих фрагментах — один seed и одно окно реального фона на дальность.',
          'Дополнительно проверены три seed и разные фоновые окна: всего 255 условий.',
          f'Человек на 300 м в регрессионном наборе: {hits}/{n} предупреждений, {stops}/{n} СТОП.',
          'Разные фоны и seed дают разные доли. Это не ошибка округления.',
          'Оба синтетических протокола не заменяют отдельный размеченный маршрут.']),6*FPS);close('255 previously frozen regression conditions; different backgrounds from diagnostic ladder')
        start('Ограничения и воспроизведение');put(card('ИТОГ / ПРОВЕРЯЕМЫЙ ПРОТОТИП','Честные ограничения. Воспроизводимые проверки.',[
          'Синтетика дополняет реальные проверки, а не заменяет их.',
          'Ложные и пропущенные реальные объекты требуют независимых меток.',
          'Дальние малые ящики и крутые повороты остаются сложными случаями.',
          'Документация, JSON, тесты, Docker/RViz и исходники — в репозитории.',
          'github.com/EhimenNathan/tunnelguard-lct2026']),6*FPS);close('limitations and reproduction')
        listing=scratch/'concat.txt';listing.write_text(''.join("file '"+str(p.resolve()).replace('\\','/')+"'\n" for p in parts),encoding='utf-8')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','warning','-f','concat','-safe','0','-i',str(listing),'-an','-c:v','libx264','-crf','19','-maxrate','1700k','-bufsize','3400k','-preset','fast','-threads','2','-pix_fmt','yuv420p','-r',str(FPS),'-movflags','+faststart',str(out)],check=True)
    assert out.stat().st_size<95*1024**2
    manifest.update(duration_s=round(timeline,3),output_bytes=out.stat().st_size,output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),samples=samples,
      generator_sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in ['finals_comprehensive_video.py','finals_cab_video.py','finals_video.py']})
    out.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Comprehensive video saved',manifest['duration_s'],manifest['output_bytes'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality',required=True);p.add_argument('--dense',required=True);p.add_argument('--web',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.quality,a.dense,a.web,a.out)
