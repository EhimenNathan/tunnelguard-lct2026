import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import fsSync from 'node:fs';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..').replaceAll('\\','/');
const DIR=process.env.TG_DECK_BUILD_DIR || ROOT+'/outputs/presentation_build';
const p=await PresentationFile.importPptx(await FileBlob.load(ROOT+'/presentation/TunnelGuard_LCT2026.pptx'));
p.slides.keep(Array.from({length:11},(_,i)=>i));
p.view.hideGuides();p.view.hideGridlines();p.view.hidePlaceholders();
const inspect=await p.inspect({select:'$.slides[*].elements[?(@.kind == "textbox")]',include:['id','kind','text']});
const replacements=[
 ['Мы не распознаём классы объектов','Восстанавливаем путь и габарит по облаку точек. Гибрид физики, ансамблей LightGBM/CatBoost и временного подтверждения обнаруживает посторонние объекты и передаёт СТОП / ВНИМАНИЕ в ROS 2.'],
 ['Лидар сам восстанавливает путь','Разделяем измеренный путь и прогноз с неопределённостью. C++-ядро и временная память сохраняют слабые дальние цели. На полном Falcon восстановлены 13 кадров СТОП без ухудшения зафиксированных отрицательных контролей.'],
 ['ROS 2 Humble + Docker; вход','ROS 2 Humble, Docker и RViz в браузере. Вход: PointCloud2 с адаптацией осей и монтажа. Геометрия пути и габарита, C++-ядро, два ансамбля LightGBM/CatBoost, временное подтверждение. Falcon: первое совпадение человека 256,56 м по автоматической траектории. Полные новые записи: 1667/1667 сообщений, около 10 Гц в облаке.'],
 ['«Второй машинист»','Прототип раннего предупреждения для системы беспилотного поезда. Выдаёт расстояние, габарит, неопределённость и решения СТОП / ВНИМАНИЕ. Запуск: закреплённый Docker-образ, noVNC и RViz. Дальнейшее развитие: проверка по маркерам организаторов, повышение чувствительности к малым целям, сложные кривые и испытания на целевом ПК.'],
 ['Рельсы, сечение тоннеля, кривизна', 'Рельсы и сечение тоннеля задают ось пути. Сенсорная адаптация оценивает ориентацию и монтаж. Временная память стабилизирует редкие дальние отражения.'],
 ['У каждой точки пути есть', 'Неопределённость σ(x) ограничивает доверие к пути. Измеренная геометрия отделена от экстраполяции. СТОП требует подтверждённой цели и достоверного положения внутри габарита.'],
 ['На 200 м', 'На 200 м'],
 ['Человек на 160 м', 'Исторический пример слева относится к квалификации. В новых движущихся тестах на кривых ±500 м остаются пропуски. Видимость ограничивается стенами тоннеля, а дальняя экстраполяция сохраняет растущую неопределённость.'],
 ['v1: ⅓', 'Два специалиста: объекты у пола и подвешенные объекты. В каждом ½ LightGBM + ½ CatBoost в пространстве логитов. PI-MLP относится к исторической версии.'],
 ['38 физических признаков кластера.', '38 физических признаков. Обучение использует кандидатов и лучевые инъекции. Новые организаторские маркеры проверяются отдельно. Пыль выводится информационно.'],
 ['Обучение на 4 тоннелях', 'Квалификация: leave-one-tunnel-out. Финал: полный Falcon, 68 +255 фиксированных условий и 51 анатомический тест. Реальные классовые FP/FN не заменены синтетическими долями.'],
 ['Запуск: ', 'Запуск: закреплённый Docker-образ, RViz через noVNC и ros2 bag play. Ядро: C++ и NumPy/Numba. 46 native-тестов и проверка воспроизводимости через Compose.'],
 ['23 ГБ записей', 'Потоковый разбор архивов и облачные проверки обошли ограничение локального диска. Пространственное слияние восстановило 13 кадров СТОП на Falcon. Сложные кривые и малые цели на 300 м остаются границами прототипа; для них добавлены отдельные диагностические тесты.'],
 ['Метро перевозит миллионы','Выбрали задачу обнаружения препятствий в тоннеле: темнота, кривые, стрелки и редкие дальние отражения требуют объяснимой геометрии и проверяемой оценки риска. Цель команды: воспроизводимый прототип с понятными границами применимости.']
];
for(const rec of inspect.records??[]){if(rec.kind!=='textbox'||!rec.text)continue;for(const [prefix,value] of replacements){if(rec.text.startsWith(prefix)){p.resolve(rec.id).text.replace(rec.text,value);console.log('Updated mandatory textbox',rec.slideIndex,rec.name);}}}
const C={bg:'#091725',ink:'#F1F6FC',muted:'#A9BED2',accent:'#43DCC4',red:'#FF6F8D',gold:'#F8C76C'};const FONT='Montserrat';
function text(s,content,x,y,w,h,size=28,color=C.ink,bold=false){const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=content;t.text.style={typeface:FONT,fontSize:size,color,bold,verticalAlignment:'top',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};return t;}
function slide(title,source=''){const s=p.slides.add();s.background.fill=C.bg;text(s,title,60,42,1160,110,44,C.ink,true);text(s,'СахалИИн  /  TunnelGuard  /  финал ЛЦТ 2026',60,679,1000,24,16,C.muted);s.speakerNotes.text=source;return s;}
function image(s,path,x=60,y=154,w=1160,h=470){return s.images.add({blob:new Uint8Array(fsSync.readFileSync(path)),contentType:'image/png',alt:path.split('/').at(-1),fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function table(s,values,x,y,w,h,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});for(let i=0;i<values.length;i++)for(let j=0;j<values[0].length;j++){const c=t.getCell(i,j);c.fill=i===0?'#18354B':C.bg;c.text.style={typeface:FONT,fontSize:i===0?23:22,color:i===0?C.accent:C.ink,bold:i===0};}t.borders.assign({fill:'#2E4A60',width:1});return t;}
function link(s,label,url,y){const t=text(s,'',60,y,1160,45,28,C.accent);t.text=[[{run:label,style:{typeface:FONT,fontSize:'28px',color:C.accent,underline:'sng'},link:{uri:url,isExternal:true}}]];}
for(const [id,value] of [
 ['sh/u9sja98z','Проверка геометрии на кривых\nИсторический пример слева относится к квалификации. В движущихся тестах на кривых ±500 м остаются пропуски. Видимость ограничивают стены тоннеля, дальняя экстраполяция сохраняет растущую неопределённость.'],
 ['sh/k3uhcb6h','Физические признаки\n38 признаков кластера. Обучение использует кандидатов и лучевые инъекции. Маркеры организаторов проверяются отдельно. Пыль выводится информационно.'],
 ['sh/y1czalob','Проверка обобщения\nКвалификация: leave-one-tunnel-out. Финал: полный Falcon, 68 +255 фиксированных условий и 51 анатомический тест. Синтетика не заменяет реальные классовые FP/FN.'],
 ['sh/4zyhofml','Два специалиста\nОбъекты у пола и подвешенные объекты. В каждом ½ LightGBM + ½ CatBoost в пространстве логитов. Временная память дополняет покадровую оценку. PI-MLP относится к исторической версии.']
]){const t=p.resolve(id);t.text.replace(t.text.toString(),value);}
// Source slides 7--11 retain the qualification order and master/layout.
const picInspect=await p.inspect({select:'$.slides[7,10].elements[?(@.kind == "image")]',include:['id','kind']});
for(const im of picInspect.records??[])p.delete(im.id);
function qtext(s,v,x,y,w,h,size=17,color='#26232B',bold=false){return text(s,v,x,y,w,h,size,color,bold);}
function qbox(s,title,body,x,y,w,h){s.shapes.add({geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill:'#FFFFFF',line:{fill:'#560080',width:2}});qtext(s,title,x+9,y+12,w-18,50,14,'#4E0875',true);qtext(s,body,x+9,y+70,w-18,h-72,12,'#676477');}
function arrow(s,x,y,w=24){s.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:12},fill:'#FF0066',line:{fill:'none',width:0}});}
// Retain original cards but separate title/body styling after replacing text.
p.delete(['sh/u9sja98z','sh/k3uhcb6h','sh/y1czalob','sh/4zyhofml']);
const geoSlide=p.slides.items[8];qtext(geoSlide,'Проверка геометрии на кривых',60,506,760,30,18,'#560080',true);qtext(geoSlide,'Исторический пример слева относится к квалификации. В движущихся тестах на кривых ±500 м остаются пропуски. Видимость ограничивают стены тоннеля, дальняя экстраполяция сохраняет растущую неопределённость.',60,545,760,105,16);
const mlCards=p.slides.items[10];
for(const [x,title,body] of [[57,'Физические признаки','38 признаков кластера. Обучение использует кандидатов и лучевые инъекции. Маркеры организаторов проверяются отдельно. Пыль выводится информационно.'],[465,'Проверка обобщения','Квалификация: leave-one-tunnel-out. Финал: полный Falcon, 68 +255 условий и 51 анатомический тест. Синтетика не заменяет реальные классовые FP/FN.'],[873,'Два специалиста','Объекты у пола и подвешенные объекты. В каждом ½ LightGBM +½ CatBoost в логитах. Временная память дополняет покадровую оценку. PI-MLP относится к старой версии.']]){qtext(mlCards,title,x,513,350,30,18,'#560080',true);qtext(mlCards,body,x,547,350,105,14);}
const arch=p.slides.items[7];
qtext(arch,'Docker, Ubuntu 22.04, ROS 2 Humble. Геометрия C++, NumPy/Numba',155,163,940,24,16,'#560080');
qbox(arch,'Облако точек','ROS 2 bag\nили живой лидар\nFalcon / Hesai',152,278,118,150);arrow(arch,271,342);
qbox(arch,'Подготовка','PointCloud2\nОси и монтаж\nПлотность лучей',297,278,118,150);arrow(arch,416,342);
qbox(arch,'Геометрия пути','Рельсы и их непрерывность\n\nШаблон сечения\n\nГлобальный DP\n\nГаусс–Ньютон\n\nНеопределённость σ(x)',442,205,215,315);arrow(arch,658,262);
qbox(arch,'Габарит','Измерения\nЭкстраполяция\nЗоны с учётом σ',687,205,135,140);
qbox(arch,'Кандидаты','Кластеры\nФизические тесты\nПыль отдельно',687,380,135,140);arrow(arch,823,435);
qbox(arch,'Два эксперта','У пола / подвес\n½ LightGBM\n½ CatBoost',852,380,135,140);
qbox(arch,'Трек и память','Треки\nДальняя память\nСвежие возвраты',852,205,135,140);arrow(arch,988,262);
qbox(arch,'Решение','СВОБОДНО\nВНИМАНИЕ\nСТОП\n\nROS 2 и RViz',1017,205,125,315);
for(const [x,y,geometry] of [[747,348,'downArrow'],[917,348,'upArrow']])arch.shapes.add({geometry,position:{left:x,top:y,width:12,height:29},fill:'#FF0066',line:{fill:'none',width:0}});
qtext(arch,'Кандидаты проходят оценку и временное подтверждение. Рассеяние не запускает СТОП.',155,541,990,30,15,'#560080');
arch.speakerNotes.text='Текущий алгоритм: src/tunnel_guard/tunnel_guard/core/{geometry,detector,farfield,scorer}.py. Нативная редактируемая схема в композиции квалификационного слайда 8. Схема не включает непроверенные экспериментальные изменения.';
const ml=p.slides.items[10];
qtext(ml,'Физические признаки и два специалиста',75,145,1120,35,26,'#560080',true);
qbox(ml,'Кластер и физика','38 признаков: форма, опора, оболочка, угол и плотность возвратов',90,220,230,205);arrow(ml,325,315,36);
qbox(ml,'Объект у пола','½ LightGBM + ½ CatBoost\n\nСреднее логитов, JSON-модели',370,190,280,140);
qbox(ml,'Подвешенный объект','½ LightGBM + ½ CatBoost\n\nОтдельный эксперт для подвеса',370,350,280,125);
arrow(ml,660,315,36);qbox(ml,'Решение и память','Геометрия и свежие возвраты\n\nСлияние по пространству\n\nВременная память и трек\n\nПылевой флаг отдельно',710,190,460,285);
ml.speakerNotes.text='Текущие модели в config/obstacle_scorer.json и obstacle_scorer_hanging.json. Веса 1/2 +1/2 в пространстве логитов. Старый PI-MLP не входит в текущую поставку. Числа AP квалификации удалены из этой текущей схемы; исторические результаты доступны в репозитории.';
let s=slide('Реальный Falcon: первое совпадение человека',ROOT+'/docs/finals/spatial/quality/candidate/summary.json\nhttps://github.com/EhimenNathan/tunnelguard-lct2026/actions/runs/39197603363\nАвтоматическая траектория, 4443 кадра. 256,56 м при t=4,5 с. Одна цель; не гарантированная дальность любого объекта.');
image(s,ROOT+'/web/assets/falcon_first_detection.png',40,140,1200,485);text(s,'256,56 м по автоматической траектории. Это наблюдение одной цели',60,630,1160,34,22,C.muted);
s=slide('Устойчивость СТОП после пространственного исправления',ROOT+'/docs/finals/spatial/quality/spatial_gates.json\n'+ROOT+'/docs/finals/spatial/stop_gaps.json\n14/14 заранее заданных условий. Два изменения вместе; не изолированная абляция. Реальные FP/FN без независимого полного эталона неизвестны.');
s.charts.add('bar',{position:{left:70,top:175,width:650,height:390},categories:['СТОП на 200–250 м'],series:[{name:'До',values:[705],fill:'#7791A8'},{name:'После',values:[718],fill:C.accent}],barOptions:{direction:'column',grouping:'clustered'},hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:22,fill:C.ink}},chartFill:C.bg,plotAreaFill:C.bg,yAxis:{min:0,max:750,textStyle:{typeface:FONT,fontSize:19,fill:C.muted}},xAxis:{textStyle:{typeface:FONT,fontSize:22,fill:C.ink}},dataLabels:{showValue:false,position:'outEnd',textStyle:{typeface:FONT,fontSize:28,fill:C.ink,bold:true}}});
text(s,'705',285,175,100,48,30,C.ink,true);text(s,'718',490,169,100,48,30,C.ink,true);
text(s,'13 кадров восстановлены',770,185,440,100,39,C.accent,true);text(s,'24 → 11\nкадров без СТОП\n\n1,7 → 1,0 с\nмаксимальный провал',770,300,430,285,29,C.ink);text(s,'Знаменатель: 729 кадров. Детекции и отрицательные контроли сохранены',60,603,1160,58,23,C.muted);
s=slide('Данные квалификации и финала',ROOT+'/docs/data_sources.json\n'+ROOT+'/docs/finals/spatial/dense/qualification/summary.json\nОрганизаторы предоставили 20.10 маркеры и obstacles.md; это отдельный новый источник проверки.');
table(s,[['Источник','Тип','Проверенные кадры'],['falcon_test.zst','Реальная запись: поезд и человек','4443'],['cloud_with_fake_obj.zst','Квалификация с искусственными объектами','1510'],['new_synth_data.zst','Синтетика организаторов: три сцены','1667'],['Наши инъекции','17 дальностей, известное положение цели','68 + 255 условий']],60,175,1160,340,[370,570,220]);
text(s,'Архив квалификации восстановлен и проверен по SHA-256. Новые маркеры организаторов получены отдельно от наших инъекций',60,550,1160,100,26,C.muted);
s=slide('Полные новые записи: около 10 Гц в ROS и RViz',ROOT+'/docs/finals/spatial/dense/\nhttps://github.com/EhimenNathan/tunnelguard-lct2026/actions/runs/39200634492\n4 vCPU AMD EPYC 7763, Linux amd64. p99 интервала выше 100мс; hard real time не доказан.');
table(s,[['Сцена','Получено','Частота','p95 callback'],['Платформа','386/386','9,9918 Гц','62,97 мс'],['Стрелка','736/736','9,9976 Гц','56,54 мс'],['Тоннель','545/545','10,0004 Гц','57,53 мс']],60,175,1160,320,[340,220,270,330]);
text(s,'C++-геометрия, 46 native-тестов. Короткий Falcon: три раза по 301/301 после ожидания DDS ACK последнего скана',60,535,1160,82,27,C.ink);text(s,'Средняя частота не гарантирует каждый период 100 мс. На другом ПК требуется измерение',60,623,1160,42,21,C.muted);
s=slide('Проверка дальности на реальном фоне',ROOT+'/web/data/finals_ladder.json\n30 кадров прогрева и 50 оцениваемых кадров, один seed на дальность. Сопоставление по продольной дистанции ±4м. Прежний person — прямоугольный прокси 0.5×0.3×1.75м; не анатомическая модель.');
const ladder=JSON.parse(await fs.readFile(ROOT+'/web/data/finals_ladder.json','utf8')).filter(r=>r.object==='person');
s.charts.add('line',{position:{left:60,top:178,width:1160,height:385},categories:ladder.map(r=>String(r.distance)),series:[{name:'Обнаружение',values:ladder.map(r=>Math.round(100*r.detected)),fill:C.accent},{name:'СТОП',values:ladder.map(r=>Math.round(100*r.stop)),fill:C.gold}],hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:20,fill:C.ink}},chartFill:C.bg,plotAreaFill:C.bg,lineOptions:{smooth:false},xAxis:{textStyle:{typeface:FONT,fontSize:17,fill:C.muted}},yAxis:{min:0,max:100,textStyle:{typeface:FONT,fontSize:18,fill:C.muted}}});
text(s,'Человек-прокси: доля из 50 кадров, %. 17 расстояний от 60 до 300 м',60,575,1160,42,26,C.ink);text(s,'Диагностика чувствительности и пропусков. Общие реальные фоны ограничивают независимость оценки',60,624,1160,45,21,C.muted);
s=slide('Анатомический человек: внутри и снаружи габарита',ROOT+'/tools/human_anatomy_inject.py\n'+ROOT+'/tools/finals_human_anatomy_eval.py\nНовый диагностический протокол, Windows NumPy/Numba. 51 условие завершено: 30 прогрева +50 оцениваемых, один seed на дальность. Сопоставление по дальности±4м и боковой координате<0.8м. Вне пути0/1700; человек300м8/50предупреждений,0/50СТОП.');
image(s,ROOT+'/web/assets/human_anatomy.png',60,158,1160,418);text(s,'60–215 м: 48/50 детекций. 250 м: 47/50. 300 м: 8/50, СТОП 0/50. Вне пути: 0/1700 предупреждений и СТОП',60,592,1160,70,24,C.muted);
s=slide('Новые маркеры и сценарии организаторов','https://disk.yandex.ru/d/SfBeRHbs8bNJAQ\nobstacles.md, SHA-256 db62201d049da4f16063abceab8a918f7091335e41a093fcb9211863df687558\nmarker-eval 39342143068, все1667кадров. 32135записеймаркеровADD/DELETE. Маркеры не содержат текстовых классов.');
table(s,[['Запись','Состав последовательности из obstacles.md'],['Платформа','Пыль, малые кубы справа и слева, человек'],['Стрелка','Малые кубы, наклонный подвесной кабель, четыре облака пыли'],['Тоннель','Малые кубы, куб 2×2×2 м, балка поперёк рельсов, пыль']],60,170,1160,330,[300,860]);
text(s,'1667 кадров обработаны с замороженным native-детектором. Экспортированы положение, ориентация и размеры маркеров',60,535,1160,85,27,C.ink);text(s,'Семантика классов и ожидаемая реакция проверяются отдельно от геометрического совпадения',60,626,1160,40,21,C.muted);
s=slide('Пыль и человек вне пути: отрицательные контроли',ROOT+'/docs/finals/spatial/quality/holdout/holdout.json\nПылевая инъекция: нежелательный СТОП в ±4м по дальности. Человек вне пути: продольная±4м и боковая<0.8м. Счётчики относятся к известной инъекции, не ко всем неизвестным фоновым объектам.');
text(s,'0 / 2550',60,190,500,100,68,C.accent,true);text(s,'нежелательных СТОП\nу пылевой инъекции',60,305,500,100,30,C.ink);text(s,'0 / 2550',690,190,500,100,68,C.accent,true);text(s,'предупреждений и СТОП\nу человека вне пути',690,305,500,100,30,C.ink);text(s,'Флаг рассеяния информационный. Пыль не должна запускать торможение',60,470,1160,90,33,C.ink,true);text(s,'Эти нули относятся к зафиксированной синтетике. Новые сцены организаторов требуют сопоставления с их маркерами',60,591,1160,70,23,C.muted);
s=slide('Фильтр провода и устойчивость дальних целей',ROOT+'/docs/finals/spatial/wire_regression.json\n'+ROOT+'/docs/SPATIAL_REVIEW_20261019_RU.md');
text(s,'21 кадр',60,180,500,100,60,C.accent,true);text(s,'памяти предупреждений',60,285,500,60,29,C.ink);text(s,'15 кадров',690,180,500,100,60,C.gold,true);text(s,'окно свежих доказательств СТОП',690,285,500,80,29,C.ink);text(s,'Подтверждённый трек удерживается до 10 кадров. Синтетический провал 0,5 с: 0/5 → 5/5 кадров СТОП',60,410,1160,130,32,C.ink);text(s,'Точный исходный реальный провод требует отдельного подтверждения. Новая сцена содержит наклонный кабель с маркером',60,573,1160,85,23,C.muted);
s=slide('Границы и следующие технические улучшения',ROOT+'/docs/SPATIAL_REVIEW_20261019_RU.md\n'+ROOT+'/docs/finals/spatial/moving_curves.json\nСинтетическая движущаяся платформа5м/с, цель~120м,30warm+50scored. Прямая/±80050/50,±50039/50и36/50детекций.');
const next=[['Малые дальние цели','Накопление с компенсацией движения и неопределённости. Проверка нескольких поз, отражаемостей и фаз скана'],['Кривые и стрелки','Несколько гипотез пути, связь пары рельсов, проверка смены кривизны и ограничение экстраполяции'],['Проверка ошибок','Геометрические маркеры, порядок из obstacles.md, отрицательные сцены и раздельные показатели по типу цели']];next.forEach((a,i)=>{text(s,a[0],60,167+i*151,1160,46,32,C.accent,true);text(s,a[1],60,220+i*151,1160,90,27,C.ink);});
s=slide('Запуск на компьютере экспертов',ROOT+'/docs/RUN_ON_PC_RU.md\n'+ROOT+'/compose.repro.yaml\nhttps://github.com/EhimenNathan/tunnelguard-lct2026/actions/runs/39341798129');
text(s,'Docker + Compose, Linux containers\nГотовый образ по SHA-256 digest\nRViz в браузере через noVNC\nПауза, скорость, шаг и смена бэга',60,169,1160,210,32,C.ink);text(s,'docker compose -f compose.repro.yaml pull\ndocker compose -f compose.repro.yaml up -d\ndocker compose -f compose.repro.yaml exec tunnelguard bash',60,410,1160,150,23,C.accent);text(s,'Внутри контейнера: tg play /data/имя_бэга 1.0 --start-paused. CI: 301/301 ROS/RViz, 12/12 операций управления. Подробная инструкция — в README',60,585,1160,75,23,C.muted);
s=slide('Демонстрация и воспроизводимые результаты','https://ehimennathan.github.io/tunnelguard-lct2026/\nhttps://github.com/EhimenNathan/tunnelguard-lct2026\nПолное видео7:20: реальныйFalcon,квалификация,триорганизаторскиесцены,68инъекций. Все показанные сканы1×.');
text(s,'СахалИИн',60,163,1160,90,66,C.ink,true);link(s,'Интерактивный сайт и видео 7:20','https://ehimennathan.github.io/tunnelguard-lct2026/#film',300);link(s,'Код, README и исходные результаты','https://github.com/EhimenNathan/tunnelguard-lct2026',385);link(s,'Версия квалификационного этапа','https://ehimennathan.github.io/tunnelguard-lct2026/qualification.html',470);text(s,'Геометрия объясняет решение. Тесты показывают его границы',60,575,1160,85,33,C.accent,true);
await fs.mkdir(DIR+'/renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(DIR+`/renders/slide_${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));console.log('Render',i+1);}
await (await PresentationFile.exportPptx(p)).save(DIR+'/candidate.pptx');
await fs.writeFile(DIR+'/authored.json',JSON.stringify({slides:p.slides.items.length,mandatory:[1,2,3,4,5],nativeTables:[14,15,18],nativeCharts:[13,16],sourceTemplate:ROOT+'/../ЛЦТ2026 Шаблон презентации.pptx'},null,2));
console.log('Candidate saved',p.slides.items.length);



