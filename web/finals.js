/* Сайт воспроизводит сохранённый вывод. Браузер не исполняет детектор. */
(() => {
'use strict';
const $=s=>document.querySelector(s), fmt=(v,n=1)=>v==null?'—':Number(v).toFixed(n), pct=v=>v==null?'—':(100*v).toFixed(1)+' %';
const get=async p=>{const r=await fetch(p);if(!r.ok)throw Error('Не удалось загрузить '+p);return r.json();};
let frames=[],index=0,playing=false,acc=0,previous=performance.now(),loadId=0;
const host=$('#viewport'), renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setClearColor(0x09121e);renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.prepend(renderer.domElement);
const scene=new THREE.Scene();scene.fog=new THREE.FogExp2(0x09121e,.0015);
const camera=new THREE.PerspectiveCamera(55,1,.1,1800), controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=.12;controls.maxDistance=650;controls.minDistance=.5;
const cloud=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:.075,color:0x99b3c4,sizeAttenuation:true,transparent:true,opacity:.8}));
scene.add(cloud);
const pathGroup=new THREE.Group(),boxes=new THREE.Group();scene.add(pathGroup,boxes);
const grid=new THREE.GridHelper(800,80,0x233c50,0x112333);grid.position.set(0,-2.1,-200);scene.add(grid);
function clearGroup(group){while(group.children.length){const c=group.children[0];group.remove(c);if(c.geometry)c.geometry.dispose();if(c.material)c.material.dispose();}}
function line(points,color,opacity=1){const geom=new THREE.BufferGeometry().setFromPoints(points);const l=new THREE.Line(geom,new THREE.LineBasicMaterial({color,transparent:true,opacity}));pathGroup.add(l);}
function view(mode){const f=frames[index],d=f?.truth_x||f?.boxes[0]?.d||150,l=0.35;
  const setups={cab:[[0,1.3,9],[.3,0,-100]],overview:[[35,35,18],[0,0,-100]],top:[[.3,180,-160],[.3,0,-160]],target:[[l+7,4,-d+12],[l,-.9,-d]]};
  const [pos,target]=setups[mode]||setups.cab;camera.position.set(...pos);controls.target.set(...target);controls.update();}
function draw(){const f=frames[index];if(!f)return;
 const xyz=new Float32Array(f.p.length);for(let j=0;j<f.p.length;j+=3){xyz[j]=f.p[j+1];xyz[j+1]=f.p[j+2];xyz[j+2]=-f.p[j];}
 const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.BufferAttribute(xyz,3));cloud.geometry.dispose();cloud.geometry=geom;
 clearGroup(pathGroup);clearGroup(boxes);
 const profile=[[-1.45,0],[1.45,0],[1.65,1.3],[1.45,3.5],[-1.45,3.5],[-1.65,1.3],[-1.45,0]];
 const rings=f.line.map(([x,y,z])=>profile.map(([l,h])=>new THREE.Vector3(y+l,z+h,-x)));
 rings.forEach((ring,i)=>{if(i%2===0)line(ring,f.line[i][0]<=f.measured?0x6cdeeb:0x576986,.42);});
 for(let j=0;j<6;j++)for(let i=1;i<rings.length;i++)line([rings[i-1][j],rings[i][j]],f.line[i][0]<=f.measured?0x6cdeeb:0x576986,.45);
 if(f.line.length)line(f.line.map(([x,y,z])=>new THREE.Vector3(y,z,-x)),0x9feff5,.8);
 f.boxes.forEach(o=>{const size=o.size.map(v=>Math.max(v,.18)),g=new THREE.BoxGeometry(size[1],size[2],size[0]);const edges=new THREE.EdgesGeometry(g);g.dispose();const box=new THREE.LineSegments(edges,new THREE.LineBasicMaterial({color:o.zone===2?0xff7379:0xf1c37d}));box.position.set(o.c[1],o.c[2],-o.c[0]);boxes.add(box);});
 $('.view-label').textContent=f.organizer_synth?f.dataset_label+' / HESAI':f.synthetic?'СИНТЕТИКА / ЧЕЛОВЕК 200 М + ПЫЛЬ 120 М / РЕАЛЬНЫЙ ФОН':'FALCON / СТАТИЧНЫЙ ПОЕЗД / РЕАЛЬНЫЕ ДАННЫЕ';
 const states=['СВОБОДНО','ВНИМАНИЕ','СТОП'];$('#state').textContent=states[f.level];$('#state').style.color=['#6cdeeb','#f1c37d','#ff7379'][f.level];
 $('#frame-meta').textContent=`Кадр ${f.k} · ${f.boxes.length} препятствий · прореженное облако ${xyz.length/3} точек`;
 $('#target-d').textContent=f.truth_x==null?'не размечено':fmt(f.truth_x)+' м';$('#env-d').textContent=fmt(f.env,0)+' м';$('#measured-d').textContent=fmt(f.measured,0)+' м';$('#clear-d').textContent=fmt(f.clear,0)+' м';$('#frame-ms').textContent=fmt(f.ms)+' мс';$('#time').textContent=fmt(f.t)+' с';$('#timeline').value=index;
}
async function loadClip(){const id=++loadId;playing=false;$('#play').textContent='▶ Смотреть';$('#loading').style.display='grid';$('#loading').textContent='Загружаем реальные кадры…';try{const idScene=Number($('#scene').value);const path=idScene<4?`data/falcon_clip_${idScene}.json`:`data/new_synth_${['platform','switch','tunnels'][idScene-4]}.json`;const data=await get(path);if(id!==loadId)return;frames=data;index=idScene===0?Math.min(18,frames.length-1):(idScene>=4?Math.min(2,frames.length-1):0);if(idScene>=4)$('#camera').value='overview';$('#timeline').max=frames.length-1;draw();view($('#camera').value);$('#loading').style.display='none';}catch(e){$('#loading').textContent=e.message;}}
$('#scene').addEventListener('change',loadClip);$('#camera').addEventListener('change',()=>view($('#camera').value));
$('#play').addEventListener('click',()=>{playing=!playing;$('#play').textContent=playing?'Ⅱ Пауза':'▶ Смотреть';});
$('#timeline').addEventListener('input',e=>{index=Number(e.target.value);draw();});
new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(host);
function tick(now){const dt=Math.min(now-previous,100);previous=now;if(playing&&frames.length){acc+=dt*Number($('#speed').value);if(acc>=500){acc%=500;index=(index+1)%frames.length;draw();}}controls.update();renderer.render(scene,camera);requestAnimationFrame(tick);}requestAnimationFrame(tick);loadClip();

let dustRows=[];get('data/dust_zones.json').then(x=>{dustRows=x;lab();}).catch(()=>{});
let ladder=[],object='person';const distances=[60,90,100,120,150,175,180,190,200,215,220,230,250,260,270,280,300], names={person:'Человек', 'box_0.5':'Ящик 0,5 м','box_0.3':'Ящик 0,3 м',dust:'Пылевой аэрозоль'};
function lab(){const i=Number($('#distance').value),d=distances[i],row=ladder.find(r=>r.object===object&&r.distance===d);$('#distance-label').textContent=d;$('#object-title').textContent=names[object]+' · '+d+' м';if(!row)return;
 $('#hit-label').textContent=object==='dust'?'Ложное препятствие':'Обнаружение';$('#stop-label').textContent=object==='dust'?'Ложный СТОП':'СТОП';$('#dust-readout').hidden=object!=='dust';$('#dust-rate').textContent=pct(dustRows.find(x=>x.distance===d)?.dust_zone_rate);
 $('#hit-rate').textContent=pct(row.detected);$('#stop-rate').textContent=pct(row.stop);$('#hit-bar').style.width=(row.detected*100)+'%';$('#stop-bar').style.width=(row.stop*100)+'%';$('#returns').textContent=fmt(row.returns_per_frame);$('#delay').textContent=row.first_frame==null?'не обнаружен':fmt(row.first_frame/10)+' с';
 $('#matrix').replaceChildren(...distances.map((x,j)=>{const r=ladder.find(r=>r.object===object&&r.distance===x),b=document.createElement('button');b.title=x+' м · '+pct(r?.detected);b.setAttribute('aria-label',b.title);b.className=j===i?'selected':'';const fill=document.createElement('i');fill.style.height=((r?.detected||0)*100)+'%';b.append(fill);b.addEventListener('click',()=>{$('#distance').value=j;lab();});return b;}));}
$('#distance').addEventListener('input',lab);document.querySelectorAll('[data-object]').forEach(b=>b.addEventListener('click',()=>{object=b.dataset.object;document.querySelectorAll('[data-object]').forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));lab();}));get('data/finals_ladder.json').then(x=>{ladder=x;lab();}).catch(e=>{$('#object-title').textContent=e.message;});
Promise.all([get('data/finals_summary.json'),get('data/cloud_live.json')]).then(([s,c])=>{
 $('#hero-range').textContent=fmt(s.first_detection?.truth_x,1);$('#hero-env').textContent=fmt(s.envelope_median_m,0);$('#hero-rate').textContent=fmt(c.output_hz,2);
 const p=s.bands.filter(b=>b.n&&b.lo<250),hit=p.reduce((a,b)=>a+b.hit,0),n=p.reduce((a,b)=>a+b.n,0);
 const tests=[['1. Дальность обнаружения',fmt(s.first_detection?.truth_x)+' м','Первое сопоставленное обнаружение человека; СТОП — '+fmt(s.first_stop?.truth_x)+' м. Не максимальная гарантированная дальность.'],['2. Дальность габарита',fmt(s.envelope_median_m,0)+' м · медиана','Измеренная геометрия: медиана '+fmt(s.measured_median_m,0)+' м. Остальная дальность — экстраполяция с неопределённостью.'],['3. Устойчивость габарита',s.geometry_dropouts_after_30+' пропаданий','После 30 кадров прогрева. p95 покадрового смещения оси на 150 м: '+fmt(s.jitter['150'].p95_cm,2)+' см; на 300 м: '+fmt(s.jitter['300'].p95_cm,2)+' см.'],['4. Ложные срабатывания',s.beside_stop+' / '+s.beside+' СТОП на человеке вне пути','Несопоставленных с человеком кадров СТОП: '+s.unmatched_stop_frames+'. Их нельзя автоматически объявлять ложными; нужна проверка сцены.'],['5. Пропуски',pct(hit/n)+' · обнаружение <250 м','Доля по доступной автоматической траектории. Неизвестная траектория: '+s.unknown_reference_frames+' кадров; интерполяция: '+s.interpolated_frames+'. Полные FN неизвестны.'],['6. Работа в реальном времени',fmt(c.output_hz,2)+' Гц · ROS','Живая проверка в облаке: '+c.received+'/'+c.expected+' сообщений; p95 обработки '+fmt(c.processing_ms_p95)+' мс; полного callback '+fmt(c.callback_ms_p95)+' мс. Это измеренный результат, не hard-real-time гарантия.'],['7. Фильтр','17 дистанций × 4 воздействия','Синтетические человек, два ящика и пыль. Итоги каждой дистанции доступны в лаборатории выше. Реальные аэрозоли требуют нового бэга.'],['8. Дополнительные возможности','3D / C++ / адаптация','Оси Falcon/ROS, калибровка монтажа, оценка угловой плотности, дальняя память, треки и удержание объекта.'],['9. Простота запуска','Облачный Docker + RViz','Codespaces/noVNC, пауза и скорость штатного ros2 bag play, tg play для переключения записей без перезапуска контейнера.']];
 $('#score-body').replaceChildren(...tests.map(row=>{const tr=document.createElement('tr');row.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td);});return tr;}));
}).catch(e=>{$('#score-body').textContent=e.message;});
})();
