"""Source-backed static Russian publication figures for the improvement release.
python tools/finals_improvement_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent.parent;D=R/'docs/finals/improvement';A=R/'web/assets'
BG='#10243a';INK='#e9f1f4';MUTED='#b7c9d7';OLD='#ad91ff';NEW='#21e6a7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#506579','savefig.facecolor':BG})
def load(p):return json.loads((D/p).read_text(encoding='utf-8'))
def style(ax):
 ax.set_facecolor(BG);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
def finish(fig,name):
 for p in [D/name,A/name]:fig.savefig(p,dpi=220,facecolor=BG)
 plt.close(fig)
def main():
 a,b=[load(f'paired/{v}/summary.json') for v in ['baseline','candidate']]
 la,lb=[load(f'paired/{v}/ladder.json') for v in ['baseline','candidate']]
 fig,axes=plt.subplots(2,2,figsize=(16,11),facecolor=BG);fig.subplots_adjust(left=.07,right=.97,top=.86,bottom=.13,hspace=.48,wspace=.26)
 fig.text(.07,.95,'Больше слабых целей. Меньше времени на кадр.',fontsize=24,weight='bold')
 fig.text(.07,.905,'Финальный этап · парное сравнение с замороженным Docker-образом · 4443 кадра + 68 условий',fontsize=13,color=MUTED)
 ax=axes[0,0]
 for s,c,label,marker in [(a,OLD,'Замороженная версия','o'),(b,NEW,'Обновлённая версия','s')]:
  ax.plot(range(6),[100*r['hit']/r['n'] for r in s['bands']],color=c,label=label,marker=marker,lw=2.5)
 ax.set_xticks(range(6),[f"{r['lo']}–{r['hi']}" for r in a['bands']]);ax.set_ylim(-3,105);ax.set_title('Человек: совпадения по автоэталону',loc='left',weight='bold',pad=15);ax.set_ylabel('Доля кадров, %');ax.set_xlabel('Диапазон, м · n = 387 / 714 / 550 / 407 / 729 / 825');ax.legend(frameon=False,fontsize=10,labelcolor=INK,loc='lower left');style(ax)
 for ax,kind,title in [(axes[0,1],'box_0.5','Инъекция ящика 0,5 м'),(axes[1,0],'box_0.3','Инъекция ящика 0,3 м')]:
  for rows,c,mark in [(la,OLD,'o'),(lb,NEW,'s')]:
   rr=[r for r in rows if r['object']==kind];ax.plot([r['distance'] for r in rr],[100*r['detected'] for r in rr],marker=mark,color=c,lw=2.3,ms=4)
  ax.set_ylim(-3,105);ax.set_xlim(55,305);ax.set_xticks([60,100,150,200,250,300]);ax.set_xlabel('Дистанция, м · 50 кадров на условие');ax.set_ylabel('Есть детекция, %');ax.set_title(title,loc='left',weight='bold',pad=15);style(ax)
 ax=axes[1,1];tim=load('dense_paired.json')['summaries'];x=np.arange(2)
 for i,(key,color,label) in enumerate([('baseline',OLD,'Замороженная'),('candidate',NEW,'Обновлённая')]):
  vals=[tim[key]['p50_ms'],tim[key]['p95_ms']];ax.bar(x+(i-.5)*.32,vals,.30,color=color,label=label)
  for xx,v in zip(x+(i-.5)*.32,vals):ax.text(xx,v+5,f'{v:.1f}',ha='center',fontsize=12,color=INK)
 ax.axhline(100,color=INK,ls='--',lw=1);ax.text(.96,103,'100 мс',ha='right',fontsize=10,color=MUTED);ax.set_ylim(0,max(tim['baseline']['p95_ms'],tim['candidate']['p95_ms'])*1.22);ax.set_xticks(x,['Медиана','p95']);ax.set_ylabel('Время полного ядра, мс');ax.set_title('Плотная синтетика: Windows, парный тест',loc='left',weight='bold',pad=15);style(ax)
 fig.text(.07,.055,'Качество: Linux/native, один seed на дистанцию; автоэталон не является независимой разметкой.\nВремя: одинаковые 90 кадров после 30 прогрева, порядок версий чередуется; без ROS/GUI, не WCET.',color=MUTED,fontsize=11,linespacing=1.6)
 finish(fig,'improvement_comparison.png')
 rows=load('low_ablation.json')['trials'];fig,aa=plt.subplots(1,2,figsize=(16,7),facecolor=BG);fig.subplots_adjust(left=.07,right=.97,top=.76,bottom=.21,wspace=.28)
 fig.text(.07,.93,'Абляция отдельного канала низких препятствий',fontsize=23,weight='bold');fig.text(.07,.85,'Меняется только lowfield.enabled · seeds 1001, 2002, 3003 · 30 + 50 кадров · общий реальный фон',color=MUTED,fontsize=13)
 specs=[('box_0.3',90,'hit'),('box_0.3',100,'hit'),('box_0.3',150,'hit')]
 def bars(ax,specs,labels,title):
  x=np.arange(len(specs))
  for i,(v,col,label) in enumerate([('without_low',OLD,'Без низкого канала'),('current',NEW,'Полная версия')]):
   vals=[]
   for kind,d,metric in specs:
    rr=[r for r in rows if r['object']==kind and r['distance']==d];vals.append(100*sum(r['variants'][v][metric] for r in rr)/sum(r['n'] for r in rr))
   ax.bar(x+(i-.5)*.32,vals,.30,color=col,label=label)
   for xx,y in zip(x+(i-.5)*.32,vals):ax.text(xx,y+2,f'{y:.1f}%',ha='center',fontsize=11,color=INK)
  ax.set_xticks(x,labels);ax.set_ylim(0,110);ax.set_ylabel('Доля кадров, % · n=150 на столбец');ax.set_title(title,loc='left',weight='bold',pad=14);style(ax)
 bars(aa[0],specs,['90 м','100 м','150 м'],'Ящик 0,3 м: есть детекция')
 bars(aa[1],[('person',300,'stop'),('dust',150,'stop')],['Человек 300 м\nСТОП','Пыль 150 м\nЛожный СТОП'],'Контроль: человек и пыль')
 aa[0].legend(frameon=False,fontsize=10,labelcolor=INK,loc='upper left')
 fig.text(.07,.07,'Три новых seed проверяют чувствительность к синтезу, а не независимое обобщение на другие тоннели.\nОдинаковые фоновые фрагменты; доверительные интервалы реальной точности не вычисляются.',fontsize=11,color=MUTED,linespacing=1.6)
 finish(fig,'ablation_low_obstacles.png')
if __name__=='__main__':main()
