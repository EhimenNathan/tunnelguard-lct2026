"""Публикационные русские графики текущего этапа из проверяемых JSON.
python tools/finals_current_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent.parent
D=R/'docs/finals';O=R/'web/assets';O.mkdir(exist_ok=True)
BG='#09121e';INK='#e9f1f4';MUTED='#9bb0bd';CYAN='#6cdeeb';RED='#ff7379';GOLD='#efbf7c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':MUTED,
 'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#304657','savefig.facecolor':BG})
def axes_style(ax):
 ax.set_facecolor(BG);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.13);ax.set_axisbelow(True)
def finish(fig,name):
 fig.savefig(O/name,dpi=220);fig.savefig(D/name,dpi=220);plt.close(fig)
def main():
 ladder=json.loads((D/'ladder.json').read_text(encoding='utf-8'))
 fig,aa=plt.subplots(2,2,figsize=(15,10),facecolor=BG)
 titles=[('person','Человек 0,5 × 0,3 × 1,75 м'),('box_0.5','Ящик 0,5 м'),('box_0.3','Ящик 0,3 м'),('dust','Пыль: желателен нулевой ложный СТОП')]
 for ax,(kind,title) in zip(aa.flat,titles):
  rows=[r for r in ladder if r['object']==kind];x=[r['distance'] for r in rows]
  ax.plot(x,[100*r['detected'] for r in rows],'-o',color=CYAN,label='Есть препятствие',lw=2.3,ms=4)
  ax.plot(x,[100*r['stop'] for r in rows],'-s',color=RED,label='СТОП',lw=2,ms=4)
  axes_style(ax);ax.set_title(title,loc='left',fontweight='bold',pad=13,color=INK)
  ax.set_ylim(-3,106);ax.set_xlim(55,305);ax.set_xticks([60,100,150,180,200,230,260,280,300]);ax.set_xlabel('Дистанция инъекции, м');ax.set_ylabel('Доля 50 кадров, %')
  if kind=='dust':ax.text(175,45,'0 / 850 кадров СТОП\n17 дистанций',ha='center',color=CYAN,fontsize=16,fontweight='bold')
  ax.legend(loc='lower left' if kind=='person' else 'upper right',frameon=False,labelcolor=INK,fontsize=10)
 fig.suptitle('ЛЕСТНИЦА ДАЛЬНОСТЕЙ · ТЕКУЩИЙ FALCON',x=.07,ha='left',y=.97,fontsize=20,fontweight='bold')
 fig.text(.07,.92,'Реальный фон + синтетические лучевые инъекции · все 68 условий, включая неудачи',color=MUTED,fontsize=12)
 fig.text(.07,.025,'30 кадров прогрева + 50 с объектом; один seed на дистанцию. Доли включают задержку подтверждения.\nЭмпирическая модель возвратов, не независимый тест и не гарантированная дальность. Пыль — помеха, а не целевое препятствие.',fontsize=10,color=MUTED)
 fig.subplots_adjust(left=.07,right=.97,top=.86,bottom=.12,hspace=.40,wspace=.24);finish(fig,'distance_ladder_current.png')
 ab=json.loads((D/'ablation_current.json').read_text(encoding='utf-8'))
 names={'full':'Полная система','no_far':'Без дальней памяти','no_mount':'Без калибровки монтажа','no_ml':'Без ML-экспертов','no_scatter':'Без фильтра рассеяния'}
 fig=plt.figure(figsize=(15,9),facecolor=BG);gs=fig.add_gridspec(2,2,width_ratios=[1.7,1],left=.24,right=.94,top=.82,bottom=.15,hspace=.8,wspace=.45)
 ax=fig.add_subplot(gs[:,0]);y=np.arange(len(ab));axes_style(ax)
 ax.barh(y-.18,[100*r['hit_rate'] for r in ab],height=.31,color=CYAN,label='Обнаружение')
 ax.barh(y+.18,[100*r['stop_rate'] for r in ab],height=.31,color=RED,label='СТОП')
 ax.set_yticks(y,[names[r['variant']] for r in ab]);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('Сопоставленные кадры, %');ax.set_title('Дальний человек · кадры 0–300',loc='left',pad=20,fontweight='bold')
 for i,r in enumerate(ab):
  for v,yy in [(r['hit_rate'],i-.18),(r['stop_rate'],i+.18)]:ax.text(100*v+1,yy,f'{100*v:.1f}',va='center',fontsize=10)
 ax.legend(loc='lower right',frameon=False,labelcolor=INK)
 ax2=fig.add_subplot(gs[0,1]);axes_style(ax2)
 ax2.bar([0,1],[0,5],color=[MUTED,CYAN],width=.55);ax2.set_xticks([0,1],['Без удержания','Удержание 10 кадров'],rotation=15,ha='right');ax2.set_ylim(0,6);ax2.set_yticks([0,1,2,3,4,5]);ax2.set_ylabel('СТОП в провале, из 5');ax2.set_title('Провод · потеря точек на 0,5 с',loc='left',fontsize=11,pad=16,fontweight='bold')
 ax2.text(0,.3,'0/5',ha='center');ax2.text(1,5.2,'5/5',ha='center',color=CYAN)
 ax3=fig.add_subplot(gs[1,1]);axes_style(ax3)
 values=[353/387*100,382/387*100];ax3.bar([0,1],values,color=[MUTED,CYAN],width=.55);ax3.set_xticks([0,1],['Память от 60 м','Память от 30 м'],rotation=15,ha='right');ax3.set_ylim(0,110);ax3.set_ylabel('СТОП, %');ax3.set_title('Реальный человек · 0–50 м',loc='left',fontsize=11,pad=15,fontweight='bold')
 for i,v in enumerate(values):ax3.text(i,v+2,f'{v:.2f}%',ha='center',fontsize=10)
 fig.suptitle('АБЛЯЦИЯ · ТЕКУЩИЙ ЭТАП',x=.07,ha='left',y=.97,fontsize=20,fontweight='bold')
 fig.text(.07,.91,'Три отдельных контролируемых эксперимента · одинаковые настройки внутри каждого сравнения',color=MUTED,fontsize=11)
 fig.text(.07,.035,'Слева: 301 кадр Falcon, автоматический эталон; не независимая оценка FP/FN.\nСправа: синтетический провод и абляция перекрытия дальней памяти на полном реальном прогоне. Источники: docs/finals/*.json.',fontsize=10,color=MUTED)
 finish(fig,'ablation_current.png')
 print('Созданы два графика текущего этапа',flush=True)
if __name__=='__main__':main()
