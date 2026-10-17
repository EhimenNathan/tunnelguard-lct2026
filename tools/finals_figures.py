import json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent.parent
out=r/'web/assets';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#e9f1f4','axes.labelcolor':'#9bb0bd',
                     'xtick.color':'#9bb0bd','ytick.color':'#cfdae0','font.size':11})
variants=[('full','Полная система'),('no_shell_veto','Без вето оболочки'),('no_shape','Без угловых размеров'),
          ('no_temporal_geometry','Без временной геометрии'),('no_sigma_zones','Без зон неопределённости'),
          ('no_ml','Без обученного ансамбля'),('straight_corridor','Только прямой коридор')]
counts=[json.loads((r/f'tools/results/ablation_{k}.json').read_text())['fp_frames'] for k,_ in variants]
fig,ax=plt.subplots(figsize=(11,6.4),facecolor='#09121e');ax.set_facecolor('#09121e')
ax.barh(range(len(variants)),counts,color=['#6cdeeb']+['#58788f']*5+['#e5a86e'],height=.58)
ax.invert_yaxis();ax.set_yticks(range(len(variants)),[v for _,v in variants]);ax.set_xscale('log');ax.set_xlim(1,800)
ax.set_xlabel('Кадры ложного СТОП · логарифмическая шкала',labelpad=15)
for i,n in enumerate(counts):ax.text(n*1.08,i,f'{n} / 2 287',va='center',fontsize=10,color='#e9f1f4')
for sp in ax.spines.values():sp.set_visible(False)
ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=12)
fig.text(.04,.94,'ЗАЧЕМ НУЖНА ГЕОМЕТРИЯ И ВРЕМЯ',fontsize=17,weight='bold')
fig.text(.04,.885,'Сохранённая абляция отборочной версии · Pandar · сентябрь 2026',fontsize=10,color='#9bb0bd')
fig.text(.04,.04,'Исторические результаты; не повторный тест текущей Falcon-версии.\nОбученная часть использовала эти записи: сравнение диагностическое.',fontsize=9,color='#9bb0bd')
fig.subplots_adjust(left=.34,right=.90,top=.81,bottom=.20);fig.savefig(out/'ablation.png',dpi=170);plt.close(fig)

print('Сохранена историческая абляция: web/assets/ablation.png')
