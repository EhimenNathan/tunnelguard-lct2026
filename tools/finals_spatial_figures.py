"""Russian comparison of the spatial candidate against the current frozen runtime."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/finals/spatial/quality'
BG,INK,MUTED='#10243a','#e9f1f4','#b7c9d7'

def main():
    a,b=[json.loads((D/name/'summary.json').read_text()) for name in ['baseline','candidate']]
    gate=json.loads((D/'spatial_gates.json').read_text());assert gate['passed']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#506579'})
    fig,axes=plt.subplots(1,3,figsize=(18,7),facecolor=BG,gridspec_kw={'width_ratios':[1.05,1.05,1.1]})
    fig.subplots_adjust(left=.055,right=.98,top=.74,bottom=.31,wspace=.4)
    fig.text(.055,.94,'Пространственное объединение и приоритет достоверного СТОП',fontsize=23,weight='bold')
    fig.text(.055,.85,'Одинаковые 4443 кадра Falcon · native C++ / Numba · веса модели неизменны · исходные JSON доступны',fontsize=13,color=MUTED)
    for ax in axes:
        ax.set_facecolor(BG);ax.spines[['top','right']].set_visible(False)
    band=lambda s,lo:next(v for v in s['bands'] if v['lo']==lo)
    vals=[band(s,200)['stop'] for s in [a,b]];n=band(a,200)['n']
    axes[0].barh([1,0],100*np.array(vals)/n,color=['#ad91ff','#21e6a7'],height=.48)
    axes[0].set_yticks([1,0],['До','После']);axes[0].set_xlim(0,112);axes[0].set_xlabel('Доля кадров, %');axes[0].set_title('СТОП · человек на 200–250 м',loc='left',weight='bold',pad=18)
    for y,v in zip([1,0],vals):axes[0].text(100*v/n-2,y,f'{v}/{n}',ha='right',va='center',color=BG,weight='bold',fontsize=15)
    axes[0].text(.02,-.21,f'+{vals[1]-vals[0]} сопоставленных кадров СТОП\nЭто не число новых объектов.',transform=axes[0].transAxes,color=MUTED,fontsize=11)
    for offset,source,color,label in [(-.18,a,'#ad91ff','До'),(.18,b,'#21e6a7','После')]:
        axes[1].bar(np.arange(2)+offset,[100*band(source,250)[k]/band(source,250)['n'] for k in ['hit','stop']],width=.32,color=color,label=label)
    axes[1].set_xticks([0,1],['Детекция','СТОП']);axes[1].set_ylim(0,100);axes[1].set_ylabel('Доля кадров, %');axes[1].set_title('250–320 м · без ухудшения',loc='left',weight='bold',pad=18);axes[1].legend(frameon=False,labelcolor=INK,loc='upper right',fontsize=10)
    axes[1].text(.02,-.21,'602/825 детекций · 404/825 СТОП\nДальняя зона всё ещё имеет пропуски.',transform=axes[1].transAxes,color=MUTED,fontsize=11)
    axes[2].set_axis_off();axes[2].text(0,.92,'Проверки перед принятием',weight='bold',fontsize=16);axes[2].text(0,.68,f"{sum(gate['checks'].values())}/{len(gate['checks'])}",fontsize=48,color='#21e6a7',weight='bold');axes[2].text(0,.51,'Все зафиксированные условия пройдены.',fontsize=11,color=MUTED)
    lines=['255 синтетических регрессионных условий','Ни один положительный случай не ухудшен','Пыль: 0/2550 нежелательных СТОП (±4 м)','Вне пути: 0/2550 предупреждений/СТОП','Реальные FP/FN без меток неизвестны']
    for i,line in enumerate(lines):axes[2].text(0,.37-i*.105,line,fontsize=11,color=MUTED)
    fig.text(.055,.035,'Реальные совпадения — по автоматической траектории, не независимым меткам.\n10 несопоставленных кадров СТОП сохранены: конец записи, локальная диагностика воспроизводит 1 с удержания при отсутствии свежих возвратов.\nДанные проверялись ранее: 255 условий являются регрессией, а не новым независимым holdout.',fontsize=10.5,color=MUTED,linespacing=1.5)
    for path in [ROOT/'web/assets/spatial_comparison.png',D.parent/'spatial_comparison.png']:
        fig.savefig(path,dpi=200,facecolor=BG)
    plt.close(fig)
if __name__=='__main__':main()
