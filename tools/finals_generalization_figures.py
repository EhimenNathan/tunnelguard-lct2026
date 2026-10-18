"""Rebuild Russian figures from checked-in generalization results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'docs/finals/generalization'
BG, INK, MUTED = '#10243a', '#e9f1f4', '#b7c9d7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,
                    'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
                    'axes.edgecolor':'#506579'})


def main():
    rows = json.loads((D/'candidate/holdout.json').read_text())['trials']
    old = json.loads((D/'baseline/holdout.json').read_text())['trials']
    fig, axes = plt.subplots(1,3,figsize=(18,7),facecolor=BG)
    fig.subplots_adjust(left=.055,right=.98,top=.76,bottom=.23,wspace=.25)
    fig.text(.055,.94,'Новые seed и фоновые окна: где видим цель, а где теряем',fontsize=23,weight='bold')
    fig.text(.055,.86,'255 условий · 17 дистанций · 3 seed · 30 кадров прогрева + 50 с инъекцией',color=MUTED,fontsize=14)
    for ax,kind,title in zip(axes,['person','box_0.5','box_0.3'],['Человек','Ящик 0,5 м','Ящик 0,3 м']):
        ds=sorted({r['distance'] for r in rows})
        def values(rr,metric):
            return np.array([[100*r[metric]/r['n'] for r in rr if r['object']==kind and r['distance']==d] for d in ds])
        h,s,b=values(rows,'hit'),values(rows,'stop'),values(old,'hit')
        ax.fill_between(ds,h.min(1),h.max(1),color='#21e6a7',alpha=.18)
        ax.plot(ds,h.mean(1),color='#21e6a7',marker='o',ms=4,lw=2,label='Есть детекция: новая')
        ax.plot(ds,b.mean(1),color='#ad91ff',ls='--',lw=2,label='Есть детекция: f2af938')
        ax.plot(ds,s.mean(1),color='#ff8d80',lw=2,label='СТОП: новая')
        ax.set_title(title,loc='left',weight='bold',pad=14);ax.set_ylim(-3,103)
        ax.set_xticks([60,100,150,200,250,300]);ax.set_xlabel('Дистанция, м');ax.set_ylabel('Доля кадров, %')
        ax.set_facecolor(BG);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,labelcolor=INK,fontsize=10,loc='lower left')
    fig.text(.055,.075,'Полоса — минимум/максимум трёх испытаний, не доверительный интервал. Среднее: n=150 кадров на дистанцию.\nПыль: 0/2550 ложных СТОП в ±4 м от инъекции. Человек вне пути: 0/2550 предупреждений и СТОП.\nОбщие реальные фоны; это не реальная precision/recall.',color=MUTED,fontsize=11,linespacing=1.5)
    for path in [D/'holdout_distance.png',ROOT/'web/assets/holdout_distance.png']:
        fig.savefig(path,dpi=200,facecolor=BG)
    plt.close(fig)
    source=ROOT/'docs/finals/dense_cloud/accepted'
    if source.exists():
        names=['synthetic_platform_cloud_only','synthetic_switch_cloud_only','synthetic_tunnels_cloud_only']
        reports=[json.loads((source/n/'live_check.json').read_text()) for n in names]
        assert all(r['passed'] and r['received']==r['expected'] for r in reports)
        fig,ax=plt.subplots(figsize=(14,7),facecolor=BG)
        fig.subplots_adjust(left=.08,right=.96,top=.74,bottom=.23)
        fig.text(.08,.94,'Три полных записи. 1667 из 1667 сообщений.',fontsize=24,weight='bold')
        fig.text(.08,.85,'ROS 2 + RViz / noVNC · 4 vCPU AMD EPYC 9V74 · Linux amd64 · без потерь и с чистым закрытием',color=MUTED,fontsize=12)
        x=np.arange(3)
        for offset,key,color,label in [(-.17,'processing_ms_p95','#21e6a7','p95 обработки до публикации'),(.17,'callback_ms_p95','#ad91ff','p95 полного callback')]:
            vals=[r[key] for r in reports];ax.bar(x+offset,vals,.30,color=color,label=label)
            for xx,y in zip(x+offset,vals):ax.text(xx,y+2,f'{y:.1f}',ha='center')
        ax.axhline(100,color='#ffc247',ls='--',lw=1.5,label='Бюджет 10 Гц: 100 мс')
        ax.set_ylim(0,115);ax.set_ylabel('Время, мс')
        ax.set_xticks(x,[f'{n}\n{r["received"]}/{r["expected"]} · {r["output_hz"]:.3f} Гц' for n,r in zip(['Платформа','Стрелка','Тоннель'],reports)])
        ax.set_facecolor(BG);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
        ax.legend(frameon=False,labelcolor=INK,loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,fontsize=10)
        fig.text(.08,.08,'Полные записи 38,5 / 73,5 / 54,4 с. p99 интервала выдачи: 127,6 / 131,4 / 128,2 мс.\nСредняя частота и p95 не являются WCET: равномерность каждого периода и hard real time не доказаны.',fontsize=11,color=MUTED,linespacing=1.6)
        for path in [source/'full_ros_performance.png',ROOT/'web/assets/full_ros_performance.png']:
            fig.savefig(path,dpi=200,facecolor=BG)
        plt.close(fig)


if __name__ == '__main__':
    main()
