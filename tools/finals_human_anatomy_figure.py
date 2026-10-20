from pathlib import Path
import sys,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'tools'))
from human_anatomy_inject import anatomy,inject_human
from falcon_inject import sensor_to_level,FLOOR
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#EDF4FC','axes.labelcolor':'#A9BED2','xtick.color':'#A9BED2','ytick.color':'#A9BED2','font.size':13})
fig,axes=plt.subplots(1,4,figsize=(15.5,6),facecolor='#091725',gridspec_kw={'width_ratios':[1.2,1,1,1]})
for i,(ax,d) in enumerate(zip(axes,[0,60,150,300])):
 ax.set_facecolor('#091725');ax.set_xlim(-.46,.46);ax.set_ylim(-.03,1.85);ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Боковая координата, м')
 for _,c,rad in anatomy(float(d),0,0):
  ax.add_patch(Ellipse((c[1],c[2]-FLOOR),2*rad[1],2*rad[2],facecolor='#1B5B6B' if i==0 else 'none',edgecolor='#69C6C6' if i==0 else '#425E73',lw=1.4,alpha=.8))
 if i:
  _,_,points=inject_human(np.array([[1,10,0]],np.float32),np.zeros(1,np.float32),float(d),0,np.random.default_rng(88000+d));q=sensor_to_level(points);ax.scatter(q[:,1],q[:,2]-FLOOR,s=22,c='#43DCC4');ax.set_title(f'{d} м\n{len(q)} возвратов в этом кадре',color='#EDF4FC',pad=18,fontsize=14)
 else:ax.set_title('Эталон тела\n1,75 м высотой',color='#EDF4FC',pad=18,fontsize=14);ax.set_ylabel('Высота над основанием, м')
fig.text(.05,.985,'АНАТОМИЧЕСКАЯ МОДЕЛЬ И ДИСКРЕТНЫЕ ЛИДАРНЫЕ ВОЗВРАТЫ',fontsize=17,fontweight='bold',va='top')
fig.text(.05,.025,'Контур — известная форма инъекции. Детектор получает только точки. Один кадр, заданный seed; эмпирическая лучевая модель Falcon.',fontsize=11,color='#A9BED2')
fig.subplots_adjust(left=.055,right=.985,bottom=.16,top=.79,wspace=.55);dest=r/'web/assets/human_anatomy.png';fig.savefig(dest,dpi=180,facecolor=fig.get_facecolor());plt.close(fig);print(dest)
