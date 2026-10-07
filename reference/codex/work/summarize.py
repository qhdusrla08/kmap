import sys,json,ast,shutil
from pathlib import Path
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw,ImageFont
O=Path('work');R=Path('outputs');R.mkdir(exist_ok=True)
data=json.loads((O/'xray_labeled.json').read_text(encoding='utf-8'));sp=json.loads((O/'split.json').read_text());pr=json.loads((O/'robustness_hgb_predictions.json').read_text());ob=pd.read_csv(O/'xray_objects_unique.csv');test=[z for z in data.values() if sp[z['stem']]=='test']
tree=ast.parse((O/'experiments.py').read_text(encoding='utf-8'));funcs=[t for t in tree.body if isinstance(t,ast.FunctionDef) and t.name in ['iou','eval_pred']];exec(compile(ast.Module(body=funcs,type_ignores=[]),'<metrics>','exec'))
p={k:(np.array(v['boxes']),np.array(v['scores'])) for k,v in pr['inpaint'].items()};_,hits=eval_pred(p,test,.05)
t=ob[ob.stem.isin([z['stem'] for z in test])].copy();t['hit']=[int(k in hits[s]) for s,k in zip(t.stem,t.k)];t['date']=t.stem.str.split('_').str[1];t['prefix']=t.stem.str.split('_').str[0]
tr=ob[ob.stem.map(sp)=='train'];small=float(tr.area.quantile(.25));contrast=float(tr.clean_contrast.quantile(.25));t['small']=t.area<=small+1e-6;t['low_contrast']=t.clean_contrast<=contrast;t['marker_overlap']=t.marker_pixels>0
coh=[]
for col in ['small','low_contrast','marker_overlap','prefix']:
 for key,g in t.groupby(col):coh.append({'factor':col,'level':str(key),'n':len(g),'TP':int(g.hit.sum()),'FN':int((1-g.hit).sum()),'recall':float(g.hit.mean())})
bydate=t.groupby('date').agg(n=('hit','size'),tp=('hit','sum'));rng=np.random.default_rng(42);draws=rng.integers(0,len(bydate),(10000,len(bydate)));n=bydate.n.to_numpy();tp=bydate.tp.to_numpy();boot=tp[draws].sum(1)/n[draws].sum(1)
sumry={'cohorts':coh,'cutoffs_from_train':{'small_area_lte':small,'low_contrast_lte':contrast},'date_cluster_bootstrap_recall_CI95':np.quantile(boot,[.025,.975]).tolist(),'FN_records':t[t.hit==0].to_dict('records'),'date_stats':bydate.reset_index().to_dict('records')}
dec=pd.DataFrame(json.loads((O/'robustness_hgb_results.json').read_text())['decoy_object_overlap']);sumry['product_decoy_touched_objects']=int((dec.product_decoy_overlap>0).sum())
# AP at stricter IoU: apply the same score ordering and one-to-one rule.
def ap_at(thr):
 rows=[];used={z['stem']:set() for z in test};ng=sum(len(z['boxes']) for z in test)
 for s,(b,sc) in p.items():
  for box,score in zip(b,sc):rows.append((float(score),s,box))
 rows.sort(key=lambda z:-z[0]);flags=[]
 for sc,s,b in rows:
  ov=iou([b],data[s]['boxes'])[0];ok=-1
  for k in np.argsort(-ov):
   if ov[k]>=thr and k not in used[s]:ok=k;break
  flags.append(ok>=0)
  if ok>=0:used[s].add(ok)
 tp=np.cumsum(flags);prec=tp/np.arange(1,len(tp)+1);rec=tp/ng;mr=np.r_[0,rec,1];mp=np.maximum.accumulate(np.r_[0,prec,0][::-1])[::-1]
 return float(np.sum(np.diff(mr)*mp[1:]))
sumry['AP_by_IoU']={f'{q:.2f}':ap_at(q) for q in np.arange(.5,.951,.05)};sumry['mean_AP50_95_custom']=float(np.mean(list(sumry['AP_by_IoU'].values())))
(O/'final_summary.json').write_text(json.dumps(sumry,ensure_ascii=False,indent=2),encoding='utf-8');t.to_csv(O/'hgb_test_objects.csv',index=False)
# Plot from actual results.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
res=json.loads((O/'xray_results.json').read_text())['experiments'];rows=[x for x in res if x['train_variant']==x['test_variant']]
fig,ax=plt.subplots(figsize=(11,4.8));ys=np.arange(len(rows));ax.barh(ys-.18,[x['AP50']*100 for x in rows],height=.32,label='AP50',color='#4379a8');ax.barh(ys+.18,[x['recall']*100 for x in rows],height=.32,label='고정 임계값 재현율',color='#dd944b');ax.set_yticks(ys,[x['model']+' / '+x['train_variant'] for x in rows]);ax.set_xlim(0,112);ax.set_xlabel('%');ax.set_title('X-ray: 순위 성능과 실제 판정 성능은 다를 수 있음');ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
for j,x in enumerate(rows):ax.text(x['AP50']*100+.7,j-.18,f"{x['AP50']*100:.1f}",va='center',fontsize=9);ax.text(x['recall']*100+.7,j+.18,f"{x['recall']*100:.1f}",va='center',fontsize=9)
fig.tight_layout();fig.savefig(R/'model_comparison.png',dpi=160);plt.close(fig)
rr=json.loads((O/'robustness_hgb_results.json').read_text())['experiments'];rr=[x for x in rr if x['seed']==42];names={'inpaint':'기본 인페인팅','dilation2':'제거 범위 2 px','dilation3':'제거 범위 3 px','product_decoy':'제품 내부 가짜 제거 흔적','erase_marker_interior':'표식 내부까지 지우기','median3':'3×3 중앙값 필터'}
fig,ax=plt.subplots(figsize=(11,4.6));v=[x['recall']*100 for x in rr];ax.barh(range(len(rr)),v,color=['#4379a8']*4+['#bc5851']*2);ax.set_yticks(range(len(rr)),[names[x['variant']] for x in rr]);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('고정 임계값 재현율 (%)');ax.set_title('HGB 견고성 실험: 미세한 내부 신호 손상에 취약');ax.spines[['top','right']].set_visible(False)
for j,x in enumerate(rr):ax.text(v[j]+1,j,f"{v[j]:.1f}% ({x['TP']}/133)",va='center')
fig.tight_layout();fig.savefig(R/'robustness.png',dpi=160);plt.close(fig)
shutil.copy(O/'preprocessing_examples.png',R/'preprocessing_examples.png')
print(json.dumps(sumry,ensure_ascii=False,indent=2),flush=True)
