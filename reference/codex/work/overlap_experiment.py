import sys,ast,json
from pathlib import Path
import numpy as np,pandas as pd,cv2
from PIL import Image
from scipy.ndimage import maximum_filter
from sklearn.ensemble import HistGradientBoostingClassifier
O=Path('work');data=json.loads((O/'xray_labeled.json').read_text(encoding='utf-8'))
tree=ast.parse((O/'experiments.py').read_text(encoding='utf-8'));exec(compile(ast.Module(body=[t for t in tree.body if isinstance(t,ast.FunctionDef) and t.name in ['variants','iou','candidates','eval_pred','nms']],type_ignores=[]),'<functions>','exec'))
items=sorted(data.values(),key=lambda z:(z['date'],z['stem']));train=[z for z in items if z['date'] not in ['20200623','20200624']];val=[z for z in items if z['date']=='20200623'];test=[z for z in items if z['date']=='20200624'];xx=[];yy=[];cache={}
for i,z in enumerate(items):
 a=np.array(Image.open(z['path']).convert('RGB'));vs,mask=variants(a,i+42);b,f,s=candidates(vs['inpaint'])
 if z in train:
  ov=iou(b,z['boxes']).max(1);keep=(ov>=.5)|(ov<.1);xx.append(f[keep]);yy.append((ov[keep]>=.5).astype(int))
 else:cache[z['stem']]=(b,f)
xx=np.concatenate(xx);yy=np.concatenate(yy);m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1,random_state=42).fit(xx,yy)
vp={z['stem']:nms(cache[z['stem']][0],m.predict_proba(cache[z['stem']][1])[:,1]) for z in val};grid=np.linspace(.05,.95,19);score=[eval_pred(vp,val,t)[0]['F1'] for t in grid];th=float(grid[np.argmax(score)])
p={z['stem']:nms(cache[z['stem']][0],m.predict_proba(cache[z['stem']][1])[:,1]) for z in test};met,hits=eval_pred(p,test,th);met.update(AP50=eval_pred(p,test)[0]['AP50'],threshold=th)
ob=pd.read_csv(O/'xray_objects_unique.csv');ob=ob[ob.stem.isin(p)].copy();ob['hit']=[k in hits[s] for s,k in zip(ob.stem,ob.k)];ob['overlap']=ob.marker_pixels>0
ob['proposal_hit']=[bool(iou(cache[s][0],[data[s]['boxes'][k]]).max()>=.5) for s,k in zip(ob.stem,ob.k)]
ob['ideal_fixed_box_iou']=[min(10,w)*min(10,h)/(100+w*h-min(10,w)*min(10,h)) for w,h in zip(ob.w,ob.h)]
ob['center_hit']=[bool(any(sc>=th and box[0]<=(b[0]+b[2])/2<=box[2] and box[1]<=(b[1]+b[3])/2<=box[3] for b,sc in zip(*p[s]))) for s,k in zip(ob.stem,ob.k) for box in [data[s]['boxes'][k]]]
cohort=ob.groupby('overlap').agg(n=('hit','size'),TP=('hit','sum'),recall=('hit','mean'),proposal_hits=('proposal_hit','sum'),center_hits=('center_hit','sum'),median_area=('area','median')).reset_index().to_dict('records')
result={'purpose':'supplementary held-out-date interpolation, not prospective chronological evaluation','train_images':len(train),'val_images':len(val),'test_images':len(test),'train_dates':sorted(set(z['date'] for z in train)),'val_date':'20200623','test_date':'20200624','metrics':met,'cohorts':cohort}
(O/'overlap_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');ob.to_csv(O/'overlap_test_objects.csv',index=False);print(json.dumps(result,ensure_ascii=False,indent=2))
