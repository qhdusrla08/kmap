import sys,ast,json
from pathlib import Path
import numpy as np,pandas as pd,cv2,joblib
from PIL import Image
from scipy.ndimage import maximum_filter
from sklearn.ensemble import RandomForestClassifier
O=Path('work');data=json.loads((O/'xray_labeled.json').read_text(encoding='utf-8'));sp=json.loads((O/'split.json').read_text(encoding='utf-8'))
tree=ast.parse((O/'experiments.py').read_text(encoding='utf-8'));funcs=[t for t in tree.body if isinstance(t,ast.FunctionDef) and t.name in ['variants','iou','candidates','eval_pred','nms']];exec(compile(ast.Module(body=funcs,type_ignores=[]),'<experiment_functions>','exec'))
items=sorted(data.values(),key=lambda z:(z['date'],z['stem']));train=[z for z in items if sp[z['stem']]=='train'];val=[z for z in items if sp[z['stem']]=='val'];test=[z for z in items if sp[z['stem']]=='test'];xx=[];yy=[];cache={};meta=[]
for i,z in enumerate(items):
 a=np.array(Image.open(z['path']).convert('RGB'));vs,mask=variants(a,i+42)
 if sp[z['stem']]=='train':
  b,f,s=candidates(vs['inpaint']);ov=iou(b,z['boxes']).max(1);keep=(ov>=.5)|(ov<.1);xx.append(f[keep]);yy.append((ov[keep]>=.5).astype(int));continue
 variants_extra={'inpaint':vs['inpaint']}
 if sp[z['stem']]=='test':
  gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);base=(a.max(2).astype(int)-a.min(2).astype(int)>40).astype(np.uint8)
  for rad in [2,3]:variants_extra['dilation'+str(rad)]=cv2.inpaint(gray,cv2.dilate(base,np.ones((2*rad+1,2*rad+1),np.uint8)),3,cv2.INPAINT_TELEA)
  rng=np.random.default_rng(i+2026);dm=np.zeros_like(mask);ys,xs=np.where(vs['inpaint']<np.median(vs['inpaint'])-30)
  for k in range(3):
   j=int(rng.integers(len(xs)));x,y=int(xs[j]),int(ys[j]);cv2.rectangle(dm,(x-9,y-9),(x+9,y+9),1,3)
  variants_extra['product_decoy']=cv2.inpaint(vs['inpaint'],dm,3,cv2.INPAINT_TELEA)
  null=np.zeros_like(mask)
  for x,y,w,h,area in z['components']:cv2.rectangle(null,(max(0,x-1),max(0,y-1)),(x+w,y+h),1,-1)
  variants_extra['erase_marker_interior']=cv2.inpaint(gray,null,3,cv2.INPAINT_TELEA)
  variants_extra['median3']=cv2.medianBlur(vs['inpaint'],3)
  for j,box in enumerate(z['boxes']):
   x1,y1,x2,y2=np.round(box).astype(int);meta.append({'stem':z['stem'],'object':j,'product_decoy_overlap':int(dm[y1:y2,x1:x2].sum())})
 for name,g in variants_extra.items():
  b,f,s=candidates(g);cache[z['stem'],name]=(b,f)
 if i%100==0:print('robust features',i,flush=True)
xx=np.concatenate(xx);yy=np.concatenate(yy);out=[];saved={}
for seed in [42,84,2026]:
 m=RandomForestClassifier(n_estimators=100,max_depth=16,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=4,random_state=seed).fit(xx,yy)
 p={z['stem']:nms(cache[z['stem'],'inpaint'][0],m.predict_proba(cache[z['stem'],'inpaint'][1])[:,1]) for z in val};grid=np.linspace(.05,.95,19);sc=[eval_pred(p,val,t)[0]['F1'] for t in grid];th=float(grid[np.argmax(sc)])
 names=['inpaint','dilation2','dilation3','product_decoy','erase_marker_interior','median3'] if seed==42 else ['inpaint']
 for name in names:
  pr={z['stem']:nms(cache[z['stem'],name][0],m.predict_proba(cache[z['stem'],name][1])[:,1]) for z in test}
  met,hits=eval_pred(pr,test,th);met.update(AP50=eval_pred(pr,test)[0]['AP50'],seed=seed,variant=name,threshold=th);out.append(met)
  if seed==42:saved[name]={k:{'boxes':b.tolist(),'scores':s.tolist()} for k,(b,s) in pr.items()}
  print(met,flush=True)
 if seed==42:joblib.dump(m,O/'inpaint_rf.joblib')
(O/'robustness_results.json').write_text(json.dumps({'experiments':out,'decoy_object_overlap':meta},indent=2),encoding='utf-8');(O/'robustness_predictions.json').write_text(json.dumps(saved),encoding='utf-8')
print('DONE',flush=True)
