import sys,json
from pathlib import Path
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from scipy.ndimage import maximum_filter
O=Path('work');R=Path('outputs');R.mkdir(exist_ok=True)
def dump(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
data=json.loads((O/'xray_labeled.json').read_text(encoding='utf-8'));items=sorted(data.values(),key=lambda z:(z['date'],z['stem']))
dates=sorted(set(z['date'] for z in items));train_dates=dates[:5];val_dates=dates[5:14];test_dates=dates[14:]
for z in items:z['split']='train' if z['date'] in train_dates else ('val' if z['date'] in val_dates else 'test')
dump('split.json',{z['stem']:z['split'] for z in items})
def variants(a,seed=0):
 mask=(a.max(2).astype(int)-a.min(2).astype(int)>40).astype(np.uint8)
 # Exact BMP color masks; one-pixel dilation preserves small internal objects as far as possible.
 mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
 gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 inp=cv2.inpaint(gray,mask,3,cv2.INPAINT_TELEA)
 masked=gray.copy();masked[mask>0]=int(np.median(gray))
 rng=np.random.default_rng(seed);dec=np.zeros_like(mask)
 for _ in range(3):
  x=int(rng.integers(25,a.shape[1]-25));y=int(rng.integers(25,a.shape[0]-25));cv2.rectangle(dec,(x-9,y-9),(x+9,y+9),1,3)
 decoy=cv2.inpaint(inp,dec,3,cv2.INPAINT_TELEA)
 # Counterfactual markers are removed at original locations, then moved to arbitrary locations.
 moved=inp.copy();moved[dec>0]=76
 return {'raw':gray,'mask':masked,'inpaint':inp,'decoy':decoy,'moved':moved},mask
def iou(bs,gt):
 bs=np.asarray(bs).reshape(-1,4);gt=np.asarray(gt).reshape(-1,4)
 lo=np.maximum(bs[:,None,:2],gt[None,:,:2]);hi=np.minimum(bs[:,None,2:],gt[None,:,2:]);inter=np.maximum(0,hi-lo).prod(2)
 return inter/np.maximum(1,(bs[:,2:]-bs[:,:2]).prod(1)[:,None]+(gt[:,2:]-gt[:,:2]).prod(1)[None,:]-inter)
def candidates(gray):
 # Image-only proposal generation, never uses labels or color-mask locations.
 bh=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((11,11),np.uint8)).astype(float)
 peak=(bh==maximum_filter(bh,size=5))&(bh>=3)
 peak[:10]=0;peak[-10:]=0;peak[:,:10]=0;peak[:,-10:]=0
 yy,xx=np.where(peak)
 order=np.argsort(-bh[yy,xx],kind='stable')[:800];xx=xx[order];yy=yy[order]
 boxes=np.stack([xx-5,yy-5,xx+5,yy+5],1).astype(float)
 feat=[]
 for x,y in zip(xx,yy):
  p=gray[y-8:y+9,x-8:x+9].astype(float);q=cv2.resize(p,(9,9),interpolation=cv2.INTER_AREA)
  f=np.r_[(q-q.mean()).ravel()/32,q.mean()/255,q.std()/32,bh[y,x]/32,p[7:10,7:10].mean()/255]
  feat.append(f)
 return boxes,np.asarray(feat,np.float32),bh[yy,xx]
cache={};stats=[]
for ii,z in enumerate(items):
 a=np.array(Image.open(z['path']).convert('RGB'));vs,mask=variants(a,ii+42)
 for variant,gray in vs.items():
  b,f,s=candidates(gray);ov=iou(b,z['boxes']);best=ov.max(1) if len(b) else np.array([])
  cache[(z['stem'],variant)]=(b,f,s,best)
  if variant in ['raw','mask','inpaint']:
   stats.append({'stem':z['stem'],'split':z['split'],'variant':variant,'proposals':len(b),'objects':len(z['boxes']),'proposal_hits':int((ov.max(0)>=.5).sum()) if len(b) else 0})
 if ii%100==0:print('features',ii,flush=True)
dump('proposal_stats.json',stats)
# Metrics: confidence ranked, one-to-one IoU matching, AP at 0.5 with monotone PR envelope.
def eval_pred(preds,subset,threshold=0):
 allp=[];ng=sum(len(z['boxes']) for z in subset);matched={z['stem']:set() for z in subset};hits={z['stem']:[] for z in subset}
 for z in subset:
  b,s=preds[z['stem']]
  for box,score in zip(b,s):
   if score>=threshold:allp.append((float(score),z['stem'],box))
 allp.sort(key=lambda p:-p[0]);tp=[]
 for score,stem,b in allp:
  ov=iou([b],data[stem]['boxes'])[0];order=np.argsort(-ov);got=-1
  for k in order:
   if ov[k]>=.5 and int(k) not in matched[stem]:got=int(k);break
  tp.append(got>=0)
  if got>=0:matched[stem].add(got)
 t=np.cumsum(tp);f=np.cumsum(1-np.array(tp,dtype=int));rec=t/max(ng,1);prec=t/np.maximum(1,t+f)
 mrec=np.r_[0,rec,1];mpre=np.r_[0,prec,0];mpre=np.maximum.accumulate(mpre[::-1])[::-1]
 ap=float(np.sum(np.diff(mrec)*mpre[1:]));nt=int(t[-1]) if len(t) else 0;nf=len(tp)-nt
 return {'AP50':ap,'recall':nt/ng,'precision':nt/max(1,nt+nf),'F1':2*nt/max(1,ng+nt+nf),'TP':nt,'FP':nf,'FN':ng-nt,'images':len(subset),'objects':ng,'FP_per_image':nf/len(subset)},matched
def nms(boxes,scores):
 keep=[]
 for i in np.argsort(-scores):
  if scores[i]<.001:continue
  if keep and iou([boxes[i]],boxes[keep]).max()>.2:continue
  keep.append(i)
  if len(keep)>=30:break
 return boxes[keep],scores[keep]
subsets={s:[z for z in items if z['split']==s] for s in ['train','val','test']}
results=[];allpred={};cohorts=[]
for variant in ['raw','mask','inpaint']:
 xx=[];yy=[]
 for z in subsets['train']:
  b,f,s,ov=cache[z['stem'],variant];keep=(ov>=.5)|(ov<.1);xx.append(f[keep]);yy.append((ov[keep]>=.5).astype(int))
 xx=np.concatenate(xx);yy=np.concatenate(yy);print('training',variant,xx.shape,int(yy.sum()),flush=True)
 for name in ['RF','HGB']:
  if variant=='mask' and name=='HGB':continue
  model=(RandomForestClassifier(n_estimators=100,max_depth=16,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=4,random_state=42) if name=='RF' else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1,random_state=42))
  model.fit(xx,yy)
  test_variants=[variant]+(['decoy'] if variant=='inpaint' else [])+(['inpaint','moved'] if variant=='raw' else [])
  pred={}
  for z in subsets['val']:
   b,f,s,ov=cache[z['stem'],variant];pred[z['stem']]=nms(b,model.predict_proba(f)[:,1])
  grid=np.linspace(.05,.95,19);scores=[eval_pred(pred,subsets['val'],t)[0]['F1'] for t in grid];th=float(grid[int(np.argmax(scores))])
  for tv in test_variants:
   p={}
   for z in subsets['test']:
    b,f,s,ov=cache[z['stem'],tv];p[z['stem']]=nms(b,model.predict_proba(f)[:,1])
   ap,_=eval_pred(p,subsets['test']);met,matched=eval_pred(p,subsets['test'],th)
   met.update(AP50=ap['AP50'],model=name,train_variant=variant,test_variant=tv,threshold=th,validation_F1=max(scores));results.append(met)
   allpred[f'{name}_{variant}_{tv}']={stem:{'boxes':b.tolist(),'scores':s.tolist()} for stem,(b,s) in p.items()}
   if name=='RF' and variant=='inpaint' and tv=='inpaint':
    for z in subsets['test']:
     a=np.array(Image.open(z['path']).convert('RGB'));vs,mask=variants(a);gray=vs['inpaint']
     for k,box in enumerate(z['boxes']):
      x1,y1,x2,y2=np.round(box).astype(int);pa=gray[y1:y2,x1:x2];contrast=float(np.median(pa)-np.min(pa))
      cohorts.append({'stem':z['stem'],'area':(x2-x1)*(y2-y1),'contrast':contrast,'overlap_dilated':int(mask[y1:y2,x1:x2].sum()),'hit':k in matched[z['stem']],'machine':z['machine'],'date':z['date']})
   print('result',met,flush=True)
# Marker-only shortcut: fit only a global size shrink factor on train, never a valid final detector.
def marker_pred(z,scale):
 bs=[]
 for x,y,w,h,n in z['components']:
  if w<4 or h<4:continue
  cx=x+w/2;cy=y+h/2;bs.append([cx-w*scale/2,cy-h*scale/2,cx+w*scale/2,cy+h*scale/2])
 return np.array(bs).reshape(-1,4),np.ones(len(bs))
opts=np.linspace(.35,1,14);sc=max(opts,key=lambda s:eval_pred({z['stem']:marker_pred(z,s) for z in subsets['train']},subsets['train'])[0]['F1'])
mm=eval_pred({z['stem']:marker_pred(z,sc) for z in subsets['test']},subsets['test'])[0];mm['size_scale']=sc
dump('xray_results.json',{'splits':{s:{'images':len(z),'objects':sum(len(t['boxes']) for t in z),'dates':sorted(set(t['date'] for t in z))} for s,z in subsets.items()},'experiments':results,'marker_only':mm})
dump('xray_predictions.json',allpred);pd.DataFrame(cohorts).to_csv(O/'xray_test_cohorts.csv',index=False)
print('DONE',flush=True)
