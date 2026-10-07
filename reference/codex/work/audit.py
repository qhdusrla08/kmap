import sys, json, hashlib, re, os
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image,ImageDraw
from collections import Counter,defaultdict
import cv2
B=Path(os.environ['XRAY_DATA_ROOT'])
O=Path('work'); O.mkdir(exist_ok=True)
def dump(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
X=B
fs=[p for p in X.rglob('*') if p.is_file()]
raw=[p for p in fs if p.suffix.lower()=='.bmp' and 'NgImage' in str(p.parent)]
rawmap=defaultdict(list)
for p in raw:rawmap[p.stem].append(p)
ld=next(p for p in X.rglob('labels') if '라벨링' in str(p))
labels={p.stem:p for p in ld.glob('*.txt')}
jpegs=[p for p in fs if p.suffix.lower()=='.jpg' and '라벨링' in str(p)]
jm=defaultdict(list)
for p in jpegs:jm[p.stem].append(p)
records=[]; objects=[]; hashes=Counter(); imgs={};bad=[]
for i,p in enumerate(raw):
 a=np.array(Image.open(p).convert('RGB'));h,w=a.shape[:2]
 sha=hashlib.sha256(a.tobytes()).hexdigest();hashes[sha]+=1
 spread=a.max(2).astype(int)-a.min(2).astype(int)
 mask=(spread>40).astype(np.uint8)
 nc,cc,stats,cent=cv2.connectedComponentsWithStats(mask,8)
 comps=[list(map(int,z)) for z in stats[1:] if z[4]>=4]
 row={'stem':p.stem,'path':str(p),'width':w,'height':h,'sha':sha,'marker_pixels':int(mask.sum()),'marker_components':len(comps),'machine':p.stem.split('_')[0],'date':p.stem.split('_')[1],'has_txt':p.stem in labels}
 if row['has_txt']:
  boxes=[]
  for line in labels[p.stem].read_text().splitlines():
   v=list(map(float,line.split()))
   if len(v)!=5:bad.append([p.stem,line]);continue
   c,x,y,bw,bh=v;box=np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]);boxes.append(box.tolist())
   x1,y1,x2,y2=np.round(box).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
   patch=mask[y1:y2,x1:x2]
   hit=0;best=0
   for xx,yy,ww,hh,area in comps:
    inter=max(0,min(x2,xx+ww)-max(x1,xx))*max(0,min(y2,yy+hh)-max(y1,yy))
    union=(x2-x1)*(y2-y1)+ww*hh-inter
    best=max(best,inter/union if union else 0)
   objects.append({'stem':p.stem,'class':c,'w':bw*w,'h':bh*h,'area':bw*w*bh*h,'x':x,'y':y,'marker_overlap':int(patch.sum()),'marker_iou':best,'valid':bool(0<=x-bw/2<x+bw/2<=1 and 0<=y-bh/2<y+bh/2<=1)})
  row['n_boxes']=len(boxes);row['boxes']=boxes;row['components']=comps
  imgs[p.stem]=row
 records.append(row)
 if i%500==0: print('audit images',i,flush=True)
pd.DataFrame([{k:v for k,v in t.items() if k not in ['boxes','components']} for t in records]).to_csv(O/'xray_inventory.csv',index=False,encoding='utf-8-sig')
objects=list({(o['stem'],o['x'],o['y'],o['w'],o['h']):o for o in objects}.values())
pd.DataFrame(objects).to_csv(O/'xray_objects.csv',index=False)
dump('xray_labeled.json',imgs)
conflicts=[]
for p in fs:
 if p.suffix=='.txt' and p.stem in labels and p!=labels[p.stem]:
  try:
   a=np.loadtxt(p,ndmin=2);b=np.loadtxt(labels[p.stem],ndmin=2)
   conflicts.append({'stem':p.stem,'path':str(p),'same':bool(a.shape==b.shape and np.allclose(a,b)),'n':len(a)})
  except Exception:pass
ob=pd.DataFrame(objects)
summary={'extensions':dict(Counter(p.suffix for p in fs)),'raw_count':len(raw),'raw_unique_stems':len(rawmap),'raw_unique_pixels':len(hashes),'raw_duplicate_extra':sum(v-1 for v in hashes.values()),'raw_with_marker':sum(t['marker_pixels']>0 for t in records),'label_files':len(labels),'labels_matching_raw':len(imgs),'labels_without_raw':sorted(set(labels)-set(imgs)),'jpeg_copies':len(jpegs),'jpeg_unique_stems':len(jm),'labeled_marker_images':sum(t['marker_pixels']>0 for t in imgs.values()),'labeled_empty':sum(t['n_boxes']==0 for t in imgs.values()),'objects':len(ob),'object_describe':ob.describe().to_dict(),'boxes_with_marker':int((ob.marker_overlap>0).sum()),'invalid_boxes':int((~ob.valid).sum()),'txt_conflicts':conflicts,'labeled_dates':dict(Counter(t['date'] for t in imgs.values())),'labeled_machines':dict(Counter(t['machine'] for t in imgs.values())),'sizes':dict(Counter(str((t['width'],t['height'])) for t in records))}
dump('audit_summary.json',summary)
# Representative contact sheet: no added boxes, RGB originals.
sel=list(imgs.values())[::max(1,len(imgs)//12)][:12]
sheet=Image.new('RGB',(1152,3*380),'white'); dr=ImageDraw.Draw(sheet)
for i,t in enumerate(sel):
 im=Image.open(t['path']).convert('RGB');im.thumbnail((380,345));x=(i%3)*384;y=(i//3)*285
 sheet.paste(im,(x,y+20));dr.text((x+4,y+4),t['stem'],fill='black')
sheet.save(O/'xray_contact.png')
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str),flush=True)
