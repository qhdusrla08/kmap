import sys,json,re,hashlib,os
from pathlib import Path
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw,ImageFont
B=Path(os.environ['XRAY_DATA_ROOT']);O=Path('work')
def dump(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
d=json.loads((O/'xray_labeled.json').read_text(encoding='utf-8'));objects=[];overlap_images=0;quality=[];thumbs=[];stems=[]
for z in d.values():
 a=np.array(Image.open(z['path']).convert('RGB'));h,w=a.shape[:2];mask=(a.max(2).astype(int)-a.min(2).astype(int)>40).astype(np.uint8);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 dil=cv2.dilate(mask,np.ones((3,3),np.uint8));inp=cv2.inpaint(gray,dil,3,cv2.INPAINT_TELEA)
 any_overlap=False
 for k,box in enumerate(z['boxes']):
  x1,y1,x2,y2=np.round(box).astype(int);m=mask[y1:y2,x1:x2];dm=dil[y1:y2,x1:x2];pa=inp[y1:y2,x1:x2];old=gray[y1:y2,x1:x2]
  ious=[];inside=[]
  for xx,yy,ww,hh,area in z['components']:
   inter=max(0,min(x2,xx+ww)-max(x1,xx))*max(0,min(y2,yy+hh)-max(y1,yy));union=(x2-x1)*(y2-y1)+ww*hh-inter
   ious.append(inter/union if union else 0);inside.append(xx<=((x1+x2)/2)<=xx+ww and yy<=((y1+y2)/2)<=yy+hh)
  frac=float(dm.mean());any_overlap|=m.sum()>0
  objects.append({'stem':z['stem'],'k':k,'w':box[2]-box[0],'h':box[3]-box[1],'area':(box[2]-box[0])*(box[3]-box[1]),'x':(x1+x2)/2/w,'y':(y1+y2)/2/h,'marker_pixels':int(m.sum()),'dilated_fraction':frac,'center_inside_marker':any(inside),'marker_iou':max(ious,default=0),'clean_contrast':int(np.median(pa))-int(pa.min()),'raw_contrast':int(np.median(old))-int(old.min())})
 overlap_images+=any_overlap
 thumbs.append(cv2.resize(inp,(32,32),interpolation=cv2.INTER_AREA).ravel().astype(float));stems.append(z['stem'])
ob=pd.DataFrame(objects);ob.to_csv(O/'xray_objects_unique.csv',index=False)
ds=Path(next(B.rglob('라벨링 6종 세트')))
jpgs=list(ds.rglob('*.jpg'));jstats={}
for fol in sorted(set(p.parent for p in jpgs)):jstats[fol.name]=len(list(fol.glob('*.jpg')))
alltxt=[]
for fol in B.rglob('*'):
 if fol.is_dir() and fol.name in ['labels','YOLO_darknet']:
  alltxt+=list(fol.glob('*.txt'))
summary={'unique_labeled_images':len(d),'unique_objects':len(ob),'images_with_marker_inside_txt':int(overlap_images),'objects_with_marker_inside_txt':int((ob.marker_pixels>0).sum()),'objects_center_enclosed_by_marker':int(ob.center_inside_marker.sum()),'objects_affected_by_dilation':int((ob.dilated_fraction>0).sum()),'objects_dilation_over_25pct':int((ob.dilated_fraction>.25).sum()),'object_stats':ob.describe().to_dict(),'jpeg_folder_counts':jstats,'annotation_txt_files':len(alltxt),'unique_txt_stems':len(set(p.stem for p in alltxt)),'additional_txt_stems':sorted(set(p.stem for p in alltxt)-set(d))}
dump('xray_unique_summary.json',summary)
# Three fixed examples: original, RGB-derived mask/inpaint and zooms, with TXT boxes only in bottom strip.
chosen=[list(d)[0],next(k for k in d if k.startswith('001_20200622')),next(k for k in d if k.startswith('002_20200905'))]
canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas)
for row,stem in enumerate(chosen):
 z=d[stem];a=np.array(Image.open(z['path']).convert('RGB'));ma=(a.max(2).astype(int)-a.min(2).astype(int)>40).astype(np.uint8);ma=cv2.dilate(ma,np.ones((3,3),np.uint8));gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);ip=cv2.inpaint(gray,ma,3,cv2.INPAINT_TELEA);masked=gray.copy();masked[ma>0]=np.median(gray)
 for col,(v,title) in enumerate([(a,'Raw'),(masked,'Mask'),(ip,'Inpaint')]):
  im=Image.fromarray(v).convert('RGB');im.thumbnail((380,245));canvas.paste(im,(col*400,row*300+35));draw.text((col*400+8,row*300+8),title+' | '+stem,fill='black')
canvas.save(O/'preprocessing_examples.png')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
