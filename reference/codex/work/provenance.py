import os,json,hashlib,platform
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from importlib.metadata import version
B=Path(os.environ['XRAY_DATA_ROOT']);O=Path('work')
d=json.loads((O/'xray_labeled.json').read_text(encoding='utf-8'));hist=Counter();empty=positive=0
for p in B.rglob('*.xml'):
 n=len(ET.parse(p).getroot().findall('object'));hist[n]+=1
 empty+=int(n==0 and p.stem in d);positive+=int(n>0 and p.stem in d)
(O/'xml_audit.json').write_text(json.dumps({'count_histogram':dict(hist),'empty_overlapping_positive_txt':empty,'positive_overlapping_txt':positive},indent=2),encoding='utf-8')
sources=[]
for parent in B.rglob('*'):
 if parent.is_dir() and parent.name in ['labels','YOLO_darknet']:
  for p in sorted(parent.glob('*.txt')):
   sources.append({'relative_path':str(p.relative_to(B)),'bytes':p.stat().st_size,'sha256_file':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'source_file_manifest.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
packages=['numpy','pandas','scikit-learn','scipy','opencv-python-headless','Pillow','matplotlib','joblib','threadpoolctl']
(O/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':{p:version(p) for p in packages}},indent=2),encoding='utf-8')
print('XML files:',sum(hist.values()),'annotation TXT file hashes:',len(sources))
