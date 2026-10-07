import ast,json,math
from pathlib import Path
O=Path('work')
def read(n):return json.loads((O/n).read_text(encoding='utf-8'))
d=read('xray_labeled.json');a=read('audit_summary.json');sp=read('split.json');r=read('xray_results.json')
assert len(d)==500 and len({z['sha'] for z in d.values()})==500
assert a['objects']==sum(len(z['boxes']) for z in d.values())==1147
assert set(sp)==set(d)
dates={g:{z['date'] for k,z in d.items() if sp[k]==g} for g in ['train','val','test']}
assert not (dates['train']&dates['val'] or dates['train']&dates['test'] or dates['val']&dates['test'])
assert sum(v['images'] for v in r['splits'].values())==500
for row in r['experiments']+read('robustness_results.json')['experiments']+read('robustness_hgb_results.json')['experiments']:
 assert row['TP']+row['FN']==row['objects']==133
 assert math.isclose(row['F1'],2*row['TP']/(2*row['TP']+row['FP']+row['FN']),abs_tol=1e-12)
assert read('overlap_result.json')['metrics']['objects']==312
assert sum(c['n'] for c in read('overlap_result.json')['cohorts'])==312
assert len(read('source_file_manifest.json'))==530
for p in O.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
for p in ['model_comparison.png','robustness.png','preprocessing_examples.png']:assert (Path('outputs')/p).is_file()
out={'status':'passed','unique_images':500,'objects':1147,'date_splits_disjoint':True,'metrics_consistent':True,'source_annotation_files':530}
(O/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
