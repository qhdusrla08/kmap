"""Reproduce the X-ray audit and experiments using only the X-ray dataset."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-root',required=True,help='Path to the X-ray dataset directory itself')
 a=p.parse_args();source=Path(a.data_root).resolve()
 if not source.is_dir():p.error(f'Dataset directory does not exist: {source}')
 root=Path(__file__).resolve().parent;os.chdir(root)
 os.environ['XRAY_DATA_ROOT']=str(source);os.environ['PYTHONIOENCODING']='utf-8';os.environ['OMP_NUM_THREADS']='4'
 Path('outputs').mkdir(exist_ok=True);Path('work').mkdir(exist_ok=True)
 stages=['audit.py','extra_audit.py','provenance.py','experiments.py','robustness.py','robustness_hgb.py','overlap_experiment.py','summarize.py','verify.py']
 status=[]
 for stage in stages:
  print('RUN',stage,flush=True);start=time.monotonic()
  with open(Path('work')/(Path(stage).stem+'.log'),'w',encoding='utf-8') as f:
   result=subprocess.run([sys.executable,str(Path('work')/stage)],stdout=f,stderr=subprocess.STDOUT)
  status.append({'stage':stage,'exit_code':result.returncode,'elapsed_seconds':round(time.monotonic()-start,2)})
  Path('work/run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  if result.returncode:
   print(f'FAILED: work/{Path(stage).stem}.log',flush=True);return result.returncode
  print('PASS',stage,flush=True)
 print('All X-ray stages completed. Results: work/. Figures: outputs/.',flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
