"""Run the participant-1 preparation, starting from locally obtained source files."""
import argparse,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True,help='Directory containing sample_2010 through sample_2018 folders')
    p.add_argument('--work',type=Path,required=True,help='Intermediate cache directory outside the deliverable folder')
    p.add_argument('--skip-download',action='store_true',help='Use bundled external data snapshots')
    a=p.parse_args();root=Path(__file__).resolve().parent
    jobs=[]
    if not a.skip_download:jobs.append([sys.executable,str(root/'scripts/public_sources.py'),'--output',str(root)])
    jobs.extend([
        [sys.executable,str(root/'scripts/ingest.py'),'--source-root',str(a.source_root.resolve()),'--work',str(a.work.resolve()),'--output',str(root)],
        [sys.executable,str(root/'scripts/prepare.py'),'--work',str(a.work.resolve()),'--output',str(root)],
        [sys.executable,str(root/'scripts/validate_actual.py'),'--work',str(a.work.resolve()),'--output',str(root)],
        [sys.executable,str(root/'scripts/make_report.py'),'--work',str(a.work.resolve()),'--output',str(root)],
    ])
    for command in jobs:subprocess.run(command,check=True)
