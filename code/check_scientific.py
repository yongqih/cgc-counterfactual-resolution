"""Run scientific implementation tests with packaged reference outputs."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True,help='Separate directory for test workspaces and logs')
    parser.add_argument('--bundle',action='append',help='Test a named bundle; repeat to select several')
    args=parser.parse_args()
    release=Path(__file__).resolve().parents[1];work=args.work_dir.resolve()
    if work==release or release in work.parents:parser.error('Use a workspace outside the release directory')
    source=release/'code/scientific/bundles'
    available={p.name:p for p in source.iterdir() if (p/'tests').is_dir()}
    available['tahoe_joint_resolution']=release/'code/analysis/experiments/tahoe_joint_resolution_20261001'
    selected=sorted(args.bundle or available)
    if not set(selected)<=set(available):parser.error('Unknown bundle or bundle without tests')
    reference=release/'prepared/scientific_validation'
    needs_reference={'crc_pdo_application','crc_pdo_conditioning','lea_full_span'}
    if needs_reference.intersection(selected) and not reference.is_dir():parser.error('Extract the DATA archive beside the CODE archive to supply reference outputs')
    work.mkdir(parents=True,exist_ok=True);reports=[]
    for name in selected:
        target=work/name
        shutil.copytree(available[name],target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.pytest-tmp'))
        if (reference/name).is_dir():shutil.copytree(reference/name,target,dirs_exist_ok=True)
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join([str(target/'src'),str(target/'scripts'),str(target),env.get('PYTHONPATH','')])
        env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8';env['MPLBACKEND']='Agg'
        junit=work/(name+'.junit.xml')
        test_path='test_core.py' if name=='tahoe_joint_resolution' else 'tests'
        result=subprocess.run([sys.executable,'-m','pytest',test_path,'-q','-p','no:cacheprovider','--junitxml',str(junit)],cwd=target,env=env,capture_output=True)
        (work/(name+'.log')).write_bytes(result.stdout+result.stderr)
        report={'bundle':name,'exit_code':result.returncode,'log':name+'.log','junit':junit.name}
        reports.append(report);print(json.dumps(report),flush=True)
    (work/'test_results.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(any(r['exit_code'] for r in reports))
if __name__=='__main__':main()
