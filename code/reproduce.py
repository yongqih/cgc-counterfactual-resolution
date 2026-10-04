"""Execute current analyses in a separate workspace using packaged prepared inputs."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

TASKS={
 'larry':('larry_information_scale_20261003/analyze.py','Recompute all 21-clone estimates, three sensitivity variants and 3,000 conditional bootstrap draws from cell-level RNA.'),
 'celltag-calibration':('larry_information_scale_20261003/reuse_celltag.py','Reconstruct prediction sets from saved probabilities and calibration cutoffs; audit disjoint splits; recompute specificity and paired contrasts.'),
 'celltag-fit':('celltag_information_resolution_20261001/six_resolution.py','Refit LR and RF in 150 nested training/calibration/test runs on the prepared 165-clone six-fate cohort.'),
 'tahoe-summary':('tahoe_joint_resolution_20261001/summarize.py','Recompute resolution metrics and bootstrap summaries from frozen held-context sufficient statistics.'),
 'tahoe-fit':('tahoe_joint_resolution_20261001/run.py','Refit the original joint-resolution experiment; requires the documented Gram caches and CUDA.'),
 'tahoe-gene-selection':('tahoe_supported_detail_20261002/run.py','Refit train-only gene-selection policies; requires original response tensors and CUDA.')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=TASKS)
    p.add_argument('--work-dir',type=Path,help='Separate reusable workspace; no approved export is overwritten')
    p.add_argument('--input-root',type=Path,help='Original prepared data root for Tahoe refitting')
    p.add_argument('--inspect',action='store_true')
    args=p.parse_args()
    script,description=TASKS[args.task]
    if args.inspect:
        print(json.dumps({'task':args.task,'description':description,'entrypoint':script},indent=2));return
    if args.work_dir is None:p.error('--work-dir is required')
    release=Path(__file__).resolve().parents[1]
    work=args.work_dir.resolve()
    if work==release or release in work.parents:p.error('Use a workspace outside the release directory')
    if not (release/'prepared/experiments').is_dir():p.error('Extract the DATA archive beside the CODE archive first')
    work.mkdir(parents=True,exist_ok=True)
    shutil.copytree(release/'code/analysis',work,dirs_exist_ok=True)
    # Seed missing inputs only; successive stages keep freshly computed outputs.
    for src in (release/'prepared').rglob('*'):
        if src.is_file():
            dst=work/src.relative_to(release/'prepared')
            if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    if args.task in ['tahoe-fit','tahoe-gene-selection']:
        if args.input_root is None:p.error('This stage requires --input-root and CUDA; see REPRODUCIBILITY.md')
        for name in ['data','results']:
            target=(args.input_root/name).resolve()
            if not target.is_dir():p.error(f'Missing prepared input directory: {target}')
        if args.task=='tahoe-fit':
            src=args.input_root/'experiments/tahoe_joint_resolution_20261001/cache'
            if not src.is_dir():p.error(f'Missing Gram cache: {src}')
            shutil.copytree(src,work/'experiments/tahoe_joint_resolution_20261001/cache',dirs_exist_ok=True)
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    if args.input_root:env['CGC_INPUT_ROOT']=str(args.input_root.resolve())
    subprocess.run([sys.executable,str(work/'experiments'/script)],cwd=work,env=env,check=True)
if __name__=='__main__':main()
