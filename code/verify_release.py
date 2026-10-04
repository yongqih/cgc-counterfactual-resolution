"""Check the released files against their manifest; no model is fitted."""
import argparse,csv,hashlib,json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-only',action='store_true',help='Validate the GitHub/CODE subset without the DATA archive')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    rows=[r for r in manifest['files'] if not args.code_only or r['archive']=='CODE']
    errors=[]
    for row in rows:
        f=root/row['path']
        if not f.is_file():errors.append({'path':row['path'],'error':'missing'})
        elif f.stat().st_size!=row['bytes'] or digest(f)!=row['sha256']:errors.append({'path':row['path'],'error':'hash mismatch'})
    if not args.code_only:
        expected=[f'Figure_{i}' for i in range(1,6)]+[f'Extended_Data_{i}' for i in range(1,4)]+[f'Supplementary_Figure_{i}' for i in range(1,5)]
        for name in expected:
            for ext in ['pdf','svg','png','tiff']:
                if not (root/'figures'/f'{name}.{ext}').is_file():errors.append({'figure':name,'missing_format':ext})
    print(json.dumps({'status':'FAIL' if errors else 'PASS','files_verified':len(rows),'mode':'CODE' if args.code_only else 'CODE+DATA','scientific_recomputation':False,'errors':errors},indent=2))
    raise SystemExit(bool(errors))
if __name__=='__main__':main()
