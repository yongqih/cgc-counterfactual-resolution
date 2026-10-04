"""Build deterministic CODE and DATA ZIPs from the verified release manifest."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path

PREFIX='CGC'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for kind in ['CODE','DATA']:
        paths=[r['path'] for r in manifest['files'] if r['archive']==kind]
        if kind=='CODE':paths.append('RELEASE_MANIFEST.json')
        out=args.output/f'CGC_{kind}_2026-10-04.zip'
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(paths):
                p=root/name
                row=next((r for r in manifest['files'] if r['path']==name),None)
                if row and digest(p)!=row['sha256']:raise RuntimeError(f'File changed: {name}')
                info=zipfile.ZipInfo(PREFIX+'/'+name,date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
                with p.open('rb') as source,archive.open(info,'w',force_zip64=True) as target:
                    for b in iter(lambda:source.read(8*1024*1024),b''):target.write(b)
        with zipfile.ZipFile(out) as archive:
            error=archive.testzip()
            if error:raise RuntimeError(f'ZIP CRC failed: {error}')
        report.append({'file':out.name,'files':len(paths),'bytes':out.stat().st_size,'sha256':digest(out)})
    (args.output/'PACKAGE_MANIFEST.json').write_text(json.dumps({'release':manifest['release'],'archive_root':PREFIX,'archives':report,'public_upload_performed':False},indent=2),encoding='utf-8')
    with (args.output/'SHA256SUMS.txt').open('w',encoding='utf-8') as f:
        for r in report:f.write(r['sha256']+'  '+r['file']+'\n')
    (args.output/'README_UPLOAD.md').write_text('''# Upload files for the current manuscript

Upload both dated ZIPs as the replacement CODE and DATA payloads for the revised manuscript. They share one top-level directory and should be extracted together. The original public v1.0.1 record remains historical. Publishing these archives to Zenodo is a separate step from updating the GitHub repository.

On Windows, use a short extraction directory such as `C:/CGC` to accommodate the preserved historical source filenames.

The local GitHub working copy is in `../github_publication/`. It contains the CODE subset. To run prepared-data analyses in a GitHub checkout, copy `prepared/` from the DATA archive into the checkout root. Figure rendering requires only the CODE subset, which includes the current figure source tables.

Use `PACKAGE_MANIFEST.json` and `SHA256SUMS.txt` to check the upload payloads. After the new deposit has a public identifier, update the manuscript's code-availability paragraph with that identifier.
''',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
