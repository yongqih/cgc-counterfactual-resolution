"""Download the authors' public processed data with recorded provenance."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'CellTag-information-audit'}), timeout=60)

def fetch(url, path, expected_size=None, expected_md5=None):
    if not path.exists() or (expected_size and path.stat().st_size != expected_size):
        partial = path.with_suffix(path.suffix + '.partial')
        if expected_size and expected_size > 100_000_000:
            prefix_size = partial.stat().st_size if partial.exists() else 0
            chunk_size = 64 * 1024 * 1024
            ranges = [(a, min(expected_size, a + chunk_size)) for a in range(prefix_size, expected_size, chunk_size)]
            def part(bounds):
                a, b = bounds
                target = path.with_name(path.name + f'.range_{a}_{b}')
                if not target.exists() or target.stat().st_size != b-a:
                    req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b-1}', 'User-Agent':'CellTag-information-audit'})
                    with urllib.request.urlopen(req, timeout=90) as response, target.open('wb') as out:
                        assert response.status == 206
                        assert response.headers['Content-Range'].startswith(f'bytes {a}-{b-1}/')
                        while block := response.read(4 * 1024 * 1024):
                            out.write(block)
                assert target.stat().st_size == b-a
                return target
            with ThreadPoolExecutor(max_workers=6) as pool:
                for i, future in enumerate(as_completed([pool.submit(part, bounds) for bounds in ranges]),1):
                    future.result()
                    if i % 6 == 0 or i == len(ranges):
                        print(f'Completed {i}/{len(ranges)} download segments', flush=True)
            with partial.open('ab') as out:
                for a,b in ranges:
                    target = path.with_name(path.name + f'.range_{a}_{b}')
                    with target.open('rb') as source:
                        while block := source.read(8 * 1024 * 1024):
                            out.write(block)
            assert partial.stat().st_size == expected_size
            partial.replace(path)
            for a,b in ranges:
                path.with_name(path.name + f'.range_{a}_{b}').unlink()
        else:
            stream_fetch(url, partial, path)
    hashes = {name: hashlib.new(name) for name in ['md5', 'sha256']}
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            for h in hashes.values():
                h.update(block)
    record = {'url': url, 'path': str(path.relative_to(ROOT)), 'size': path.stat().st_size,
              **{name: h.hexdigest() for name, h in hashes.items()}}
    if expected_size:
        assert record['size'] == expected_size
    if expected_md5:
        assert record['md5'] == expected_md5
    print(json.dumps(record), flush=True)
    return record

def stream_fetch(url, partial, path):
        started = last = time.monotonic()
        n = 0
        with request(url) as response, partial.open('wb') as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
                n += len(block)
                if time.monotonic() - last > 25:
                    print(f'{path.name}: {n / 1e6:.0f} MB, {n / 1e6 / (time.monotonic() - started):.1f} MB/s', flush=True)
                    last = time.monotonic()
        partial.replace(path)

if __name__ == '__main__':
    metadata_url = 'https://api.figshare.com/v2/articles/24119358'
    with request(metadata_url) as f:
        metadata = json.load(f)
    (DATA / 'figshare_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest = []
    base = 'https://raw.githubusercontent.com/morris-lab/CellTag-multi-2023/5fd3778bd3b056239b04a583ebf5911242c5f4ed/'
    manifest.append(fetch(base + 'clone_tables/hsc.rna%26atac.r1%262_master_v2.csv', DATA / 'clone_table.csv'))
    manifest.append(fetch(base + 'README.md', DATA / 'authors_README.md'))
    for file in metadata['files']:
        manifest.append(fetch(file['download_url'], DATA / file['name'], file['size'], file.get('computed_md5')))
    manifest.append(fetch('https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6681nnn/GSM6681127/suppl/GSM6681127_d2_5_filtered_feature_bc_matrix.h5', DATA / 'day2_rna_counts.h5', 27462037))
    (DATA / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
