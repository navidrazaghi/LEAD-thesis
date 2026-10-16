# Full, paginated listing of the LEAD 123D release with file sizes.
# Metadata only: no dataset file is downloaded.
import json, pathlib, re, time, urllib.request
url = 'https://huggingface.co/api/datasets/ln2697/lead-123d/tree/main?recursive=true'
out, pages, err = [], 0, None
while url:
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'lead-fetch'})
            with urllib.request.urlopen(req, timeout=120) as r:
                page = json.loads(r.read()); link = r.headers.get('Link', '')
            break
        except Exception as e:
            err = e; time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f'FAILED at page {pages}: {err}')
    pages += 1
    out += [(e['path'], e.get('size', 0)) for e in page if e.get('type') == 'file']
    m = re.search(r'<([^>]+)>;\s*rel="next"', link or '')
    url = m.group(1) if m else None
    if pages % 25 == 0:
        print(f'page {pages}, files {len(out)}', flush=True)
    time.sleep(0.1)
pathlib.Path.home().joinpath('full_listing_sizes.json').write_text(json.dumps(out))
print(f'DONE pages={pages} files={len(out)}', flush=True)
