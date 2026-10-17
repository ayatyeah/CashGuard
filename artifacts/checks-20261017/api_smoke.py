"""Local integration check; reads credentials without printing or storing them."""
import json
import argparse
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wait-pid', type=int, default=0)
args = parser.parse_args()
while args.wait_pid and Path(f'/proc/{args.wait_pid}').exists():
    time.sleep(2)
report = json.loads(Path('artifacts/checks-20261017/runtime.json').read_text())
assert len(report['rows']) == 4 and all(r['label'] == r['pred'] for r in report['rows']), 'Fresh regression checks failed'
model = Path('models/scorer.json')
temporary = model.with_suffix('.json.tmp')
temporary.write_text(Path('artifacts/checks-20261017/scorer-candidate.json').read_text())
temporary.replace(model)
root = Path('/home/yeahayat/ForteWin-ui')
config = dict(line.split('=', 1) for line in (root / '.env').read_text().splitlines()
              if '=' in line and not line.startswith('#'))
headers = {'X-API-Key': config['API_KEY']}
boundary = 'cashguard-retrain-smoke-20261017'
video = Path('/home/yeahayat/ForteWin/data/test/videos/Х-22.mov')
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="checks-X22.mov"\r\n'
        'Content-Type: video/quicktime\r\n\r\n').encode() + video.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
req = urllib.request.Request('http://127.0.0.1:8081/api/jobs', data=body,
    headers={**headers, 'Content-Type': f'multipart/form-data; boundary={boundary}'})
job = json.load(urllib.request.urlopen(req, timeout=60))
print(json.dumps(job, ensure_ascii=False), flush=True)
jobid = job.get('id') or job.get('job_id')
if not jobid:
    raise RuntimeError('No job id')
start = time.monotonic()
while time.monotonic() - start < 330:
    req = urllib.request.Request(f'http://127.0.0.1:8081/api/jobs/{jobid}', headers=headers)
    result = json.load(urllib.request.urlopen(req, timeout=10))
    if result.get('status') in ('done', 'failed', 'error'):
        Path('artifacts/checks-20261017/api-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: result.get(k) for k in ('job_id', 'status', 'verdict', 'demo')}, ensure_ascii=False), flush=True)
        break
    time.sleep(3)
else:
    raise RuntimeError('API job did not complete')
