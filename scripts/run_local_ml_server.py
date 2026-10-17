#!/usr/bin/env python3
"""Run the host Go API with the team's existing local Python models. No credentials in argv."""
import argparse
import os
from pathlib import Path
from urllib.parse import quote

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-root', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
model = args.model_root.resolve()
config = dict(line.split('=', 1) for line in (root/'.env').read_text().splitlines() if line.strip() and not line.startswith('#') and '=' in line)
python = model/'.venv/bin/python'
server = root/'data/local-runtime/bin/cashguard-server'
for path in [python, server, model/'bin/cashguard-analyzer', model/'models/scorer.json']:
    if not path.is_file(): raise SystemExit(f'Missing: {path}')
env = {**os.environ, 'API_KEY': config['API_KEY'], 'DEMO_MODE': 'false',
       'DATABASE_URL': f"postgres://cashguard:{quote(config['POSTGRES_PASSWORD'], safe='')}@127.0.0.1:{config.get('POSTGRES_PORT','55432')}/cashguard?sslmode=disable",
       'DATA_DIR': str(root/'data/local-runtime'), 'ADDR': '127.0.0.1:8081',
       'ANALYZER_COMMAND': str(root/'scripts/local-model-adapter'),
       'CASHGUARD_MODEL_ROOT': str(model), 'CASHGUARD_PYTHON': str(python),
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
       'FORTE_USE_VLM': '1', 'FORTE_USE_OWL': '1', 'JOB_DEADLINE_SEC': '285'}
os.chdir(root)
os.execve(server, [str(server)], env)
