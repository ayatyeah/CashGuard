"""Go/model launcher contract, without weights or video processing."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ADAPTER = Path(__file__).resolve().parents[1] / 'scripts/local-model-adapter'

class ModelAdapterTests(unittest.TestCase):
    def test_single_argument_budget_and_offline_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'model checkout'; (root/'bin').mkdir(parents=True)
            launcher = root/'bin/cashguard-analyzer'
            launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$#" "$1" "$ANALYZER_BUDGET_SEC" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE"\n')
            launcher.chmod(0o755)
            output = subprocess.check_output([str(ADAPTER), '/tmp/video with spaces.mov', '171'], env={**os.environ, 'CASHGUARD_MODEL_ROOT': str(root)}, text=True)
            self.assertEqual(output.splitlines(), ['1', '/tmp/video with spaces.mov', '171', '1', '1'])

    def test_rejects_invalid_budgets(self):
        for budget in ['0', '-1', 'not-a-number', '10;echo bad']:
            result = subprocess.run([str(ADAPTER), '/tmp/video.mov', budget], env={**os.environ, 'CASHGUARD_MODEL_ROOT':'/nonexistent'}, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_missing_root(self):
        env = dict(os.environ); env.pop('CASHGUARD_MODEL_ROOT', None)
        result = subprocess.run([str(ADAPTER), '/tmp/video.mov'], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CASHGUARD_MODEL_ROOT', result.stderr)

if __name__ == '__main__': unittest.main()
