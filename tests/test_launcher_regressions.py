"""Run the real launcher with dependency-free model doubles (no weights needed)."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = r'''
import runpy, sys, types
from types import SimpleNamespace as NS
root, scenario = sys.argv[1:]
sys.path.insert(0, root)
def module(name, **values):
    m = types.ModuleType(name)
    m.__dict__.update(values)
    sys.modules[name] = m
    return m
module('app'); module('app.analyzers'); module('app.ml'); module('scripts')
pipeline = module('app.ml.pipeline', title=lambda *args: 'event')
sys.modules['app.ml'].pipeline = pipeline
class Analyzer:
    def __init__(self):
        self.models = NS(norms={})
    def analyze(self, path, meta, budget):
        assert 0 < budget <= 10, budget
        detections = [] if scenario == 'clean' else [NS(
            evidence_frame_ids=[1], frame_id=1, type_id='event', description='observed')]
        if scenario in ('slow', 'brisk'):  # sampling 25/60 = 0.4 fps vs 25/15 = 1.7 fps
            return NS(detections=[], warnings=['мало времени: частота кадров снижена'],
                      timings={'step': 60 if scenario == 'slow' else 15})
        return NS(detections=detections, warnings=['partial analysis'], timings={})
module('app.analyzers.ml', MLAnalyzer=Analyzer)
module('app.ml.perception', video_meta=lambda path: {'fps': 25})
module('scripts.analyze_video', frame_b64=lambda *args: '' if scenario == 'missing' else 'frame')
budget = {'expired': '-1', 'nan': 'nan', 'infinite': 'inf'}.get(scenario, '10')
sys.argv = [root + '/scripts/cashguard_analyzer.py', 'video.mov', budget]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


class LauncherRegressionTests(unittest.TestCase):
    def launch(self, scenario):
        return subprocess.run([sys.executable, '-c', RUNNER, str(ROOT), scenario],
                              capture_output=True, timeout=20)

    def test_warnings_and_small_budget_survive(self):
        for scenario in ('clean', 'violation'):
            with self.subTest(scenario=scenario):
                result = self.launch(scenario)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload['warnings'], ['partial analysis'])
                self.assertEqual(payload['verdict'], 'no_violation' if scenario == 'clean' else 'violation')

    def test_low_fps_warning_only_below_one_frame_per_second(self):
        slow = json.loads(self.launch('slow').stdout)
        self.assertEqual(len(slow['warnings']), 1)
        self.assertIn('0.4/с', slow['warnings'][0])
        brisk = json.loads(self.launch('brisk').stdout)
        self.assertEqual(brisk.get('warnings'), [])

    def test_missing_evidence_never_returns_clean_result(self):
        result = self.launch('missing')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')

    def test_invalid_remaining_budgets_fail(self):
        for scenario in ('expired', 'nan', 'infinite'):
            with self.subTest(scenario=scenario):
                result = self.launch(scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b'')


if __name__ == '__main__':
    unittest.main()
