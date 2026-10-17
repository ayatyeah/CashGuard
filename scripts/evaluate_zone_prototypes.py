"""Run after train_scorer --vlm-confirmed --with-vlm; never writes production weights."""
import json
import os
import time
from pathlib import Path

from common import ROOT, candidates_for, load_labels
import train_scorer as T
from app.ml import scorer as S


def metrics(pred, labels, videos):
    tp = sum(bool(pred.get(v)) and labels[v]['label'] == 1 for v in videos)
    fp = sum(bool(pred.get(v)) and labels[v]['label'] == 0 for v in videos)
    positives = sum(labels[v]['label'] for v in videos)
    return {'acc': (len(videos) - positives + tp - fp) / len(videos),
            'TP': tp, 'FP': fp, 'FN': positives - tp, 'fp_videos': [v for v in videos if pred.get(v) and not labels[v]['label']]}


def main():
    os.environ['FORTE_ZONE_PROTOTYPES'] = '1'
    labels = load_labels()
    videos = sorted(v for v, x in labels.items() if x['split'] == 'train' and 'base' not in x)
    if len(videos) != 40:
        raise ValueError('Expected exactly 40 original development videos')
    rows = json.loads(Path(ROOT, 'data/train_rows.json').read_text())
    rows = [r for r in rows if r['video'] in videos]
    if set(r['video'] for r in rows) != set(videos):
        raise ValueError('Regenerate data/train_rows.json with train_scorer: video coverage differs')
    baseline_model = json.loads(Path(ROOT, 'models/scorer.json').read_text())
    enabled = {k for k, v in baseline_model['stumps'].items() if v.get('enabled', True)} - set(S.ZONE_EXPERIMENTAL)
    values = {}
    start = time.perf_counter()
    for v in videos:
        _, _, candidates = candidates_for(v)
        for c in candidates:
            matches = [r for r in rows if r['video'] == v and r['kind'] == c.kind and abs(r['t'] - c.t) < .001]
            if len(matches) != 1:
                raise ValueError(f'Regenerate train_rows.json: unmatched candidate {v} {c.kind} {c.t}')
            for feature in ('desk_change', 'safe_change', 'desk_blob', 'dur'):
                if c.kind in {'next_visit', 'safe'} and feature in c.f:
                    if abs(matches[0]['f'].get(feature, float('inf')) - c.f[feature]) > 1e-7:
                        raise ValueError(f'Regenerate train_rows.json: changed feature {v} {feature}')
            for r in rows:
                if r['video'] == v and r['kind'] == c.kind and abs(r['t'] - c.t) < .001:
                    r['f'].update({k: value for k, value in c.f.items()
                                   if k in {'zncc_change', 'census_change', 'persistent_change', 'small_residual'}})
        values[v] = {k: [round(c.f[feat], 7) for c in candidates if c.kind == kind and feat in c.f]
                     for k, (kind, feat, _, _) in S.ZONE_EXPERIMENTAL.items()}
    elapsed = time.perf_counter() - start
    T.GROUPS.update({v: labels[v]['cabinet'] for v in videos})
    result = {}
    for name, added in [('baseline', set())] + [(k, {k}) for k in S.ZONE_EXPERIMENTAL] + [('combined', set(S.ZONE_EXPERIMENTAL))]:
        selection = enabled | added
        result[name] = {'cv': metrics(T.lovo_predict(rows, videos, selection), labels, videos),
                        'fit': metrics(T.predict(rows, T.train(rows, selection)), labels, videos)}
    out = {'seconds': elapsed, 'metrics': result, 'values': values,
           'stumps': {k: v for k, v in T.train(rows)['stumps'].items() if k in S.ZONE_EXPERIMENTAL}}
    path = Path(ROOT, 'data/zone_prototypes.json')
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'values'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
