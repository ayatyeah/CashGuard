"""Cache persistent-key features on all development videos, using runtime proposals.

Run before train_scorer.py. Models and dataset frames stay local.
"""
import argparse
import json
import os
from pathlib import Path

import cv2

from common import ROOT, load_labels, load_video
from app.ml.candidates import generate
from app.ml.events import analyze_scene
from app.ml.ovd import Owl
from app.ml.safe_checks import propose_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(ROOT, 'data/owl_key_scores.json')))
    args = parser.parse_args()
    results = {}
    owl = None
    for name, label in sorted(load_labels().items()):
        if 'base' in label:
            continue
        data, frames = load_video(name)
        scene = analyze_scene(data['obs'], data['meta']['width'], data['meta']['height'])
        for candidate in propose_key(scene, generate(scene, frames)):
            owl = owl or Owl(threads=6)
            values = []
            videos = [Path(ROOT, 'data', split, 'videos', name + '.mov') for split in ('train', 'test')]
            videos = [path for path in videos if path.is_file()]
            if len(videos) != 1:
                raise ValueError(f'Missing or ambiguous source video: {name}')
            cap = cv2.VideoCapture(str(videos[0]))
            for i in candidate.extra['key_idx']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, data['obs'][i].frame_id)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode source frame: {name}:{i}')
                if frame.shape[:2] != (scene.H, scene.W):
                    frame = cv2.resize(frame, (scene.W, scene.H))
                values.append(float(owl.key_score(frame, candidate.extra['key_box'])))
            cap.release()
            results[name] = {'t': candidate.t, 'idx': candidate.extra['key_idx'],
                             'values': values, 'score': min(values)}
            print(name, results[name], flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    os.replace(temporary, destination)


if __name__ == '__main__':
    main()
