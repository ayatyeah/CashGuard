import sys,json,time
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,'scripts')
from common import load_video,CACHE,load_labels
from app.ml.events import analyze_scene
from app.ml.moments import safe_zone,_box_overlap
from app.ml.candidates import _persons,crop_scaled,generate
from app.ml.ovd import Owl
owl=Owl(threads=6);owl.queries=['a key in a lock','a metal key','a safe door handle','a metal cabinet door','a padlock']
results={}
for name in sorted(load_labels()):
 d,frames=load_video(name);obs=d['obs'];sc=analyze_scene(obs,d['meta']['width'],d['meta']['height']);box=safe_zone(sc,pad=.3)
 leaves=[c for c in generate(sc,frames) if c.kind=='leave']
 if box is None or not leaves:continue
 free=[i for i in range(len(obs)) if sc.times[i]>leaves[-1].t+.5 and not _box_overlap(box,_persons(obs[i],.3),thr=.5)]
 if len(free)<2:continue
 last=free[-1];prior=[i for i in free if sc.times[i]<=sc.times[last]-1.]
 if not prior:continue
 idx=[prior[-1],last];vals=[]
 for i in idx:
  im=cv2.imread(str(Path(CACHE)/'frames'/name/f'{obs[i].frame_id:05d}.jpg'));crop=crop_scaled(im,box,1.)
  vals.append(owl.detect(crop,threshold=.01).get('a metal key',0.))
 results[name]={'t':leaves[-1].t,'idx':idx,'times':[float(sc.times[i]) for i in idx],'values':vals,'score':min(vals)}
 print(name,results[name],flush=True)
 Path('artifacts/checks-20261017/key-persistence.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
