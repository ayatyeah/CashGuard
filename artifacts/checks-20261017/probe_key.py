import sys,json,time
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,'scripts')
from common import load_video,CACHE,load_labels
from app.ml.events import analyze_scene
from app.ml.moments import safe_zone,_box_overlap
from app.ml.candidates import _persons,crop_scaled
from app.ml.ovd import Owl
owl=Owl(threads=6);owl.queries=['a key in a lock','a metal key','a safe door handle','a metal cabinet door','a padlock']
results={}
names=sys.argv[1:] or sorted(load_labels())
for name in names:
 d,_=load_video(name,with_frames=False);obs=d['obs'];sc=analyze_scene(obs,d['meta']['width'],d['meta']['height']);box=safe_zone(sc,pad=.3)
 if box is None: continue
 free=[i for i in range(len(obs)) if not _box_overlap(box,_persons(obs[i],.3),thr=.5)]
 if not free:continue
 idx=free[-1];im=cv2.imread(str(Path(CACHE)/'frames'/name/f'{obs[idx].frame_id:05d}.jpg'));crop=crop_scaled(im,box,1.)
 start=time.monotonic();res=owl.detect(crop,threshold=.01)
 results[name]={'t':obs[idx].t,'scores':res,'seconds':time.monotonic()-start};print(name,results[name],flush=True)
 Path('artifacts/checks-20261017/key-scores.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
