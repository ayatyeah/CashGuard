import sys,json,time,argparse
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,'scripts')
from common import load_video,CACHE
from app.ml.events import analyze_scene
from app.ml.moments import propose
from app.ml.candidates import crop_scaled
from app.ml.vlm import VLM
ap=argparse.ArgumentParser();ap.add_argument('names',nargs='+');args=ap.parse_args()
vlm=VLM(threads=8)
question=('These four CCTV frames show a bank transaction in chronological order: top-left, top-right, bottom-left, bottom-right. '
          'Inspect the objects exchanged, not just reaching hands. A card or paper is not a bundle of banknotes. '
          'Answer true only for directly visible actions. Return JSON booleans: '
          '{"client_gives_banknotes":false,"teller_counts_banknotes":false,"teller_returns_banknotes":false,"teller_returns_card_or_paper":false}.')
for name in args.names:
 dst=Path('artifacts/checks-20261017')/(name+'-sequence.json')
 if dst.exists():continue
 d,frames=load_video(name);sc=analyze_scene(d['obs'],d['meta']['width'],d['meta']['height'])
 moms=[m for m in propose(sc,frames,[]) if m['kind']=='visitor']
 result=[]
 for m in moms:
  tr=next(t for t in sc.tracks if t.tid==m['tid']);near=tr.idx[tr.info['d']<1.6]
  idx=[int(near[min(len(near)-1,int((len(near)-1)*f))]) for f in (.15,.4,.7,.9)]
  ims=[crop_scaled(cv2.imread(str(Path(CACHE)/'frames'/name/f'{d["obs"][i].frame_id:05d}.jpg')),m['crop'],1.) for i in idx]
  cells=[]
  for im in ims:
   h,w=im.shape[:2];s=min(320/w,320/h);im=cv2.resize(im,(max(1,int(w*s)),max(1,int(h*s))))
   cell=np.zeros((320,320,3),np.uint8);cell[:im.shape[0],:im.shape[1]]=im;cells.append(cell)
  grid=np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])])
  start=time.monotonic();ans=vlm.ask_json([grid],question,max_new_tokens=100)
  result.append({'times':[float(sc.times[i]) for i in idx],'answer':ans,'seconds':time.monotonic()-start})
  print(name,result[-1],flush=True)
 dst.write_text(json.dumps(result,ensure_ascii=False,indent=2))
