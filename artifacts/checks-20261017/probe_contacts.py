import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,'scripts')
from common import load_video,load_labels
from app.ml.events import analyze_scene
out={}
for name in load_labels():
 d,_=load_video(name,with_frames=False);sc=analyze_scene(d['obs'],d['meta']['width'],d['meta']['height'])
 rows=[]
 for tr in sc.tracks:
  if tr.role!='visitor':continue
  for k,i in enumerate(tr.idx):
   for ca in sc.tracks:
    if ca.role!='cashier':continue
    ix=np.where(ca.idx==i)[0]
    if not len(ix):continue
    j=ix[0];a=ca.kpts[j,[9,10]][ca.kps[j,[9,10]]>.3];b=tr.kpts[k,[9,10]][tr.kps[k,[9,10]]>.3]
    if not len(a) or not len(b):continue
    dist=float(np.linalg.norm(a[:,None]-b[None,:],axis=2).min()/sc.ref_h)
    if dist<.35:
     pts=np.concatenate([a,b]);pad=.3*sc.ref_h
     box=[max(0,pts[:,0].min()-pad),max(0,pts[:,1].min()-pad),min(sc.W,pts[:,0].max()+pad),min(sc.H,pts[:,1].max()+pad)]
     rows.append({'i':int(i),'t':float(sc.times[i]),'d':dist,'box':[float(x) for x in box]})
 if rows:
  # late contacts: transfer back to client would follow the initial receipt
  late=[r for r in rows if r['t']>=sc.times[-1]*.5]
  choice=max(late,key=lambda r:r['t']) if late else None
  out[name]=choice
  print(name,choice and (choice['t'],choice['d']))
Path('artifacts/checks-20261017/contacts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
