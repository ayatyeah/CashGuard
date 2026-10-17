import sys,json,time
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,'scripts')
from common import load_video,CACHE
from app.ml.vlm import VLM
from app.ml.candidates import crop_scaled
vlm=VLM(threads=8); contacts=json.load(open('artifacts/checks-20261017/contacts.json'))
q='Three close-up CCTV frames in time order, left to right. Are banknotes visibly being handed from the bank teller to the customer? Distinguish a bundle of cash from an identity card or paper document. Answer JSON with boolean keys "cash_bundle_visible", "cash_handed_to_customer", "card_or_document_handover".'
for name in sys.argv[1:]:
 c=contacts.get(name)
 if not c:continue
 d,_=load_video(name,with_frames=False);ts=np.array([o.t for o in d['obs']]);idx=[int(np.argmin(abs(ts-c['t']-offset))) for offset in (-3,-1,0)]
 ims=[crop_scaled(cv2.imread(str(Path(CACHE)/'frames'/name/f'{d["obs"][i].frame_id:05d}.jpg')),c['box'],1.) for i in idx]
 start=time.monotonic();ans=vlm.ask_json(ims,q,max_new_tokens=80)
 result={'times':[float(ts[i]) for i in idx],'answer':ans,'seconds':time.monotonic()-start}
 Path(f'artifacts/checks-20261017/{name}-closeup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(name,result,flush=True)
