import sys,json,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,'scripts')
from common import load_labels
from app.ml.semantic import ClipEmbedder,probe_scores
probes={
 'cash_returned_to_client':(['a bank teller handing a stack of banknotes back to a customer','a cashier giving money to a customer'],['a customer giving money to a cashier','a cashier returning an identity card to a customer','a bank teller counting money at a desk']),
 'collection_bag_emptied_no_count':(['a woman putting money from a cash collection bag into a safe','a bank teller emptying a money bag into a cabinet'],['a bank teller counting banknotes by hand','a bank teller using a banknote counting machine','a woman sitting at a desk']),
 'safe_key_left':(['a key left in the lock of a metal safe','a metal cabinet with a key in its door lock'],['a closed metal cabinet without a key','a woman locking a safe','a bank office desk']),
 'rejects_merged':(['a woman taking rejected banknotes from a counting machine and putting them into a stack of money'],['a woman checking banknotes with an ultraviolet detector','a woman counting banknotes','a woman sorting banknotes by denomination']),
}
clip=ClipEmbedder(threads=6)
bank={k:(clip.texts(p),clip.texts(n)) for k,(p,n) in probes.items()}
labels=load_labels();out={}
for name,lab in sorted(labels.items()):
 d=pickle.load(open(Path('data/cache')/(name+'.pkl'),'rb')); times=np.array([o.t for o in d['obs']])
 scores=probe_scores(d['clip_wa'],bank)
 out[name]={}
 for k,a in scores.items():
  smooth=np.convolve(a,np.ones(3)/3,mode='same'); i=int(smooth.argmax())
  out[name][k]={'max':float(smooth[i]),'t':float(times[i]),'near':max([float(smooth[np.abs(times-t)<8].max()) for t,typ in lab['types'] if typ==k],default=None)}
Path('artifacts/checks-20261017/probes.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for k in probes:
 print('\n',k)
 for n in sorted(out,key=lambda n:out[n][k]['max'],reverse=True)[:12]: print(n,labels[n]['label'],out[n][k])
