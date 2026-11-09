"""Prospectively frozen retrospective family-held-out CV; fresh fits only."""
from pathlib import Path
import sys,json,hashlib,time,pickle,itertools,csv
from collections import Counter,defaultdict
H=Path(__file__).resolve().parent;E=H.parent
sys.path[:0]=[str(E/'edit_v1'),str(E/'edit_v2'),str(E/'edit_v8_submission')]
import numpy as np,torch
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem,rdChemReactions
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from model import EditModel,make_example,batch,loss
from train import seed_all
from chemistry import token,execute,canonical
from compatibility import pair_features,compatible
RDLogger.DisableLog('rdApp.*');torch.set_num_threads(1)
read=lambda p:[json.loads(x) for x in Path(p).open() if x.strip()]
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2))
def sig(r):return tuple(sorted(tuple(a) for a in r['primitive_program']))
def structure(s):return canonical(s,False)
SOURCE=E/'edit_v10_data_expansion/baselines/any_role_disjoint_common_train.jsonl'
rows=read(SOURCE);families=Counter(sig(r) for r in rows if len(r['actions'])>1)
# Greedy count balancing without model outcomes.
folds=[[],[],[]];loads=[0,0,0]
for g,n in sorted(families.items(),key=lambda t:(-t[1],t[0])):
 f=min(range(3),key=lambda x:(loads[x],x));folds[f].append(g);loads[f]+=n
splits=[]
for f,gs in enumerate(folds):
 rawtest=[r for r in rows if sig(r) in gs]
 blocked={structure(r[k]) for r in rawtest for k in ['parent','product']}
 ids={i for r in rawtest for i in r.get('evidence_reaction_ids',[r['reaction_id']])}
 train=[r for r in rows if sig(r) not in gs and not any(structure(r[k]) in blocked for k in ['parent','product']) and not ids.intersection(r.get('evidence_reaction_ids',[r['reaction_id']]))]
 primitives={a for r in train for a in sig(r)}
 test=[r for r in rawtest if all(a in primitives for a in sig(r))]
 audit=dict(fold=f,train=len(train),held_records=len(rawtest),supported_records=len(test),unsupported_records=len(rawtest)-len(test),supported_families=len({sig(r) for r in test}),parents=len({r['parent'] for r in test}),removed_overlap=len(rows)-len(rawtest)-len(train))
 assert not {sig(r) for r in train}&{sig(r) for r in test}
 splits.append((train,test,audit))
protocol=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),folds=3,seeds=[41,53,67],epochs=12,selection='fixed final epoch; no held-out checkpoint selection',split='entire primitive multisets withheld; any-role stereo-free molecule overlap and evidence IDs removed from training',exclusions='unsupported primitives reported, excluded only from supported-combination estimand',modes=['base','hard','full','no_relation','no_count'],caps=[32,64,128],local_actions=64,products=64,max_edits=3,scope='retrospective cross-validation within historical training pool, not new independent external evidence',audits=[x[2] for x in splits],pair_controls='unrecorded compatible pairs; not verified biological negatives',retrieval='not refitted; previous fold-specific retrieval unchanged')
if (H/'protocol.json').exists():assert json.loads((H/'protocol.json').read_text())==protocol
else:write(H/'protocol.json',protocol)
print('AUDIT',protocol['audits'],flush=True)
class Truncated:
 def __init__(self,c):self.c=c
 def decision_function(self,x):return self.c.decision_function(x[:,:98])
def fitpairs(train,vocab):
 rng=np.random.RandomState(17029);xx=[];yy=[]
 for row in train:
  ex=make_example(row,vocab,labels=False);aa=ex['actions'];look={tuple(a):i for i,a in enumerate(aa)};truth=[look[tuple(a)] for a in row['actions']];pos={tuple(sorted(p)) for p in itertools.combinations(truth,2)};neg=set()
  for _ in range(128):
   if len(aa)<2:break
   i=int(rng.choice(truth)) if rng.rand()<.5 else int(rng.randint(len(aa)));j=int(rng.randint(len(aa)));p=tuple(sorted((i,j)))
   if i!=j and p not in pos and compatible(aa[i],aa[j]):neg.add(p)
   if len(neg)>=24:break
  pp=sorted(pos)+sorted(neg)
  if pp:xx.append(pair_features(aa,Chem.GetDistanceMatrix(Chem.MolFromSmiles(row['parent'])),pp));yy.extend([1]*len(pos)+[0]*len(neg))
 x=np.concatenate(xx);full=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=500,random_state=17));short=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=500,random_state=17));full.fit(x,yy);short.fit(x[:,:98],yy)
 return full,Truncated(short),dict(positive=int(sum(yy)),contrasts=len(yy)-sum(yy))
def decode(ex,values,prior,clf,mode):
 rank=np.argsort(-values,kind='stable')[:64];aa=[ex['actions'][i] for i in rank];n=len(aa);pairs=list(itertools.combinations(range(n),2));ok=np.ones((n,n),bool);bonus=np.zeros((n,n))
 if pairs:
  xx=pair_features(aa,Chem.GetDistanceMatrix(Chem.MolFromSmiles(ex['parent'])),pairs)
  vv=np.clip(clf.decision_function(xx),-5,5) if mode not in ['base','hard'] else np.zeros(len(pairs))
  for (i,j),v in zip(pairs,vv):ok[i,j]=ok[j,i]=compatible(aa[i],aa[j]);bonus[i,j]=bonus[j,i]=v
 groups=[]
 for k in range(1,min(3,n)+1):
  ix=np.array(list(itertools.combinations(range(n),k)));sc=values[rank[ix]].sum(1).astype(float);mask=np.ones(len(ix),bool);bs=np.zeros(len(ix))
  for a,b in itertools.combinations(range(k),2):mask&=ok[ix[:,a],ix[:,b]];bs+=bonus[ix[:,a],ix[:,b]]
  if mode!='base':ix=ix[mask];sc=sc[mask];bs=bs[mask]
  if not len(ix):continue
  sc+=bs/max(1,k*(k-1)//2);sc=sc-np.logaddexp.reduce(sc)+(np.log(1/3) if mode=='no_count' else prior[k-1]);groups.extend((float(s),tuple(int(rank[j]) for j in z)) for s,z in zip(sc,ix))
 groups.sort(key=lambda z:(-z[0],z[1]));pred=[];progs=[];at=[];seen=set();valid=0;attempts=0
 for attempts,(_,ids) in enumerate(groups[:128],1):
  acts=[ex['actions'][j] for j in ids]
  try:p=execute(ex['parent'],acts)
  except Exception:continue
  valid+=1
  if p==ex['parent'] or p in seen:continue
  seen.add(p);pred.append(p);progs.append(acts);at.append(attempts)
  if len(pred)==64:break
 return dict(predictions=pred,programs=progs,at=at,attempts=attempts,valid=valid,top_actions=[list(a) for a in aa])
def template(r):
 l=Chem.MolFromSmiles(r['mapped_parent']);q=Chem.MolFromSmiles(r['mapped_product']);la={a.GetAtomMapNum():a for a in l.GetAtoms()};ra={a.GetAtomMapNum():a for a in q.GetAtoms()}
 def bonds(m):return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))):str(b.GetBondType()) for b in m.GetBonds()}
 lb,rb=bonds(l),bonds(q);cen=set(la)^set(ra)
 for t in set(lb)|set(rb):
  if lb.get(t)!=rb.get(t):cen.update(t)
 for t in set(la)&set(ra):
  if (la[t].GetFormalCharge(),la[t].GetTotalNumHs())!=(ra[t].GetFormalCharge(),ra[t].GetTotalNumHs()):cen.add(t)
 ctx=set(cen)
 for atoms in [la,ra]:
  for t in cen&set(atoms):ctx.update(a.GetAtomMapNum() for a in atoms[t].GetNeighbors())
 try:
  ls=Chem.MolFragmentToSmarts(l,[a.GetIdx() for t,a in la.items() if t in ctx],isomericSmarts=True);rs=Chem.MolFragmentToSmarts(q,[a.GetIdx() for t,a in ra.items() if t in ctx],isomericSmarts=True);rx=rdChemReactions.ReactionFromSmarts('('+ls+')>>('+rs+')');rx.Initialize();return rx
 except Exception:return None
for fi,(train,test,audit) in enumerate(splits):
 out=H/f'fold{fi}';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():continue
 for name,rr in [('train',train),('test',test)]: (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
 vocab=sorted({token(a) for r in train for a in r['actions']});full,short,stats=fitpairs(train,vocab);write(out/'pair_fit.json',stats)
 exs=[make_example(r,vocab) for r in train];assert all(x['action_coverage'] for x in exs)
 for seed in [41,53,67]:
  final=out/f'predictions_seed{seed}.jsonl'
  if final.exists():continue
  checkpoint=out/f'model_seed{seed}.pt';history_path=out/f'history_seed{seed}.json'
  if checkpoint.exists() and history_path.exists():
   history=json.loads(history_path.read_text());assert len(history)==12 and history[-1]['epoch']==12
   cp=torch.load(checkpoint,map_location='cpu');assert cp['vocab']==vocab
   model=EditModel(vocab,'flat');model.load_state_dict(cp['state']);print('RESUME_DECODE',fi,seed,flush=True)
  else:
   seed_all(seed);model=EditModel(vocab,'flat');opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4);freq=Counter(sig(r) for r in train);w=np.array([1/np.sqrt(freq[sig(r)]) for r in train]);w/=w.sum();rng=np.random.RandomState(seed);history=[]
   for ep in range(1,13):
    model.train();inds=rng.choice(len(exs),len(exs),replace=True,p=w);ll=[];t=time.time()
    for j in range(0,len(inds),24):
     b=batch([exs[i] for i in inds[j:j+24]]);opt.zero_grad();v=loss(model,b);assert torch.isfinite(v);v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ll.append(float(v.detach()))
    history.append(dict(epoch=ep,loss=float(np.mean(ll)),seconds=time.time()-t));print('TRAIN',fi,seed,ep,round(history[-1]['loss'],4),flush=True)
   write(out/f'history_seed{seed}.json',history);torch.save(dict(state=model.state_dict(),vocab=vocab),out/f'model_seed{seed}.pt')
  model.eval();res=[];pairscores=[]
  for ri,r in enumerate(test):
   ex=make_example(dict(parent=r['parent'],ec3='unknown'),vocab,labels=False)
   with torch.no_grad():sc,co,_,_=model(batch([ex]))
   values=sc.numpy();prior=co.log_softmax(1)[0].numpy()
   for mode in protocol['modes']:
    z=decode(ex,values,prior,short if mode=='no_relation' else full,mode)
    for p,acts in zip(z['predictions'],z['programs']):assert execute(r['parent'],acts)==p
    z.update(reaction_id=r['reaction_id'],parent=r['parent'],target=r['product'],fold=fi,seed=seed,mode=mode,family=json.dumps(sig(r)),truth_actions=r['actions']);res.append(z)
   if (ri+1)%10==0:print('DECODE',fi,seed,ri+1,len(test),flush=True)
  final.write_text(''.join(json.dumps(r)+'\n' for r in res));write(out/f'pair_scores_seed{seed}.json',pairscores)
 write(out/'complete.json',dict(status='complete',audit=audit))
write(H/'complete.json',dict(status='complete',folds=3,seeds=3))
