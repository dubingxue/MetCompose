"""Verified aggregation of all fixed experiments. Incomplete pools never released."""
import json,csv,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
H=Path(__file__).resolve().parent;OLD=H.parent/'edit_v12_figure3';O=H/'analysis';O.mkdir(exist_ok=True)
SEEDS=[17,29,41,53,67];MODES=['base','hard','full','no_relation','no_count']
def read(p):return [json.loads(l) for l in p.open()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x): (O/n).write_text(json.dumps(x,indent=2)+'\n')
def csvout(n,rr):
 if not rr:return
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
tests={f:read(OLD/f'fold{f}/test.jsonl') for f in range(3)};meta={r['reaction_id']:dict(r,fold=f) for f,rr in tests.items() for r in rr}
assert len(meta)==133 and len({r['parent'] for r in meta.values()})==133
loaded={};sources={};missing=[]
for f in range(3):
 for seed in SEEDS:
  root=OLD if seed in [17,29] else H;p=root/f'fold{f}/predictions_seed{seed}.jsonl'
  if not p.exists():missing.append([f,seed]);continue
  for name in ['train.jsonl','test.jsonl']:assert sha(root/f'fold{f}'/name)==sha(OLD/f'fold{f}'/name),(f,seed,name)
  try:rr=read(p)
  except json.JSONDecodeError:missing.append([f,seed]);continue
  expected={(r['reaction_id'],m) for r in tests[f] for m in MODES};found={(r['reaction_id'],r['mode']) for r in rr}
  assert len(rr)==len(expected) and found==expected,(f,seed,'incomplete_or_duplicate')
  for r in rr:
   assert r['target']==meta[r['reaction_id']]['product'] and r['parent']==meta[r['reaction_id']]['parent']
   assert r['seed']==seed and r['fold']==f and len(r['predictions'])==len(r['at'])
   assert len(r['predictions'])==len(set(r['predictions']))<=64 and r['attempts']<=128
  loaded[f,seed]=rr;sources[str(p.relative_to(H.parent))]=sha(p)
status=dict(completed_metcompose_fits=len(loaded),required_metcompose_fits=15,missing_metcompose_fits=missing,completed_megan_folds=[f for f in range(3) if (H/f'megan/fold{f}/complete.json').exists()])
save('status.json',status)
ids=sorted(meta);families={i:json.dumps(sorted(meta[i]['primitive_program'])) for i in ids};fg=sorted(set(families.values()));assert len(fg)==25
rng=np.random.default_rng(20261105);boot=rng.integers(0,len(ids),(4000,len(ids)))
def ci(a):return list(np.quantile(a[boot].mean(1)*100,[.025,.975]))
if not missing:
 arr={};rankrows=[]
 for (_,seed),rr in loaded.items():
  for r in rr:
   for cap in [32,64,128]:
    pp=[p for p,a in zip(r['predictions'],r['at']) if a<=cap];rank=pp.index(r['target'])+1 if r['target'] in pp else 0
    arr[r['reaction_id'],seed,r['mode'],cap]=rank
    rankrows.append(dict(reaction_id=r['reaction_id'],fold=r['fold'],seed=seed,method=r['mode'],cap=cap,candidates=len(pp),rank=rank))
 csvout('all_ranks.csv',rankrows)
 metrics=[];paired=[];seedrows=[];famrows=[];loo=[]
 for cap in [32,64,128]:
  for k in [1,10,64]:
   hit={m:np.array([[int(0<arr[i,s,m,cap]<=k) for s in SEEDS] for i in ids]) for m in MODES}
   for m,a in hit.items():
    lo,hi=ci(a.mean(1));metrics.append(dict(method=m,cap=cap,k=k,recall=100*a.mean(),lower=lo,upper=hi,n_parents=len(ids),n_seeds=5,seed_min=100*a.mean(0).min(),seed_max=100*a.mean(0).max()))
    for si,s in enumerate(SEEDS):seedrows.append(dict(method=m,cap=cap,k=k,seed=s,recall=100*a[:,si].mean()))
   for m in ['base','hard','no_relation','no_count']:
    a=hit['full']-hit[m];lo,hi=ci(a.mean(1));paired.append(dict(comparator=m,cap=cap,k=k,difference=100*a.mean(),lower=lo,upper=hi,seed_min=100*a.mean(0).min(),seed_max=100*a.mean(0).max()))
   if cap==128:
    for family in fg:
     sel=[j for j,i in enumerate(ids) if families[i]==family]
     famrows.append(dict(family=family,n=len(sel),k=k,full=100*hit['full'][sel].mean(),hard=100*hit['hard'][sel].mean(),difference=100*(hit['full'][sel]-hit['hard'][sel]).mean()))
     keep=[j for j,i in enumerate(ids) if families[i]!=family]
     loo.append(dict(omitted_family=family,n_remaining=len(keep),k=k,difference=100*(hit['full'][keep]-hit['hard'][keep]).mean()))
 for n,rr in [('metrics.csv',metrics),('paired_effects.csv',paired),('seed_results.csv',seedrows),('family_effects.csv',famrows),('leave_one_family_out.csv',loo)]:csvout(n,rr)
 save('metcompose_complete.json',dict(status='complete',seeds=SEEDS,n_parents=133,n_families=25,ci='4000 paired parent-cluster resamples, conditional on five fitted seeds; seed range separately reported',source_sha256=sources))
if len(status['completed_megan_folds'])==3:
 ranks={};errs=[];msources={}
 for f in range(3):
  root=H/f'megan/fold{f}';p=root/'validation_epoch12.jsonl';protocol=json.loads((root/'protocol.json').read_text())
  assert protocol['train_sha']==sha(OLD/f'fold{f}/train.jsonl') and protocol['validation_sha']==sha(OLD/f'fold{f}/test.jsonl')
  rr=read(p);assert len(rr)==len(tests[f])
  for q in rr:
   assert len(q['reaction_ids'])==1;i=q['reaction_ids'][0];assert meta[i]['fold']==f
   ranks[i]=q['predictions'].index(meta[i]['product'])+1 if meta[i]['product'] in q['predictions'] else 0
   if q['status']!='ok':errs.append(dict(reaction_id=i,error=q.get('error','unknown')))
  msources[str(p.relative_to(H.parent))]=sha(p)
 assert set(ranks)==set(ids)
 # Primary paired architectural comparison uses the identical seed 17.
 mc={r['reaction_id']:r for f in range(3) for r in loaded[f,17] if r['mode']=='full'}
 comparison=[];rs=[]
 for i in ids:
  pp=mc[i]['predictions'];rt=pp.index(meta[i]['product'])+1 if meta[i]['product'] in pp else 0
  rs.append(dict(reaction_id=i,fold=meta[i]['fold'],metcompose_seed17_rank=rt,megan_seed17_rank=ranks[i]))
 for k in [1,10,64]:
  x=np.array([int(0<r['metcompose_seed17_rank']<=k) for r in rs]);y=np.array([int(0<r['megan_seed17_rank']<=k) for r in rs]);lo,hi=ci(x-y)
  comparison.append(dict(k=k,metcompose_seed17=100*x.mean(),megan_seed17=100*y.mean(),difference_pp=100*(x-y).mean(),lower=lo,upper=hi,n=133))
 csvout('megan_paired_ranks.csv',rs);csvout('megan_paired_comparison.csv',comparison)
 save('megan_complete.json',dict(status='complete',seed=17,epochs=12,source_sha256=msources,inference_errors=errs,scope='Fixed epoch, common training folds; no convergence claim or FLOP matching. One fitted seed per architecture in primary paired comparison.'))
print(json.dumps(status))
