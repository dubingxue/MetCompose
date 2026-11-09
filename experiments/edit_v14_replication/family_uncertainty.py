"""Family-cluster sensitivity, conditional on frozen five-seed fits."""
from pathlib import Path
import csv,json
import numpy as np
H=Path(__file__).resolve().parent;A=H/'analysis'
assert (A/'metcompose_complete.json').exists()
read=lambda p:list(csv.DictReader(p.open()))
raw=[r for r in read(A/'all_ranks.csv') if int(r['cap'])==128]
meta=read(H/'composition_audit/heldout_graph_classes.csv')
ids=[r['reaction_id'] for r in meta]; families=sorted({r['family'] for r in meta}); seeds=[17,29,41,53,67]
lookup={(r['reaction_id'],int(r['seed']),r['method']):int(r['rank']) for r in raw}
groups=[np.array([i for i,r in enumerate(meta) if r['family']==fam]) for fam in families]
out=[]
for k in [1,10,64]:
 for method in ['hard','base','no_relation','no_count']:
  delta=np.array([[int(0<lookup[i,s,'full']<=k)-int(0<lookup[i,s,method]<=k) for s in seeds] for i in ids])
  totals=np.array([delta[g].mean(1).sum() for g in groups]);sizes=np.array([len(g) for g in groups]);means=totals/sizes
  rng=np.random.default_rng(20261105);ix=rng.integers(0,len(groups),(10000,len(groups)))
  micro=totals[ix].sum(1)/sizes[ix].sum(1)*100;macro=means[ix].mean(1)*100
  for weighting,estimate,bs in [('Substrate-weighted',delta.mean()*100,micro),('Family-weighted',means.mean()*100,macro)]:
   out.append(dict(k=k,comparator=method,weighting=weighting,difference=estimate,lower=np.quantile(bs,.025),upper=np.quantile(bs,.975),n_families=len(groups),n_substrates=len(ids)))
with (A/'family_cluster_sensitivity.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(A/'family_cluster_protocol.json').write_text(json.dumps({'status':'complete','scope':'Post-hoc sensitivity to dependence within held-out primitive multisets; no data exclusions or model tuning','resamples':10000,'seed':20261105,'resampling_unit':'whole primitive-multiset family','uncertainty':'conditional on five fitted seeds and observed family universe; not new dataset validation','weighting':'micro retains each sampled family size; macro gives each sampled family equal weight','multiplicity':'no significance stars or corrected inference'},indent=2))
for r in out:
 if r['k']==10:print(r)
