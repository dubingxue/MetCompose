"""Train-only pair compatibility and fixed-budget development ablations."""
import sys, itertools, json, time, pickle
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent
# Core modules are located together in this release.
from model import EditModel, make_example, batch, operator_features
from chemistry import token, execute
from train import read,save,sha
SOURCE=HERE.parent/'edit_v1'

def compatible(a,b):
    if a[:3]==b[:3]: return False
    if a[0]=='B' and b[0]=='B' and set(a[1:3])==set(b[1:3]): return False
    if a[0]=='DEL' and a[2] in b[1:3]: return False
    if b[0]=='DEL' and b[2] in a[1:3]: return False
    return True

def pair_features(actions,dist,pairs):
    desc=np.asarray([operator_features(token(a)) for a in actions])
    result=[]
    for i,j in pairs:
        a,b=actions[i],actions[j];sa,sb=set(a[1:3]),set(b[1:3])
        d=min(dist[x,y] for x in sa for y in sb)
        da,db=a[6]-a[5],b[6]-b[5]
        result.append(np.r_[desc[i]+desc[j],abs(desc[i]-desc[j]),len(sa&sb),min(d,10)/10,da+db,abs(da-db),da*db])
    return np.asarray(result)

def fit(rows,vocab):
    rng=np.random.RandomState(17029);xx=[];yy=[];positive=0
    for row in rows:
        ex=make_example(row,vocab,labels=False);actions=ex['actions'];lookup={tuple(a):i for i,a in enumerate(actions)}
        truth=[lookup[tuple(a)] for a in row['actions']]
        pos={tuple(sorted(p)) for p in itertools.combinations(truth,2)}
        assert all(compatible(actions[i],actions[j]) for i,j in pos)
        neg=set()
        for _ in range(128):
            if len(actions)<2:break
            i=int(rng.choice(truth)) if rng.rand()<.5 else int(rng.randint(len(actions)))
            j=int(rng.randint(len(actions)));p=tuple(sorted((i,j)))
            if i!=j and p not in pos and compatible(actions[i],actions[j]):neg.add(p)
            if len(neg)>=24:break
        pairs=sorted(pos)+sorted(neg)
        if not pairs:continue
        dist=Chem.GetDistanceMatrix(Chem.MolFromSmiles(ex['parent']))
        xx.append(pair_features(actions,dist,pairs));yy.extend([1]*len(pos)+[0]*len(neg));positive+=len(pos)
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=500,random_state=17))
    clf.fit(np.concatenate(xx),yy)
    return clf,dict(training_rows=len(rows),positive_pairs=positive,contrast_pairs=len(yy)-positive,features=xx[0].shape[1],note='Unrecorded combinations are contrast labels, not verified biological negatives; encoder and scorer use same training split, no independent calibration claim')

def generate(ex,values,countlog,clf,mode):
    rank=np.argsort(-values,kind='stable')[:64];actions=[ex['actions'][i] for i in rank];n=len(rank)
    pairs=list(itertools.combinations(range(n),2));ok=np.ones((n,n),bool);compat=np.zeros((n,n))
    if pairs:
        distances=Chem.GetDistanceMatrix(Chem.MolFromSmiles(ex['parent']))
        learned=np.clip(clf.decision_function(pair_features(actions,distances,pairs)),-5,5)
        for (i,j),v in zip(pairs,learned):
            ok[i,j]=ok[j,i]=compatible(actions[i],actions[j]);compat[i,j]=compat[j,i]=v
    groups=[];rejected=0
    for k in range(1,min(3,n)+1):
        ids=np.asarray(list(itertools.combinations(range(n),k)),dtype=int)
        scores=values[rank[ids]].sum(1).astype(float)
        mask=np.ones(len(ids),bool);bonus=np.zeros(len(ids))
        for a,b in itertools.combinations(range(k),2):
            mask &= ok[ids[:,a],ids[:,b]];bonus+=compat[ids[:,a],ids[:,b]]
        if mode!='base64':
            rejected+=int((~mask).sum());ids=ids[mask];scores=scores[mask];bonus=bonus[mask]
        if not len(ids):groups.append([]);continue
        if mode.startswith('learned'):scores+=bonus/max(1,k*(k-1)//2)
        scores=scores-float(np.logaddexp.reduce(scores))+countlog[k-1]
        group=[(float(s),tuple(int(rank[i]) for i in ix)) for s,ix in zip(scores,ids)]
        group.sort(key=lambda p:(-p[0],p[1]));groups.append(group)
    if mode.endswith('quota'):
        # Round robin across edit lengths prevents a length prior from starving a class.
        programlist=[]
        for step in range(128):
            for group in groups:
                if step<len(group):programlist.append(group[step])
        programlist=programlist[:128]
    else:programlist=sorted(sum(groups,[]),key=lambda p:(-p[0],p[1]))[:128]
    predictions=[];programs=[];seen=set();attempts=0;valid=0
    for score,ids in programlist:
        attempts+=1;edits=[ex['actions'][i] for i in ids]
        try:p=execute(ex['parent'],edits)
        except Exception:continue
        valid+=1
        if p==ex['parent'] or p in seen:continue
        seen.add(p);predictions.append(p);programs.append(edits)
        if len(predictions)>=64:break
    return dict(predictions=predictions,programs=programs,attempts=attempts,valid=valid,rejected_conflicting_sets=rejected)

def main():
    torch.set_num_threads(1);out=HERE/'compatibility';out.mkdir(exist_ok=True)
    modes=['base64','filter64','learned64','filter_quota','learned_quota']
    protocol=dict(modes=modes,seeds=[17,29],pool=64,max_attempts=128,max_products=64,pair_weight=1,logit_clip=5,train_sha=sha(SOURCE/'splits/train.jsonl.gz'),dev_sha=sha(SOURCE/'splits/development.jsonl.gz'),runner_sha=sha(Path(__file__)),scope='development only; no sealed/domain reads; quota output uses round-robin order')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    vocab=json.loads((SOURCE/'splits/vocab.json').read_text());rows=read(SOURCE/'splits/train.jsonl.gz')
    clf,stats=fit(rows,vocab);(out/'training.json').write_text(json.dumps(stats,indent=2))
    with (out/'pair_scorer.pkl').open('wb') as f:pickle.dump(clf,f)
    print('TRAIN',stats,flush=True);dev=read(SOURCE/'splits/development.jsonl.gz');summary={}
    for seed in [17,29]:
        model=EditModel(vocab,'flat');model.load_state_dict(torch.load(SOURCE/'models'/f'seed{seed}'/'flat.pt',map_location='cpu')['state']);model.eval()
        outputs={m:[] for m in modes}
        baseline=read(HERE/f'flat_seed{seed}_width64.jsonl.gz')
        for r,old in zip(dev,baseline):
            ex=make_example(r,vocab,labels=False)
            with torch.no_grad():scores,counts,_,_=model(batch([ex]))
            for mode in modes:
                result=generate(ex,scores.numpy(),counts.log_softmax(1)[0].numpy(),clf,mode)
                if mode=='base64':assert result['predictions']==old['predictions']
                result.update(reaction_id=r['reaction_id'],target=r['product'],top1=r['product'] in result['predictions'][:1],top10=r['product'] in result['predictions'][:10],oracle=r['product'] in result['predictions'])
                outputs[mode].append(result)
        for mode,result in outputs.items():
            name=f'{mode}_seed{seed}';save(out/(name+'.jsonl.gz'),result)
            summary[name]={k:sum(x[k] for x in result)/len(result) for k in ['top1','top10','oracle','attempts','valid','rejected_conflicting_sets']}
            print(name,summary[name],flush=True)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
