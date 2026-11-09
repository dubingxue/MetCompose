"""Fixed-schedule CPU experiment; sealed targets never loaded here."""
import argparse,json,gzip,time,random,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from model import EditModel,make_example,batch,loss,predict
ROOT=Path(__file__).resolve().parent
VARIANTS=['flat','factor','factor_noalign','factor_noec']
def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def save(path,rows):
    tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)

def evaluate(model,rows,examples):
    predictions=[]
    for row,example in zip(rows,examples):
        result=predict(model,example);result['target']=row['product'];result['program_family']=row['primitive_program']
        result.update({f'top{k}':row['product'] in result['predictions'][:k] for k in [1,5,10]});result['oracle']=row['product'] in result['predictions'];predictions.append(result)
    metrics={key:sum(r[key] for r in predictions)/len(predictions) for key in ['top1','top5','top10','oracle','action_pool_coverage']}
    metrics.update(n=len(predictions),valid_program_fraction=sum(r['valid_programs'] for r in predictions)/max(1,sum(r['attempts'] for r in predictions)),mean_candidates=float(np.mean([len(r['predictions']) for r in predictions])))
    return predictions,metrics

def optimize(model,examples,epochs,seed,lr=.002,weights=None):
    rng=np.random.RandomState(seed);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);history=[]
    for epoch in range(epochs):
        model.train();indices=rng.choice(len(examples),len(examples),replace=True,p=weights) if weights is not None else rng.permutation(len(examples));losses=[]
        for j in range(0,len(indices),24):
            b=batch([examples[i] for i in indices[j:j+24]]);opt.zero_grad();value=loss(model,b)
            if not torch.isfinite(value):raise ValueError('nonfinite_loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(value.detach()))
        history.append(float(np.mean(losses)));print('epoch',epoch+1,round(history[-1],4),flush=True)
    return history

def main(seed):
    seed_all(seed);out=ROOT/'models'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    vocab=json.loads((ROOT/'splits/vocab.json').read_text());trainrows=read(ROOT/'splits/train.jsonl.gz');devrows=read(ROOT/'splits/development.jsonl.gz')
    trainex=[make_example(r,vocab) for r in trainrows];devex=[make_example(r,vocab) for r in devrows]
    assert all(e['action_coverage'] and e['y'].sum()>0 for e in trainex)
    family=[json.dumps(r['primitive_program']) for r in trainrows];freq=Counter(family);weights=np.array([1/np.sqrt(freq[f]) for f in family]);weights/=weights.sum()
    config=dict(seed=seed,epochs=12,batch_size=24,hidden=48,message_layers=2,lr=.002,variants=VARIANTS,train_n=len(trainrows),development_n=len(devrows),train_sha256=sha(ROOT/'splits/train.jsonl.gz'),development_sha256=sha(ROOT/'splits/development.jsonl.gz'),vocab_sha256=sha(ROOT/'splits/vocab.json'),model_sha256=sha(ROOT/'model.py'),chemistry_sha256=sha(ROOT/'chemistry.py'),runner_sha256=sha(Path(__file__)),decode=dict(top_actions=16,max_edits=3,max_program_attempts=128,max_candidates=64),negative_label_note='unrecorded edits are benchmark contrasts, not verified biological negatives')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for variant in VARIANTS:
        seed_all(seed);model=EditModel(vocab,variant);start=time.time();print('TRAIN',variant,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        history=optimize(model,trainex,12,seed,weights=weights)
        torch.save({'state':model.state_dict(),'vocab':vocab,'variant':variant,'seed':seed},out/(variant+'.pt'))
        predictions,metrics=evaluate(model,devrows,devex)
        save(out/(variant+'_development.jsonl.gz'),predictions)
        (out/(variant+'_metrics.json')).write_text(json.dumps(dict(metrics=metrics,history=history,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start),indent=2)+'\n')
        print('RESULT',variant,metrics,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.seed)
