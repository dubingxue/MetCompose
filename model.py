"""Small shared graph encoder; flat-token vs factored-operator action scorers."""
import itertools,math
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from chemistry import token,enumerate_actions,execute,order
ELEMENTS=[6,7,8,9,15,16,17,35,53,5,14,34]
KINDS=['B','ADD','DEL','Q']

def onehot(x,values):return [float(x==v) for v in values]+[float(x not in values)]
def operator_features(t):
    kind,z1,z2,old,new=t
    return onehot(kind,KINDS)+onehot(z1,ELEMENTS)+onehot(z2,ELEMENTS)+onehot(old,list(range(-3,5)))+onehot(new,list(range(-3,5)))

def make_example(row,vocab,labels=True):
    parent=row['parent'];mol=Chem.MolFromSmiles(parent);x=[];edges=[];ef=[]
    for a in mol.GetAtoms():
        x.append(onehot(a.GetAtomicNum(),ELEMENTS)+[a.GetFormalCharge()/3,a.GetDegree()/4,a.GetTotalNumHs()/4,float(a.GetIsAromatic()),float(a.IsInRing())]+onehot(int(a.GetChiralTag()),[0,1,2]))
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();f=onehot(order(b),[1,2,3,4])+[float(b.GetIsConjugated()),float(b.IsInRing())]
        edges.extend([(i,j),(j,i)]);ef.extend([f,f])
    actions=enumerate_actions(parent,vocab);lookup={tuple(t):i for i,t in enumerate(vocab)}
    truth={tuple(a) for a in row.get('actions',[])} if labels else set()
    y=np.array([float(a in truth) for a in actions],dtype=np.float32)
    ec=row.get('ec3','unknown').split('.')[0];ec=int(ec) if ec.isdigit() and 1<=int(ec)<=7 else 0
    return dict(x=np.asarray(x,dtype=np.float32),edges=np.asarray(edges,dtype=np.int64).reshape(-1,2),ef=np.asarray(ef,dtype=np.float32).reshape(-1,7),sites=np.asarray([[a[1],a[2]] for a in actions],dtype=np.int64).reshape(-1,2),types=np.asarray([lookup[token(a)] for a in actions],dtype=np.int64),y=y,ec=ec,count=len(truth),actions=actions,parent=parent,reaction_id=row.get('reaction_id','inference'),target=row.get('product') if labels else None,action_coverage=(all(a in set(actions) for a in truth) if labels else None))

def batch(examples):
    xs=[];edges=[];efs=[];sites=[];types=[];ys=[];ng=[];ag=[];offset=0
    for k,e in enumerate(examples):
        xs.append(e['x']);edges.append(e['edges']+offset);efs.append(e['ef']);sites.append(e['sites']+offset);types.append(e['types']);ys.append(e['y'])
        ng.extend([k]*len(e['x']));ag.extend([k]*len(e['types']));offset+=len(e['x'])
    return dict(x=torch.tensor(np.concatenate(xs)),edges=torch.tensor(np.concatenate(edges)),ef=torch.tensor(np.concatenate(efs)),sites=torch.tensor(np.concatenate(sites)),types=torch.tensor(np.concatenate(types)),y=torch.tensor(np.concatenate(ys)),node_graph=torch.tensor(ng,dtype=torch.long),action_graph=torch.tensor(ag,dtype=torch.long),ec=torch.tensor([e['ec'] for e in examples]),counts=torch.tensor([max(1,min(3,e['count']))-1 for e in examples]),n=len(examples))

class EditModel(nn.Module):
    def __init__(self,vocab,variant='factor',hidden=48):
        super().__init__();self.variant=variant;self.hidden=hidden;self.vocab=vocab
        self.register_buffer('operator_descriptors',torch.tensor([operator_features(t) for t in vocab],dtype=torch.float32))
        self.input=nn.Linear(22,hidden)
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(hidden+7,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.LayerNorm(hidden)) for _ in range(2)])
        self.ec=nn.Embedding(8,8);self.environment=nn.Sequential(nn.Linear(hidden*3+8,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.count=nn.Sequential(nn.Linear(hidden+8,hidden),nn.SiLU(),nn.Linear(hidden,3))
        if variant=='flat':
            self.op=nn.Embedding(len(vocab),hidden);self.score=nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,1))
        else:
            self.op=nn.Sequential(nn.Linear(self.operator_descriptors.shape[1],hidden),nn.SiLU(),nn.Linear(hidden,hidden))
            self.opbias=nn.Linear(hidden,1);self.envbias=nn.Linear(hidden,1)

    def forward(self,b):
        h=self.input(b['x']);src,dst=b['edges'].T
        degree=torch.bincount(dst,minlength=len(h)).float().clamp(min=1)[:,None]
        for msg,update in zip(self.messages,self.updates):
            aggregate=torch.zeros_like(h);aggregate.index_add_(0,dst,msg(torch.cat([h[src],b['ef']],1)))
            h=h+update(torch.cat([h,aggregate/degree],1))
        pooled=torch.zeros((b['n'],self.hidden),dtype=h.dtype);pooled.index_add_(0,b['node_graph'],h)
        pooled=pooled/torch.bincount(b['node_graph'],minlength=b['n'])[:,None]
        ec=self.ec(b['ec'] if self.variant!='factor_noec' else torch.zeros_like(b['ec']))
        i,j=b['sites'].T;g=b['action_graph']
        env=self.environment(torch.cat([h[i]+h[j],torch.abs(h[i]-h[j]),pooled[g],ec[g]],1))
        if self.variant=='flat':
            ops=self.op.weight;scores=self.score(torch.cat([env,ops[b['types']]],1)).squeeze(1)
        else:
            ops=self.op(self.operator_descriptors);o=ops[b['types']]
            scores=(env*o).sum(1)/math.sqrt(self.hidden)+self.opbias(o).squeeze(1)+self.envbias(env).squeeze(1)
        count=self.count(torch.cat([pooled,ec],1))
        return scores,count,env,ops

def loss(model,b):
    scores,counts,env,ops=model(b);terms=[]
    for g in range(b['n']):
        mask=b['action_graph']==g;pos=mask&(b['y']>0);neg=mask&(b['y']==0)
        if pos.any():terms.append(nn.functional.softplus(-scores[pos]).mean()+nn.functional.softplus(scores[neg]).mean() if neg.any() else nn.functional.softplus(-scores[pos]).mean())
    value=torch.stack(terms).mean()+.3*nn.functional.cross_entropy(counts,b['counts'])
    if model.variant in ['factor','factor_noec']:
        positive=b['y']>0
        if positive.any():value=value+.1*nn.functional.cross_entropy(env[positive]@ops.T/math.sqrt(model.hidden),b['types'][positive])
    return value

@torch.no_grad()
def predict(model,example,max_attempts=128,max_candidates=64):
    model.eval();b=batch([example]);scores,counts,_,_=model(b)
    values=scores.numpy();rank=np.argsort(-values,kind='stable')[:16]
    countlog=counts.log_softmax(1)[0].numpy();programs=[]
    for k in range(1,min(3,len(rank))+1):
        combinations=list(itertools.combinations(rank,k));s=np.array([sum(values[i] for i in c) for c in combinations],float)
        normalizer=float(np.logaddexp.reduce(s))
        programs.extend((float(v-normalizer+countlog[k-1]),tuple(int(i) for i in c)) for c,v in zip(combinations,s))
    programs.sort(key=lambda x:(-x[0],x[1]));predictions=[];seen=set();attempts=0;valid=0;program_evidence=[]
    for score,ids in programs[:max_attempts]:
        attempts+=1;actions=[example['actions'][i] for i in ids]
        try:p=execute(example['parent'],actions)
        except Exception:continue
        valid+=1
        if p in seen or p==example['parent']:continue
        seen.add(p);predictions.append(p);program_evidence.append(actions)
        if len(predictions)>=max_candidates:break
    return dict(reaction_id=example['reaction_id'],predictions=predictions,programs=program_evidence,attempts=attempts,valid_programs=valid,action_pool_coverage=example['action_coverage'])
