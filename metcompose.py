"""CPU training and inference for the original flat-token MetCompose model."""
import argparse, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from collections import Counter
from chemistry import token
from model import EditModel, make_example, batch
from train import seed_all, optimize
from compatibility import fit, generate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--data',type=Path,required=True);train.add_argument('--output',type=Path,required=True);train.add_argument('--seed',type=int,default=17);train.add_argument('--epochs',type=int,default=12)
    pred=sub.add_parser('predict');pred.add_argument('--checkpoint',type=Path,required=True);pred.add_argument('--smiles',required=True);pred.add_argument('--ec',default='unknown');pred.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    if args.command=='train':
        rows=[json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
        seed_all(args.seed);vocab=sorted({token(a) for r in rows for a in r['actions']})
        ex=[make_example(r,vocab) for r in rows]
        if not all(e['action_coverage'] and e['y'].sum()>0 for e in ex):raise ValueError('Training actions must be enumerable.')
        fam=[json.dumps(r['primitive_program']) for r in rows];counts=Counter(fam);weights=np.array([1/np.sqrt(counts[f]) for f in fam]);weights/=weights.sum()
        model=EditModel(vocab,'flat');hist=optimize(model,ex,args.epochs,args.seed,weights=weights)
        pair,stats=fit(rows,vocab);args.output.mkdir(parents=True,exist_ok=True)
        torch.save({'state':model.state_dict(),'vocab':vocab},args.output/'model.pt')
        with (args.output/'pair.pkl').open('wb') as f:pickle.dump(pair,f)
        (args.output/'training.json').write_text(json.dumps({'seed':args.seed,'epochs':args.epochs,'history':hist,'pair_fit':stats},indent=2))
    else:
        cp=torch.load(args.checkpoint/'model.pt',map_location='cpu',weights_only=False)
        model=EditModel(cp['vocab'],'flat');model.load_state_dict(cp['state']);model.eval()
        with (args.checkpoint/'pair.pkl').open('rb') as f:pair=pickle.load(f)
        ex=make_example({'parent':args.smiles,'ec3':args.ec},cp['vocab'],labels=False)
        with torch.no_grad():scores,count,_,_=model(batch([ex]))
        result=generate(ex,scores.numpy(),count.log_softmax(1)[0].numpy(),pair,'learned64')
        args.output.write_text(json.dumps({'parent':args.smiles,**result},indent=2))
        print(f"Wrote {len(result['predictions'])} candidates to {args.output}")
if __name__=='__main__':main()
