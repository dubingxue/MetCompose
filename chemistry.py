"""Executable restricted edit grammar; no learned inference uses mapped targets."""
import ast,json
from collections import Counter
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
BT={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE}

def canonical(m,stereo=True):
    if isinstance(m,str):m=Chem.MolFromSmiles(m)
    if m is None:raise ValueError('parse')
    m=Chem.Mol(m)
    for a in m.GetAtoms():a.SetAtomMapNum(0)
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    return Chem.MolToSmiles(m,isomericSmiles=stereo)

def order(b):return 0 if b is None else (4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()))

def execute(parent,actions):
    """action=(kind,i,j,z1,z2,old,new); i/j refer to original input indices."""
    mol=Chem.MolFromSmiles(parent);rw=Chem.RWMol(mol);changed=set();delete=set();seen=set()
    for kind,i,j,z1,z2,old,new in actions:
        key=(kind,i,j)
        if key in seen:raise ValueError('duplicate_action')
        seen.add(key)
        if kind=='B':
            if order(rw.GetBondBetweenAtoms(i,j))!=old:raise ValueError('bond_precondition')
            if old:rw.RemoveBond(i,j)
            if new:rw.AddBond(i,j,BT[new])
            changed.update([i,j])
        elif kind=='ADD':
            a=Chem.Atom(z2);k=rw.AddAtom(a);rw.AddBond(i,k,BT[new]);changed.add(i)
        elif kind=='DEL':
            if mol.GetAtomWithIdx(j).GetDegree()!=1:raise ValueError('not_terminal')
            delete.add(j);changed.add(i)
        elif kind=='Q':
            if rw.GetAtomWithIdx(i).GetFormalCharge()!=old:raise ValueError('charge_precondition')
            rw.GetAtomWithIdx(i).SetFormalCharge(new);changed.add(i)
        else:raise ValueError('unknown_action')
    for i in changed-delete:
        a=rw.GetAtomWithIdx(i)
        if not a.GetIsAromatic():a.SetNumExplicitHs(0);a.SetNoImplicit(False)
    for j in sorted(delete,reverse=True):rw.RemoveAtom(j)
    m=rw.GetMol();Chem.SanitizeMol(m)
    if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):raise ValueError('radical')
    return canonical(m)

def extract(mapped_parent,mapped_product):
    left,right=Chem.MolFromSmiles(mapped_parent),Chem.MolFromSmiles(mapped_product)
    if left is None or right is None:raise ValueError('parse')
    if not 4<=left.GetNumHeavyAtoms()<=80:raise ValueError('parent_size')
    if any(a.GetAtomicNum()==1 for m in [left,right] for a in m.GetAtoms()):raise ValueError('explicit_H')
    labels=[a.GetAtomMapNum() for a in left.GetAtoms()]
    if not all(labels) or len(labels)!=len(set(labels)):raise ValueError('bad_parent_maps')
    for a in left.GetAtoms():a.SetAtomMapNum(0)
    Chem.AssignStereochemistry(left,cleanIt=True,force=True)
    parent=Chem.MolToSmiles(left)
    out_order=json.loads(left.GetProp('_smilesAtomOutputOrder'))
    idx={labels[old]:new for new,old in enumerate(out_order)}
    la={labels[a.GetIdx()]:a for a in left.GetAtoms()};ra={a.GetAtomMapNum():a for a in right.GetAtoms()}
    if 0 in ra or len(ra)!=right.GetNumAtoms():raise ValueError('bad_product_maps')
    def bonds(m,maps):return {tuple(sorted((maps[b.GetBeginAtomIdx()],maps[b.GetEndAtomIdx()]))):order(b) for b in m.GetBonds()}
    lb=bonds(left,labels);rb=bonds(right,[a.GetAtomMapNum() for a in right.GetAtoms()])
    common=set(la)&set(ra);added=set(ra)-set(la);deleted=set(la)-set(ra);actions=[]
    for x in common:
        if la[x].GetAtomicNum()!=ra[x].GetAtomicNum() or la[x].GetIsotope()!=ra[x].GetIsotope():raise ValueError('element_or_isotope')
        if la[x].GetFormalCharge()!=ra[x].GetFormalCharge():
            actions.append(('Q',idx[x],idx[x],la[x].GetAtomicNum(),0,la[x].GetFormalCharge(),ra[x].GetFormalCharge()))
    for x in added:
        a=ra[x];ns=list(a.GetNeighbors())
        if len(ns)!=1 or ns[0].GetAtomMapNum() not in common or a.GetFormalCharge()!=0 or a.GetIsotope()!=0:raise ValueError('complex_addition')
        anchor=ns[0].GetAtomMapNum();bo=rb[tuple(sorted((x,anchor)))]
        if bo not in BT:raise ValueError('aromatic_edit')
        actions.append(('ADD',idx[anchor],idx[anchor],la[anchor].GetAtomicNum(),a.GetAtomicNum(),0,bo))
    for x in deleted:
        a=la[x];ns=list(a.GetNeighbors())
        if len(ns)!=1 or labels[ns[0].GetIdx()] not in common:raise ValueError('complex_deletion')
        anchor=labels[ns[0].GetIdx()];bo=lb[tuple(sorted((x,anchor)))]
        if bo not in BT:raise ValueError('aromatic_edit')
        actions.append(('DEL',idx[anchor],idx[x],la[anchor].GetAtomicNum(),a.GetAtomicNum(),bo,0))
    for pair in set(lb)|set(rb):
        if not set(pair)<=common:continue
        old,new=lb.get(pair,0),rb.get(pair,0)
        if old==new:continue
        if old==4 or new==4:raise ValueError('aromatic_edit')
        a,b=sorted(pair,key=lambda x:(la[x].GetAtomicNum(),idx[x]))
        actions.append(('B',idx[a],idx[b],la[a].GetAtomicNum(),la[b].GetAtomicNum(),old,new))
    if not 1<=len(actions)<=3:raise ValueError('edit_count')
    actions=sorted(actions)
    target=canonical(right)
    try:reconstructed=execute(parent,actions)
    except Exception:raise ValueError('execution_failure')
    if reconstructed!=target:raise ValueError('roundtrip_mismatch_including_stereo')
    return parent,target,actions

def token(a):return tuple([a[0]]+list(a[3:]))

def enumerate_actions(parent,vocab):
    m=Chem.MolFromSmiles(parent);atoms=list(m.GetAtoms());dist=Chem.GetDistanceMatrix(m);actions=[]
    for kind,z1,z2,old,new in vocab:
        for a in atoms:
            i=a.GetIdx()
            if a.GetAtomicNum()!=z1:continue
            if kind=='Q':
                if a.GetFormalCharge()==old:actions.append((kind,i,i,z1,z2,old,new))
            elif kind=='ADD':actions.append((kind,i,i,z1,z2,old,new))
            else:
                for b in (a.GetNeighbors() if old else atoms):
                    j=b.GetIdx()
                    if j==i or b.GetAtomicNum()!=z2:continue
                    bo=order(m.GetBondBetweenAtoms(i,j))
                    if bo!=old:continue
                    if kind=='B' and ((z1==z2 and i>j) or (old==0 and dist[i,j]>4)):continue
                    if kind=='DEL' and b.GetDegree()!=1:continue
                    actions.append((kind,i,j,z1,z2,old,new))
    return sorted(set(actions))
