"""Train the success-conditioned Pi0.5 feasibility adapter from cached features.

Dataset NPZ schema (all numeric, allow_pickle=False):
  source_0 ... source_N: [examples, tokens_i, feature_dim_i]
  source_mask_0 ...: bool [examples, tokens_i]
  candidate_actions: [examples, candidates, action_dim]
  candidate_mask: bool [examples, candidates]
  successes, trials: [examples, candidates]
  split: uint8 [examples], 0=train and 1=validation
Optional:
  distances_m: [examples, candidates]
  safe_mask: bool [examples, candidates]
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from embodied_feasibility_adapter_v1 import EmbodiedFeasibilityAdapter, FeasibilityAdapterConfig
from feasibility_losses_v1 import FeasibilityLossConfig, feasibility_adapter_loss

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--model-dim',type=int,default=256); p.add_argument('--num-queries',type=int,default=4)
    p.add_argument('--num-heads',type=int,default=8); p.add_argument('--num-blocks',type=int,default=2)
    p.add_argument('--ff-dim',type=int,default=768); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--eval-interval',type=int,default=50); p.add_argument('--eval-batches',type=int,default=8)
    p.add_argument('--rank-weight',type=float,default=.5); p.add_argument('--distance-weight',type=float,default=.1)
    return p.parse_args()


def save_state(path,state,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    flat=traverse_util.flatten_dict(state.filter(nnx.Param).to_pure_dict())
    arrays={'/'.join(map(str,k)):np.asarray(v.value if hasattr(v,'value') else v) for k,v in flat.items() if v is not None}
    np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    a=parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    data=np.load(a.dataset,allow_pickle=False)
    source_ids=sorted(int(k.split('_')[-1]) for k in data.files if k.startswith('source_') and not k.startswith('source_mask_'))
    if source_ids != list(range(len(source_ids))): raise ValueError(f'non-contiguous sources: {source_ids}')
    sources=[np.asarray(data[f'source_{i}'],np.float32) for i in source_ids]
    masks=[np.asarray(data[f'source_mask_{i}'],bool) for i in source_ids]
    n=len(sources[0]); required=['candidate_actions','candidate_mask','successes','trials','split']
    for k in required:
        if k not in data: raise KeyError(f'missing dataset field: {k}')
    for i,(x,m) in enumerate(zip(sources,masks,strict=True)):
        if len(x)!=n or m.shape!=x.shape[:2]: raise ValueError(f'invalid source {i}: {x.shape}, {m.shape}')
    actions=np.asarray(data['candidate_actions'],np.float32); candidate_mask=np.asarray(data['candidate_mask'],bool)
    successes=np.asarray(data['successes'],np.float32); trials=np.asarray(data['trials'],np.float32); split=np.asarray(data['split'],np.uint8)
    distances=np.asarray(data['distances_m'],np.float32) if 'distances_m' in data else None
    safe=np.asarray(data['safe_mask'],bool) if 'safe_mask' in data else None
    expected=actions.shape[:2]
    if any(x.shape!=expected for x in (candidate_mask,successes,trials)): raise ValueError('candidate label shapes disagree')
    if np.any(successes<0) or np.any(trials<successes): raise ValueError('require 0 <= successes <= trials')
    train_ids=np.flatnonzero(split==0); val_ids=np.flatnonzero(split==1)
    if not len(train_ids) or not len(val_ids): raise ValueError('dataset needs nonempty split 0 and split 1')

    cfg=FeasibilityAdapterConfig(source_dims=tuple(x.shape[-1] for x in sources),action_dim=actions.shape[-1],model_dim=a.model_dim,num_queries=a.num_queries,num_heads=a.num_heads,num_blocks=a.num_blocks,ff_dim=a.ff_dim)
    loss_cfg=FeasibilityLossConfig(rank_weight=a.rank_weight,distance_weight=a.distance_weight)
    rng=jax.random.key(a.seed); model=EmbodiedFeasibilityAdapter(cfg,rngs=nnx.Rngs(a.seed))
    graph,state=nnx.split(model); train_filter=nnx.Param; tx=optax.adamw(a.lr,weight_decay=a.weight_decay); opt=tx.init(state.filter(train_filter))

    def make_batch(ids):
        kw={
            'sources':tuple(jnp.asarray(x[ids]) for x in sources), 'masks':tuple(jnp.asarray(x[ids]) for x in masks),
            'actions':jnp.asarray(actions[ids]), 'candidate_mask':jnp.asarray(candidate_mask[ids]),
            'successes':jnp.asarray(successes[ids]), 'trials':jnp.asarray(trials[ids]),
        }
        if distances is not None: kw['distances']=jnp.asarray(distances[ids]); kw['safe']=jnp.asarray(safe[ids])
        else: kw['distances']=None; kw['safe']=None
        return kw

    def step(state,opt,batch):
        m=nnx.merge(graph,state)
        def lf(mm):
            _,logits,_=mm(batch['sources'],batch['masks'],batch['actions'])
            return feasibility_adapter_loss(logits,batch['successes'],batch['trials'],batch['candidate_mask'],config=loss_cfg,distances_m=batch['distances'],safe_mask=batch['safe'])
        (loss,metrics),grads=nnx.value_and_grad(lf,argnums=nnx.DiffState(0,train_filter),has_aux=True)(m)
        pars=state.filter(train_filter); updates,opt=tx.update(grads,opt,pars); nnx.update(m,optax.apply_updates(pars,updates))
        return nnx.state(m),opt,loss,metrics,optax.global_norm(grads)
    pstep=jax.jit(step)

    def evaluate(state):
        m=nnx.merge(graph,state); rows=[]
        for _ in range(a.eval_batches):
            ids=np.random.default_rng(a.seed+len(rows)).choice(val_ids,size=min(a.batch_size,len(val_ids)),replace=len(val_ids)<a.batch_size)
            b=make_batch(ids); _,logits,_=m(b['sources'],b['masks'],b['actions']); loss,met=feasibility_adapter_loss(logits,b['successes'],b['trials'],b['candidate_mask'],config=loss_cfg,distances_m=b['distances'],safe_mask=b['safe']); rows.append({k:float(v) for k,v in met.items()})
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

    np_rng=np.random.default_rng(a.seed); best=math.inf; history=[]; started=time.time()
    for step_n in range(1,a.steps+1):
        ids=np_rng.choice(train_ids,size=a.batch_size,replace=len(train_ids)<a.batch_size); batch=make_batch(ids)
        state,opt,loss,metrics,grad_norm=pstep(state,opt,batch)
        if step_n==1 or step_n%a.eval_interval==0 or step_n==a.steps:
            val=evaluate(state); row={'step':step_n,'train_loss':float(loss),'grad_norm':float(grad_norm),**{f'val_{k}':v for k,v in val.items()}}; history.append(row); print(json.dumps(row),flush=True)
            if val['loss']<best:
                best=val['loss']; save_state(a.output/'best_adapter.npz',state,{'step':step_n,'val_loss':best,'config':cfg.__dict__,'loss_config':loss_cfg.__dict__})
    summary={'status':'complete','dataset':str(a.dataset),'base_checkpoint_modified':False,'train_examples':len(train_ids),'val_examples':len(val_ids),'best_val_loss':best,'config':cfg.__dict__,'loss_config':loss_cfg.__dict__,'history':history,'elapsed_seconds':time.time()-started}
    (a.output/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()



