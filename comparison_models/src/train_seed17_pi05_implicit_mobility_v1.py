#!/usr/bin/env python3
"""Current-observation-only Pi0.5 mobility LoRA; loss is restricted to base dims 7:10."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import os
import einops
import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as model_lib
import openpi.models.pi0 as pi0_lib
from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils
from openpi.training import weight_loaders

ROOT=Path(os.environ.get('ROBOCASA_DATA_ROOT', Path(__file__).resolve().parents[2]))
DATA=ROOT/'实验结果'/'20260915_seed17右抽屉_20组_A到B移动纠正段_v1'
INDEX=ROOT/'实验结果'/'20260915_seed17右抽屉_移动视觉编码_v1/index.jsonl'
BASE=ROOT/'robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000'
OUT=ROOT/'mobility_checkpoints/seed17_pi05_implicit_mobility_lora_v1'
PROMPT='Move the mobile base to a suitable viewpoint for grasping the right drawer handle.'

def cli():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=DATA); p.add_argument('--index',type=Path,default=INDEX); p.add_argument('--base-checkpoint',type=Path,default=BASE); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--steps',type=int,default=1200); p.add_argument('--batch-size',type=int,default=1); p.add_argument('--horizon',type=int,default=16); p.add_argument('--lr',type=float,default=2e-5); p.add_argument('--eval-interval',type=int,default=50); p.add_argument('--eval-batches',type=int,default=8); p.add_argument('--seed',type=int,default=17); p.add_argument('--prompt',default=PROMPT); p.add_argument('--balanced-v2',action='store_true',help='sample segments uniformly, retain yaw, cap idle gaps'); p.add_argument('--lookahead-frames',type=int,default=5); p.add_argument('--terminal-stop-frames',type=int,default=8); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def dense_commands(entries,data):
 labels=np.zeros(len(entries),np.int8); by_group={}; cache={}
 for i,e in enumerate(entries): by_group.setdefault(e['source_group'],[]).append(i)
 for ids in by_group.values():
  ids.sort(key=lambda i:(entries[i]['pair_id'],entries[i]['frame'])); nxt=0
  for i in reversed(ids):
   e=entries[i]; p=e['pair_id']
   if p not in cache: cache[p]=dict(np.load(data/p/'movement.npz',allow_pickle=False))
   a=cache[p]['base_action'][e['frame'],:2]
   if np.max(np.abs(a))>1e-3: nxt=1 if abs(a[0])>=abs(a[1]) else 2
   labels[i]=nxt
 return labels,by_group

def balanced_commands(entries,data,lookahead,terminal_stops):
 commands=np.zeros((len(entries),3),np.float32); by_segment={}; cache={}; candidates={}
 for i,e in enumerate(entries): by_segment.setdefault(e['pair_id'],[]).append(i)
 for pair,ids in by_segment.items():
  ids.sort(key=lambda i:entries[i]['frame'])
  if pair not in cache: cache[pair]=dict(np.load(data/pair/'movement.npz',allow_pickle=False))
  base=np.asarray(cache[pair]['base_action'],np.float32)[:,:3]
  future=np.zeros(3,np.float32); age=lookahead+1
  for i in reversed(ids):
   a=base[entries[i]['frame']]
   if np.max(np.abs(a))>1e-3: future=a.copy(); age=0
   else: age+=1
   if age<=lookahead: commands[i]=future
  active=[i for i in ids if np.max(np.abs(commands[i]))>1e-3]
  last=max(active,key=lambda i:entries[i]['frame']) if active else ids[-1]
  terminal=[i for i in ids if entries[i]['frame']>entries[last]['frame']][-terminal_stops:]
  candidates[pair]=active+terminal
 return commands,by_segment,candidates

def masked_flow_loss(m,rng,obs,actions,train=True):
 pre,noise_rng,time_rng=jax.random.split(rng,3); obs=model_lib.preprocess_observation(pre,obs,train=train); batch_shape=actions.shape[:-2]; noise=jax.random.normal(noise_rng,actions.shape); t=jax.random.beta(time_rng,1.5,1,batch_shape)*.999+.001; te=t[...,None,None]; xt=te*noise+(1-te)*actions; target=noise-actions
 pt,pm,pa=m.embed_prefix(obs); st,sm,sa,cond=m.embed_suffix(obs,xt,t); mask=jnp.concatenate([pm,sm],1); ar=jnp.concatenate([pa,sa],0); attn=pi0_lib.make_attn_mask(mask,ar); pos=jnp.cumsum(mask,1)-1; (_,suffix),_=m.PaliGemma.llm([pt,st],mask=attn,positions=pos,adarms_cond=[None,cond]); pred=m.action_out_proj(suffix[:,-m.action_horizon:]); return jnp.mean(jnp.square(pred[...,7:10]-target[...,7:10]))

def save_adapter(state,train_filter,path,meta):
 pure=state.filter(train_filter).to_pure_dict(); flat=traverse_util.flatten_dict(pure,sep='/'); arrays={k:np.asarray(v) for k,v in flat.items()}; np.savez_compressed(path,**arrays); path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))

def main():
 a=cli(); random.seed(a.seed); np.random.seed(a.seed)
 if a.output.exists() and any(a.output.iterdir()) and not a.overwrite: raise FileExistsError(a.output)
 a.output.mkdir(parents=True,exist_ok=True); entries=[json.loads(s) for s in a.index.read_text().splitlines()]; groups=np.asarray([e['source_group'] for e in entries]); split=np.asarray([e['split'] for e in entries]); train_ids=np.flatnonzero(split=='train'); val_ids=np.flatnonzero(split=='val')
 if a.balanced_v2: commands,by_group,candidates=balanced_commands(entries,a.data,a.lookahead_frames,a.terminal_stop_frames)
 else: labels,by_group=dense_commands(entries,a.data); commands=None; candidates=None
 norm=json.loads((a.base_checkpoint/'assets/norm_stats.json').read_text())['norm_stats']; sm=np.asarray(norm['state']['mean'],np.float32); ss=np.maximum(np.asarray(norm['state']['std'],np.float32),1e-6); am=np.asarray(norm['actions']['mean'],np.float32); ast=np.maximum(np.asarray(norm['actions']['std'],np.float32),1e-6)
 cache={}
 tok=PaligemmaTokenizer(max_len=200)
 def batch(ids):
  if a.balanced_v2:
   allowed=set(map(int,ids)); pairs=[p for p,q in candidates.items() if any(i in allowed for i in q)]
   chosen=np.asarray([random.choice([i for i in candidates[random.choice(pairs)] if i in allowed]) for _ in range(a.batch_size)])
  else: chosen=np.random.choice(ids,size=a.batch_size,replace=True)
  ims=[]; wrists=[]; states=[]; acts=[]; toks=[]; tm=[]
  for idx in chosen:
   e=entries[int(idx)]; p=e['pair_id']
   if p not in cache:
    v=np.load(a.data/p/'visual_observations.npz',allow_pickle=False); cache[p]=(v['raw_main'],v['raw_wrist'],np.load(a.data/p/'policy_state.npy'))
   main,wrist,state=cache[p]; s=np.zeros(32,np.float32); s[:16]=state[e['frame']]; sn=(s-sm)/ss; t,m=tok.tokenize(a.prompt,sn); raw=np.zeros((a.horizon,32),np.float32); group_seq=by_group[p if a.balanced_v2 else e['source_group']]; at=group_seq.index(int(idx))
   for h in range(a.horizon):
    gi=group_seq[min(at+h,len(group_seq)-1)]
    if a.balanced_v2: raw[h,7:10]=commands[gi]
    else:
     lab=int(labels[gi])
     if lab==1: raw[h,7]=1.
     elif lab==2: raw[h,8]=-1.
   ims.append(main[e['frame']]); wrists.append(wrist[e['frame']]); states.append(sn); acts.append((raw-am)/ast); toks.append(t); tm.append(m)
  main_arr=np.asarray(ims); right_zeros=np.zeros_like(main_arr); obs=model_lib.Observation(images={'base_0_rgb':jnp.asarray(main_arr,jnp.float32)/127.5-1,'left_wrist_0_rgb':jnp.asarray(np.asarray(wrists),jnp.float32)/127.5-1,'right_wrist_0_rgb':jnp.asarray(right_zeros,jnp.float32)/127.5-1},image_masks={'base_0_rgb':jnp.ones(a.batch_size,dtype=jnp.bool_),'left_wrist_0_rgb':jnp.ones(a.batch_size,dtype=jnp.bool_),'right_wrist_0_rgb':jnp.zeros(a.batch_size,dtype=jnp.bool_)},state=jnp.asarray(states),tokenized_prompt=jnp.asarray(toks),tokenized_prompt_mask=jnp.asarray(tm)); return obs,jnp.asarray(acts)
 cfg=pi0_config.Pi0Config(pi05=True,action_horizon=a.horizon,max_token_len=200,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m_lora'); rng=jax.random.key(a.seed); model=cfg.create(rng); graph,state=nnx.split(model); loaded=weight_loaders.CheckpointWeightLoader(str(a.base_checkpoint/'params')).load(state.to_pure_dict()); state.replace_by_pure_dict(loaded); model=nnx.merge(graph,state)
 train_filter=nnx.All(nnx.Param,nnx_utils.PathRegex('.*lora.*')); frozen_filter=nnx.All(nnx.Param,nnx.Not(nnx_utils.PathRegex('.*lora.*'))); state=nnx.state(model); state=nnx_utils.state_map(state,frozen_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16))); train_flat=traverse_util.flatten_dict(state.filter(train_filter).to_pure_dict(),sep='/'); assert train_flat and all('_1/' in k or 'mlp_1/' in k for k in train_flat),sorted(train_flat); trainable_parameters=sum(int(np.prod(v.shape)) for v in train_flat.values()); graph=nnx.graphdef(model); model=nnx.merge(graph,state); tx=optax.adamw(a.lr,weight_decay=1e-4); opt=tx.init(state.filter(train_filter))
 def step(state,opt,rng,obs,actions):
  m=nnx.merge(graph,state)
  def lf(mm): return masked_flow_loss(mm,rng,obs,actions,True)
  loss,grads=nnx.value_and_grad(lf,argnums=nnx.DiffState(0,train_filter))(m); pars=state.filter(train_filter); updates,opt=tx.update(grads,opt,pars); nnx.update(m,optax.apply_updates(pars,updates)); return nnx.state(m),opt,loss,optax.global_norm(grads)
 pstep=jax.jit(step); best=math.inf; history=[]; started=time.time()
 for n in range(1,a.steps+1):
  rng,key=jax.random.split(rng); obs,actions=batch(train_ids); state,opt,loss,gn=pstep(state,opt,key,obs,actions)
  if n==1 or n%a.eval_interval==0 or n==a.steps:
   m=nnx.merge(graph,state); vals=[]
   for _ in range(a.eval_batches):
    rng,key=jax.random.split(rng); vo,va=batch(val_ids); vals.append(float(masked_flow_loss(m,key,vo,va,False)))
   row={'step':n,'train_masked_base_loss':float(loss),'grad_norm':float(gn),'val_masked_base_loss':float(np.mean(vals))}; history.append(row); print(json.dumps(row),flush=True)
   if row['val_masked_base_loss']<best:
    best=row['val_masked_base_loss']; save_adapter(state,train_filter,a.output/'best_adapter.npz',{'step':n,'val_masked_base_loss':best})
 summary={'status':'complete','training':'Pi0.5 action-expert LoRA using current observation only','prompt':a.prompt,'image_roles':{'base_0_rgb':'current main view','left_wrist_0_rgb':'current wrist view','right_wrist_0_rgb':'masked padding; no B image at inference'},'base_checkpoint':str(a.base_checkpoint),'base_checkpoint_modified':False,'loss_dimensions':[7,8,9],'arm_dimensions_in_loss':False,'balanced_v2':a.balanced_v2,'yaw_supervised':a.balanced_v2,'sampling':'uniform segment then eligible frame' if a.balanced_v2 else 'uniform frame','lookahead_frames':a.lookahead_frames if a.balanced_v2 else None,'terminal_stop_frames':a.terminal_stop_frames if a.balanced_v2 else None,'runtime_arm_action_policy':'force dimensions 0:7 to zero during mobility; use learned view-quality score for stopping; use original checkpoint for grasp','frames':len(entries),'train_frames':len(train_ids),'val_frames':len(val_ids),'action_horizon':a.horizon,'steps':a.steps,'best_val_masked_base_loss':best,'trainable_filter':'.*lora.* (action expert only)','trainable_parameters':trainable_parameters,'history':history,'elapsed_seconds':time.time()-started}; (a.output/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
