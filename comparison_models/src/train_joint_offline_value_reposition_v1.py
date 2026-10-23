#!/usr/bin/env python3
"""Joint offline value-guided active repositioning from successful correction trajectories.

Frozen Pi0.5 spatial/task embeddings are inputs. A single trainable adapter jointly
predicts task-conditioned success value, six base directions, and when to stop.
TD targets are formed from recorded (observation, action, next observation) transitions;
the simulator is not differentiated through and the downstream manipulation VLA is frozen.
"""
from __future__ import annotations
import argparse,json,math,random,shutil,time
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn

ROOT=Path(os.environ.get('ROBOCASA_DATA_ROOT', Path(__file__).resolve().parents[2]))
DATA=ROOT/'实验结果/20260927_seed17_seed19_seed31抽屉_全量均衡移动纠正段_v1'
EMB=ROOT/'实验结果/20260927_seed17_seed19_seed31抽屉_全量均衡移动空间视觉编码4x4_v1'
TASK_EMB=ROOT/'mobility_checkpoints/seed17_seed19_seed31_pi05_task_embeddings_v1.npz'
OUT=ROOT/'mobility_checkpoints/seed17_seed19_seed31_joint_offline_value_reposition_v1'
ACTIONS=('forward','backward','left','right','yaw_left','yaw_right')

def cli():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,default=DATA);p.add_argument('--embeddings',type=Path,default=EMB)
 p.add_argument('--task-embeddings',type=Path,default=TASK_EMB);p.add_argument('--output',type=Path,default=OUT)
 p.add_argument('--steps',type=int,default=3000);p.add_argument('--batch-size',type=int,default=96)
 p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--gamma',type=float,default=.995)
 p.add_argument('--lookahead',type=int,default=5);p.add_argument('--endpoint-frames',type=int,default=4)
 p.add_argument('--endpoint-margin',type=float,default=.35);p.add_argument('--td-weight',type=float,default=1.)
 p.add_argument('--action-weight',type=float,default=1.);p.add_argument('--stop-weight',type=float,default=.5)
 p.add_argument('--rank-weight',type=float,default=1.);p.add_argument('--anchor-weight',type=float,default=.25)
 p.add_argument('--monotonic-weight',type=float,default=.25);p.add_argument('--eval-interval',type=int,default=100)
 p.add_argument('--seed',type=int,default=20261008);p.add_argument('--overwrite',action='store_true')
 return p.parse_args()

def task_for(pair):
 if str(pair).startswith(('s17_','supp_s17a_','supp_s17b_')):return 'Open the right drawer.'
 if str(pair).startswith(('s19_','s31_','supp_s19a_')):return 'Open the left drawer.'
 raise ValueError(f'unknown task for {pair}')

def action_label(a):
 a=np.asarray(a,float);k=int(np.argmax(np.abs(a)));v=float(a[k])
 if abs(v)<=1e-3:return -1
 if k==0:return 0 if v>0 else 1
 if k==1:return 2 if v>0 else 3
 return 4 if v>0 else 5

class JointRepositionNet(nn.Module):
 def __init__(self,task_dim):
  super().__init__()
  self.token=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,64),nn.GELU())
  self.pos=nn.Parameter(torch.zeros(16,64))
  self.visual=nn.Sequential(nn.Linear(1024,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU())
  self.task=nn.Sequential(nn.LayerNorm(task_dim),nn.Linear(task_dim,128),nn.GELU())
  self.state=nn.Sequential(nn.LayerNorm(3),nn.Linear(3,32),nn.GELU())
  self.trunk=nn.Sequential(nn.Linear(416,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU())
  self.value=nn.Linear(128,1);self.action=nn.Linear(128,6);self.stop=nn.Linear(128,1)
 def forward(self,x,t,s):
  h=self.token(x.reshape(-1,16,2048))+self.pos;v=self.visual(h.flatten(1));q=self.task(t);r=self.state(s)
  z=self.trunk(torch.cat([v,q,v*q,r],-1))
  return self.value(z).squeeze(-1),self.action(z),self.stop(z).squeeze(-1)

def balanced_accuracy(pred,y,n):
 vals=[np.mean(pred[y==k]==k) for k in range(n) if np.any(y==k)]
 return float(np.mean(vals)) if vals else 0.

def main():
 a=cli();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if a.output.exists() and any(a.output.iterdir()):
  if not a.overwrite:raise FileExistsError(f'{a.output} exists; pass --overwrite')
  shutil.rmtree(a.output)
 a.output.mkdir(parents=True,exist_ok=True);started=time.time()
 entries=[json.loads(x) for x in (a.embeddings/'index.jsonl').read_text(encoding='utf-8').splitlines()]
 feat=np.load(a.embeddings/'spatial_4x4.npy',mmap_mode='r')
 if feat.shape!=(len(entries),4,4,2048):raise ValueError((feat.shape,len(entries)))
 tp=np.load(a.task_embeddings,allow_pickle=False);table={str(k):np.asarray(v,np.float32) for k,v in zip(tp['tasks'],tp['embeddings'],strict=True)}
 task_names=sorted(table);task_to_id={x:i for i,x in enumerate(task_names)};task_vectors=np.stack([table[x] for x in task_names]).astype(np.float32)
 groups={}
 for i,e in enumerate(entries):groups.setdefault(str(e['pair_id']),[]).append(i)
 for ids in groups.values():ids.sort(key=lambda i:int(entries[i]['frame']))
 n=len(entries);states=np.zeros((n,3),np.float32);task_ids=np.zeros(n,np.int64);labels=np.full(n,-1,np.int64);next_idx=np.full(n,-1,np.int64);terminal=np.zeros(n,np.float32);stop_y=np.zeros(n,np.float32);pair_of=np.empty(n,object)
 train_groups=[];val_groups=[]
 for g,ids in groups.items():
  z=np.load(a.data/g/'movement.npz',allow_pickle=False);mov=np.asarray(z['base_action'][:,:3],np.float32);bq=np.asarray(z['base_qpos'],np.float32)
  frames=[int(entries[i]['frame']) for i in ids];task=task_for(g)
  if task not in task_to_id:raise RuntimeError(f'missing task embedding {task}')
  (val_groups if str(entries[ids[0]].get('split','train'))=='val' else train_groups).append(g)
  for k,i in enumerate(ids):
   f=frames[k];states[i]=bq[f];task_ids[i]=task_to_id[task];pair_of[i]=g
   if k+1<len(ids):next_idx[i]=ids[k+1]
   else:terminal[i]=1.
   if k>=max(0,len(ids)-a.endpoint_frames):stop_y[i]=1.
   chosen=None
   for h in range(k,min(len(ids),k+a.lookahead+1)):
    if np.max(np.abs(mov[frames[h]]))>1e-3:chosen=mov[frames[h]];break
   if chosen is not None:labels[i]=action_label(chosen)
 if not train_groups or not val_groups:raise RuntimeError((len(train_groups),len(val_groups)))
 tr=np.asarray([pair_of[i] in set(train_groups) for i in range(n)]);va=~tr
 state_mean=states[tr].mean(0);state_std=states[tr].std(0);state_std=np.maximum(state_std,1e-4);states=(states-state_mean)/state_std
 transition_ids=np.flatnonzero(tr&(next_idx>=0));action_ids=np.flatnonzero(tr&(labels>=0));stop_pos=np.flatnonzero(tr&(stop_y>0));stop_neg=np.flatnonzero(tr&(stop_y==0));buckets=[np.flatnonzero(tr&(labels==k)) for k in range(6)]
 if any(len(x)==0 for x in buckets):raise RuntimeError({'action_counts':[len(x) for x in buckets]})
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');task_tensor=torch.from_numpy(task_vectors).to(dev)
 model=JointRepositionNet(task_vectors.shape[1]).to(dev);target=JointRepositionNet(task_vectors.shape[1]).to(dev);target.load_state_dict(model.state_dict());target.eval()
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);history=[];best=None
 def batch(ids,net=model):
  ids=np.asarray(ids,dtype=np.int64);x=torch.from_numpy(np.asarray(feat[ids],np.float32)).to(dev);s=torch.from_numpy(states[ids]).to(dev);t=task_tensor[torch.from_numpy(task_ids[ids]).to(dev)];return net(x,t,s)
 @torch.no_grad()
 def evaluate():
  model.eval();ids=np.flatnonzero(va);vals=[];acts=[];stops=[]
  for b in range(0,len(ids),256):
   q=ids[b:b+256];vl,al,sl=batch(q);vals.extend(torch.sigmoid(vl).cpu().numpy());acts.extend(al.argmax(-1).cpu().numpy());stops.extend((torch.sigmoid(sl)>=.5).cpu().numpy())
  vals=np.asarray(vals);acts=np.asarray(acts);stops=np.asarray(stops);lookup={x:j for j,x in enumerate(ids)}
  aid=np.asarray([i for i in ids if labels[i]>=0]);ap=np.asarray([acts[lookup[i]] for i in aid]);ay=labels[aid]
  stop_true=stop_y[ids].astype(bool);stop_bal=.5*((stops[stop_true].mean() if stop_true.any() else 0.)+((~stops[~stop_true]).mean() if (~stop_true).any() else 0.))
  gaps=[];rank=[];jumps=[];mono=[]
  for g in val_groups:
   gi=groups[g];gv=np.asarray([vals[lookup[i]] for i in gi]);k=min(a.endpoint_frames,len(gi));gap=float(gv[-k:].mean()-gv[:k].mean());gaps.append(gap);rank.append(gap>=a.endpoint_margin);jumps.append(float(np.max(np.abs(np.diff(gv)))) if len(gv)>1 else 0.);mono.append(float(np.mean(np.diff(gv)>=0)) if len(gv)>1 else 1.)
  tid=np.asarray([i for i in ids if next_idx[i]>=0]);cur=np.asarray([vals[lookup[i]] for i in tid]);nxt=np.asarray([vals[lookup[next_idx[i]]] for i in tid]);reward=terminal[next_idx[tid]];td=np.abs(cur-(reward+(1-reward)*a.gamma*nxt))
  model.train();return {'action_balanced_accuracy':balanced_accuracy(ap,ay,6),'stop_balanced_accuracy':float(stop_bal),'endpoint_margin_accuracy':float(np.mean(rank)),'endpoint_separation_mean':float(np.mean(gaps)),'td_mae':float(np.mean(td)),'monotonic_step_fraction':float(np.mean(mono)),'mean_max_adjacent_jump':float(np.mean(jumps))}
 for step in range(1,a.steps+1):
  per=max(1,a.batch_size//6);act_ids=np.concatenate([np.random.choice(x,per,replace=True) for x in buckets]);np.random.shuffle(act_ids);vl,alog,sl=batch(act_ids);action_loss=nn.functional.cross_entropy(alog,torch.from_numpy(labels[act_ids]).to(dev))
  trans=np.random.choice(transition_ids,a.batch_size,replace=True);vcur,_,_=batch(trans);ni=next_idx[trans]
  with torch.no_grad():vnext=torch.sigmoid(batch(ni,target)[0]);reward=torch.from_numpy(terminal[ni]).to(dev);td_target=reward+(1-reward)*a.gamma*vnext
  td_loss=nn.functional.smooth_l1_loss(torch.sigmoid(vcur),td_target)
  ns=a.batch_size//2;sid=np.concatenate([np.random.choice(stop_pos,ns,replace=True),np.random.choice(stop_neg,a.batch_size-ns,replace=True)]);_,_,slog=batch(sid);stop_loss=nn.functional.binary_cross_entropy_with_logits(slog,torch.from_numpy(stop_y[sid]).to(dev))
  chosen=random.choices(train_groups,k=min(a.batch_size,len(train_groups)));lo=[];hi=[];early=[];late=[]
  for g in chosen:
   ids=groups[g];k=min(a.endpoint_frames,len(ids));lo.append(random.choice(ids[:k]));hi.append(random.choice(ids[-k:]));x,y=sorted(random.sample(range(len(ids)),2));early.append(ids[x]);late.append(ids[y])
  vlo=torch.sigmoid(batch(lo)[0]);vhi=torch.sigmoid(batch(hi)[0]);ve=torch.sigmoid(batch(early)[0]);vla=torch.sigmoid(batch(late)[0]);rank_loss=torch.relu(a.endpoint_margin-(vhi-vlo)).mean();anchor_loss=nn.functional.smooth_l1_loss(vlo,torch.full_like(vlo,.15))+nn.functional.smooth_l1_loss(vhi,torch.full_like(vhi,.85));mono_loss=torch.relu(.01-(vla-ve)).mean()
  loss=a.action_weight*action_loss+a.td_weight*td_loss+a.stop_weight*stop_loss+a.rank_weight*rank_loss+a.anchor_weight*anchor_loss+a.monotonic_weight*mono_loss
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  with torch.no_grad():
   for pt,p in zip(target.parameters(),model.parameters()):pt.mul_(.995).add_(p,alpha=.005)
  if step==1 or step%a.eval_interval==0 or step==a.steps:
   ev=evaluate();score=ev['action_balanced_accuracy']+ev['stop_balanced_accuracy']+2*ev['endpoint_margin_accuracy']+ev['monotonic_step_fraction']-ev['td_mae']-.25*ev['mean_max_adjacent_jump'];row={'step':step,'loss':float(loss.detach()),'action_loss':float(action_loss.detach()),'td_loss':float(td_loss.detach()),'stop_loss':float(stop_loss.detach()),'rank_loss':float(rank_loss.detach()),'anchor_loss':float(anchor_loss.detach()),'monotonic_loss':float(mono_loss.detach()),'selection_score':float(score),**ev};history.append(row);print(json.dumps(row),flush=True)
   if best is None or score>best['score']:best={'score':score,'step':step,'metrics':ev,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
 payload={'version':'joint_offline_value_reposition_v1','model':best['state'],'action_names':ACTIONS,'task_names':task_names,'task_embeddings':torch.from_numpy(task_vectors),'base_state_mean':torch.from_numpy(state_mean),'base_state_std':torch.from_numpy(state_std),'input':'frozen Pi0.5 4x4x2048 spatial tokens + frozen task embedding + normalized base qpos','outputs':'task success value logit + six-direction base action logits + learned stop logit','training':'shared adapter jointly optimized with balanced action imitation, offline TD(0), A/B ranking and anchors, monotonic ordering, and stop supervision; frozen downstream VLA','gamma':a.gamma,'hyperparameters':vars(a)}
 torch.save(payload,a.output/'joint_reposition.pt');summary={'status':'complete','frames':n,'pairs':len(groups),'train_pairs':len(train_groups),'val_pairs':len(val_groups),'action_counts':{ACTIONS[k]:int(np.sum(labels==k)) for k in range(6)},'best_step':best['step'],'selection_score':best['score'],'val':best['metrics'],'checkpoint':str(a.output/'joint_reposition.pt'),'elapsed_seconds':time.time()-started};(a.output/'metrics.json').write_text(json.dumps({'summary':summary,'history':history},ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
