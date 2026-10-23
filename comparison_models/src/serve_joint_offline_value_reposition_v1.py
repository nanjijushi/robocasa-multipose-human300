#!/usr/bin/env python3
"""Runtime for the joint offline value/action/stop repositioning model."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
import jax.numpy as jnp
from flax import nnx

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_stage12_base',HERE/'serve_seed17_drawer_stage12.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

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

class JointMobilityPolicy(base.MobilityPolicy):
    def __init__(self,args):
        checkpoint=args.view_quality_checkpoint
        if checkpoint is None:raise ValueError('joint runtime requires --view-quality-checkpoint')
        args.view_quality_checkpoint=None
        try:super().__init__(args)
        finally:args.view_quality_checkpoint=checkpoint
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload.get('version')!='joint_offline_value_reposition_v1':
            raise RuntimeError(f'Unexpected joint model version: {payload.get("version")}')
        self.joint_action_names=tuple(payload['action_names'])
        names=[str(x) for x in payload['task_names']]
        vectors=np.asarray(payload['task_embeddings'],np.float32)
        self.joint_task_table={name:vectors[i] for i,name in enumerate(names)}
        self.joint_model=JointRepositionNet(vectors.shape[1])
        self.joint_model.load_state_dict(payload['model']);self.joint_model.eval()
        self.joint_state_mean=np.asarray(payload['base_state_mean'],np.float32)
        self.joint_state_std=np.asarray(payload['base_state_std'],np.float32)
        self.joint_embed_graph,self.joint_embed_state=nnx.split(self.model)
        def encode_spatial(st,batch):
            mdl=nnx.merge(self.joint_embed_graph,st);tokens,_=mdl.PaliGemma.img(batch,train=False)
            b,n,dim=tokens.shape
            if n!=256:raise RuntimeError(f'Expected 256 image tokens, got {n}')
            return tokens.reshape((b,16,16,dim)).reshape((b,4,4,4,4,dim)).mean(axis=(2,4)).astype(jnp.float32)
        self.joint_spatial_embed=base.jax.jit(encode_spatial)
        self._metadata.update({'joint_reposition_checkpoint':str(checkpoint),'joint_reposition_version':payload['version'],'movement_action_source':'joint learned six-direction head','stop_source':'joint learned stop head'})

    def _task_vector(self,prompt):
        p=str(prompt).strip()
        if p in self.joint_task_table:return self.joint_task_table[p]
        low=p.lower()
        key='Open the right drawer.' if 'right' in low and 'drawer' in low else ('Open the left drawer.' if 'left' in low and 'drawer' in low else None)
        if key is None or key not in self.joint_task_table:raise ValueError(f'Unsupported joint task prompt: {p!r}')
        return self.joint_task_table[key]

    def _predict_joint(self,raw):
        main=np.asarray(raw['observation/image'],dtype=np.float32)
        qpos=np.asarray(raw.get('observation/base_qpos'),dtype=np.float32).reshape(-1)
        if qpos.shape!=(3,):raise ValueError(f'observation/base_qpos must have shape (3,), got {qpos.shape}')
        grid=np.asarray(self.joint_spatial_embed(self.joint_embed_state,jnp.asarray(main[None]/127.5-1.0)),np.float32)
        task=torch.from_numpy(self._task_vector(raw.get('prompt',''))[None].astype(np.float32))
        state=torch.from_numpy(((qpos-self.joint_state_mean)/self.joint_state_std)[None].astype(np.float32))
        with torch.no_grad():
            value,actions,stop=self.joint_model(torch.from_numpy(grid),task,state)
            value=float(torch.sigmoid(value)[0]);stop=float(torch.sigmoid(stop)[0])
            probs=torch.softmax(actions[0],dim=-1).numpy().astype(np.float32)
        trans=probs[:4];trans=trans/max(float(trans.sum()),1e-8)
        yaw=probs[4:];yaw=yaw/max(float(yaw.sum()),1e-8)
        mode=np.asarray([probs[:4].sum(),probs[4:].sum()],np.float32)
        return value,stop,probs,mode,trans,yaw

    def infer(self,raw):
        if bool(raw.get('score_only',False)):
            value,stop,probs,mode,trans,yaw=self._predict_joint(raw)
            return {'actions':np.zeros((1,12),np.float32),'stage':'view_score','arm_locked':True,'score_only':True,
                    'view_quality':value,'joint_stop_probability':stop,
                    'joint_action_probabilities':probs,'joint_action_names':self.joint_action_names,
                    'spatial_mode_probabilities':mode,
                    'spatial_translation_probabilities':trans,
                    'spatial_translation_names':self.joint_action_names[:4],
                    'spatial_yaw_probabilities':yaw,'spatial_yaw_names':self.joint_action_names[4:],
                    'view_scorer_version':'joint_offline_value_reposition_v1'}
        return super().infer(raw)

    def close(self):
        self.joint_spatial_embed=None;self.joint_model=None
        super().close()

base.MobilityPolicy=JointMobilityPolicy
if __name__=='__main__':base.main()
