#!/usr/bin/env python3
"""Extract frozen Pi0.5 image tokens pooled from 16x16 to a 4x4 spatial grid."""
from __future__ import annotations
import argparse,json,sys,time,shutil
from pathlib import Path
import os
import numpy as np
ROOT=Path(os.environ.get('ROBOCASA_DATA_ROOT', Path(__file__).resolve().parents[2]));OPENPI=Path(os.environ.get('OPENPI_ROOT', ROOT/'third_party/openpi'))/'src';SOURCE=ROOT/'实验结果/20260922_seed17_seed19_seed31抽屉_三场景移动纠正段_v1';OUT=ROOT/'实验结果/20260922_seed17_seed19_seed31抽屉_空间视觉编码4x4_v1';CKPT=ROOT/'robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000'
def cli():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--checkpoint',type=Path,default=CKPT);p.add_argument('--config',default='pi05_pretrain_human300_inference');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-frames',type=int);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def main():
 a=cli();manifest=json.loads((a.source/'manifest.json').read_text());entries=[];images=[]
 for si,seg in enumerate(manifest['segments']):
  with np.load(a.source/seg['pair_id']/'visual_observations.npz',allow_pickle=False) as z:arr=z['raw_main']
  for frame,img in enumerate(arr):entries.append({'sample_index':len(entries),'pair_id':seg['pair_id'],'source_group':seg['source_group'],'frame':frame,'split':'val' if si%5==4 else 'train'});images.append(img)
 if a.max_frames is not None:entries=entries[:a.max_frames];images=images[:a.max_frames]
 if a.output.exists():
  if not a.overwrite:raise FileExistsError(a.output)
  shutil.rmtree(a.output)
 a.output.mkdir(parents=True);sys.path.insert(0,str(OPENPI))
 import jax,jax.numpy as jnp
 from flax import nnx
 from openpi.models import model as model_lib
 from openpi.training import config as config_lib
 from openpi_client import image_tools
 params=model_lib.restore_params(a.checkpoint/'params',dtype=jnp.bfloat16);model=config_lib.get_config(a.config).model.load(params);graph,state=nnx.split(model)
 def forward(st,batch):
  mdl=nnx.merge(graph,st);tokens,_=mdl.PaliGemma.img(batch,train=False);b,n,d=tokens.shape;side=jnp.asarray(n**.5,dtype=jnp.int32)
  # Pi0.5 currently returns 256 tokens = 16x16. Fail loudly if that changes.
  tokens=tokens.reshape((b,16,16,d));return tokens.reshape((b,4,4,4,4,d)).mean(axis=(2,4)).astype(jnp.float32)
 encode=jax.jit(forward);out=None;started=time.time()
 for begin in range(0,len(images),a.batch_size):
  batch=[]
  for img in images[begin:begin+a.batch_size]:batch.append(image_tools.resize_with_pad(np.asarray(img,np.uint8),224,224).astype(np.float32)/127.5-1)
  feat=np.asarray(encode(state,jnp.asarray(np.stack(batch))))
  if out is None:
   if feat.shape[1:]!=(4,4,2048):raise RuntimeError(f'unexpected pooled shape {feat.shape}')
   out=np.lib.format.open_memmap(a.output/'spatial_4x4.npy',mode='w+',dtype=np.float16,shape=(len(images),4,4,2048))
  out[begin:begin+len(feat)]=feat.astype(np.float16);out.flush();print(f'encoded {begin+len(feat)}/{len(images)}',flush=True)
 with (a.output/'index.jsonl').open('w',encoding='utf-8') as f:
  for e in entries:f.write(json.dumps(e,ensure_ascii=False)+'\n')
 summary={'status':'complete','frames':len(entries),'shape':list(out.shape),'dtype':'float16','encoder':'frozen Pi0.5 PaliGemma.img tokens, 16x16 average-pooled to 4x4; spatial order preserved','elapsed_seconds':time.time()-started,'jax_backend':jax.default_backend()};(a.output/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
