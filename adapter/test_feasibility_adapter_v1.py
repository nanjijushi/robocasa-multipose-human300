from __future__ import annotations
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import embodied_feasibility_adapter_v1 as adapter_mod
import feasibility_losses_v1 as loss_mod
from feasibility_adapter_checkpoint_v1 import save_adapter_npz,load_adapter_npz
cfg=adapter_mod.FeasibilityAdapterConfig(source_dims=(64,96,128),model_dim=32,num_queries=4,num_heads=4,num_blocks=2,ff_dim=64)
model=adapter_mod.EmbodiedFeasibilityAdapter(cfg,rngs=nnx.Rngs(0));b=2;candidates=4
sources=[jnp.ones((b,16,64)),jnp.ones((b,7,96))*2,jnp.ones((b,7,128))*3];masks=[jnp.ones(x.shape[:2],dtype=bool) for x in sources]
actions=jnp.array([[[0,0,0,1],[.1,0,0,0],[0,.1,0,0],[0,0,.1,0]]]*b,dtype=jnp.float32)
probs,logits,tokens=model(sources,masks,actions);assert probs.shape==(b,candidates);assert tokens.shape==(b,4,32)
successes=jnp.array([[2,8,5,1],[5,5,5,5]],dtype=jnp.float32);trials=jnp.ones((b,candidates))*10;valid=jnp.ones((b,candidates),dtype=bool)
dist=jnp.array([[.8,.6,.7,.9],[.8,.6,.7,.9]],dtype=jnp.float32);safe=jnp.ones_like(valid)
loss,metrics=loss_mod.feasibility_adapter_loss(logits,successes,trials,valid,distances_m=dist,safe_mask=safe);assert jnp.isfinite(loss)
graph,state=nnx.split(model)
def objective(state):
 m=nnx.merge(graph,state);_,z,_=m(sources,masks,actions);return loss_mod.rollout_value_loss(z,successes,trials,valid)
grads=jax.grad(objective)(state);assert float(jnp.linalg.norm(grads['queries'].value))>0
ckpt=Path('/tmp/adapter_roundtrip_v1.npz');save_adapter_npz(ckpt,model,{'test':True})
restored=adapter_mod.EmbodiedFeasibilityAdapter(cfg,rngs=nnx.Rngs(999));restored=load_adapter_npz(ckpt,restored);restored_probs,_,_=restored(sources,masks,actions)
assert float(jnp.max(jnp.abs(restored_probs-probs)))<1e-7
print({'status':'ok','prob_shape':probs.shape,'token_shape':tokens.shape,'loss':float(loss),'checkpoint_roundtrip_max_error':float(jnp.max(jnp.abs(restored_probs-probs))),**{k:float(v) for k,v in metrics.items()}})
