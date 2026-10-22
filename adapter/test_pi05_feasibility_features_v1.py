from __future__ import annotations
import types
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.models import pi0_config
import embodied_feasibility_adapter_v1 as adapter_lib
from pi05_feasibility_features_v1 import extract_frozen_pi05_features

class FakeImageEncoder:
    def __call__(self,image,train=False):
        b=image.shape[0]; base=jnp.mean(image,axis=(1,2,3))[:,None,None]
        return jnp.broadcast_to(base,(b,4,16)),None
class FakeLLM:
    def __call__(self,x,method=None,**kwargs):
        if method=='embed':
            return jnp.broadcast_to(x[...,None].astype(jnp.float32),(x.shape[0],x.shape[1],16))/100
        prefix,suffix=x
        return (prefix+0.5,None),None
class FakePi:
    def __init__(self): self.PaliGemma=types.SimpleNamespace(img=FakeImageEncoder(),llm=FakeLLM())

cfg=pi0_config.Pi0Config(pi05=True,paligemma_variant='dummy',action_expert_variant='dummy',discrete_state_input=False)
obs=cfg.fake_obs(batch_size=2)
features=extract_frozen_pi05_features(FakePi(),obs)
assert len(features.sources)==len(obs.images)+1
assert features.source_names[-1]=='paligemma:final_task_conditioned_prefix'
dims=tuple(int(x.shape[-1]) for x in features.sources)
adapter=adapter_lib.EmbodiedFeasibilityAdapter(adapter_lib.FeasibilityAdapterConfig(source_dims=dims,model_dim=32,num_queries=4,num_heads=4,num_blocks=1,ff_dim=64),rngs=nnx.Rngs(2))
actions=jnp.zeros((2,3,4)); actions=actions.at[:,0,3].set(1)
probs,logits,tokens=adapter(features.sources,features.source_masks,actions)
assert probs.shape==(2,3) and tokens.shape==(2,4,32)
print({'status':'ok','source_names':features.source_names,'source_shapes':[x.shape for x in features.sources],'prob_shape':probs.shape})
