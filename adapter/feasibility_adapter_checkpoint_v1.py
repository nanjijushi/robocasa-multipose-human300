"""Portable NPZ checkpoint helpers for the NNX feasibility adapter."""
from __future__ import annotations
import json
from pathlib import Path
from flax import traverse_util
import flax.nnx as nnx
import numpy as np


def _key(path): return '/'.join(map(str,path))


def save_adapter_npz(path: Path, model_or_state, metadata: dict):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    state=nnx.state(model_or_state) if isinstance(model_or_state,nnx.Module) else model_or_state
    flat=traverse_util.flatten_dict(state.filter(nnx.Param).to_pure_dict())
    np.savez_compressed(path,**{_key(k):np.asarray(v.value if hasattr(v,'value') else v) for k,v in flat.items() if v is not None})
    path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


def load_adapter_npz(path: Path, model):
    """Load by matching serialized names to the reference model's typed paths."""
    path=Path(path); archive=np.load(path,allow_pickle=False); state=nnx.state(model)
    reference=traverse_util.flatten_dict(state.filter(nnx.Param).to_pure_dict())
    numeric_reference={k:v for k,v in reference.items() if v is not None}
    missing=[_key(k) for k in numeric_reference if _key(k) not in archive]
    unexpected=sorted(set(archive.files)-{_key(k) for k in numeric_reference})
    if missing or unexpected: raise ValueError({'missing':missing,'unexpected':unexpected})
    loaded=dict(reference)
    loaded.update({k:archive[_key(k)] for k in numeric_reference})
    pure=traverse_util.unflatten_dict(loaded)
    state.replace_by_pure_dict(pure)
    graph=nnx.graphdef(model)
    return nnx.merge(graph,state)


