#!/usr/bin/env python3
"""Serve the seed-17 mobility LoRA, optionally switching to the original drawer policy.

The LoRA is merged into the plain action expert in host RAM before model creation.
This removes LoRA branches from the inference graph and keeps GPU use close to the
original Pi0.5 policy. During mobility, arm/gripper actions are forced to zero.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import os
import socket
import time

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

class TaskViewNet(nn.Module):
    def __init__(self, vd, td):
        super().__init__()
        self.v=nn.Sequential(nn.LayerNorm(vd),nn.Linear(vd,64),nn.GELU())
        self.t=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(192,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,v,t):
        v=self.v(v);t=self.t(t)
        return self.head(torch.cat([v,t,v*t],-1)).squeeze(-1)

import openpi.models.model as model_lib
from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from openpi.training import config as train_config_lib


ROOT = Path(os.environ.get('ROBOCASA_DATA_ROOT', Path(__file__).resolve().parents[2]))
BASE = ROOT / "robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
ADAPTER = Path(__file__).resolve().parents[1] / "weights/pure_mobility_lora_v2.npz"
RIGHT_PROMPT = "Move the mobile base to a suitable viewpoint for grasping the right drawer handle."
LEFT_PROMPT = "Move the mobile base to a suitable viewpoint for grasping the left drawer handle."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("mobility", "full"), default="mobility")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--base-checkpoint", type=Path, default=BASE)
    p.add_argument("--adapter", type=Path, default=ADAPTER)
    p.add_argument("--movement-prompt", default=None,
                   help="fixed mobility prompt; omit to infer left/right from the task prompt")
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--view-quality-checkpoint", type=Path, default=None)
    p.add_argument("--action-gain-checkpoint", type=Path, default=None)
    p.add_argument("--continuous-policy-checkpoint", type=Path, default=None)
    p.add_argument("--hierarchical-policy-checkpoint", type=Path, default=None)
    p.add_argument("--spatial-hierarchical-policy-checkpoint", type=Path, default=None)
    p.add_argument("--view-quality-threshold", type=float, default=0.78)
    p.add_argument("--view-quality-confirmations", type=int, default=3)
    p.add_argument("--decision-log", type=Path, default=None)
    p.add_argument("--max-base-command", type=float, default=0.35)
    p.add_argument("--base-translation-deadband", type=float, default=0.08)
    p.add_argument("--base-yaw-deadband", type=float, default=0.075)
    p.add_argument("--decision-action-window", type=int, default=4)
    p.add_argument("--stop-command-threshold", type=float, default=0.08)
    p.add_argument("--stop-confirmations", type=int, default=3)
    p.add_argument("--motion-start-threshold", type=float, default=0.12)
    p.add_argument("--minimum-mobility-inferences", type=int, default=5)
    p.add_argument("--maximum-mobility-inferences", type=int, default=0,
                   help="force grasp handoff at this inference count; 0 disables the guard")
    p.add_argument("--episode-reset-idle-seconds", type=float, default=1.0)
    p.add_argument("--mobility-reset-idle-seconds", type=float, default=30.0,
                   help="reset mobility stop-state only after a real request gap")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    for name in ("action_gain_checkpoint", "continuous_policy_checkpoint", "hierarchical_policy_checkpoint", "spatial_hierarchical_policy_checkpoint"):
        if getattr(args, name) is not None:
            p.error(f"--{name.replace('_', '-')} belongs to a different experiment and is not included")
    return args


def merge_lora(base_checkpoint: Path, adapter_path: Path):
    logging.info("Restoring the original checkpoint to host RAM as bfloat16")
    params = model_lib.restore_params(
        base_checkpoint / "params", restore_type=np.ndarray, dtype=jnp.bfloat16
    )
    flat = traverse_util.flatten_dict(params, sep="/")
    with np.load(adapter_path, allow_pickle=False) as z:
        adapter = {key: np.asarray(z[key]) for key in z.files}

    a_keys = sorted(key for key in adapter if key.endswith("lora_a"))
    if len(a_keys) != 5:
        raise RuntimeError(f"Expected five LoRA matrix pairs, found {len(a_keys)}: {a_keys}")
    merged = []
    for a_key in a_keys:
        b_key = a_key[:-1] + "b"
        if b_key not in adapter:
            raise KeyError(f"Missing matching LoRA B matrix: {b_key}")
        if "/attn/" in a_key:
            target = a_key.rsplit("/", 1)[0] + "/w"
        else:
            target = a_key[: -len("_lora_a")]
        if target not in flat:
            raise KeyError(f"Base parameter not found for {a_key}: expected {target}")
        a = adapter[a_key].astype(np.float32, copy=False)
        b = adapter[b_key].astype(np.float32, copy=False)
        delta = np.einsum("...ir,...rj->...ij", a, b, optimize=True)
        if delta.shape != flat[target].shape:
            raise ValueError(f"Shape mismatch for {target}: {delta.shape} != {flat[target].shape}")
        flat[target] = (flat[target].astype(np.float32) + delta).astype(flat[target].dtype)
        merged.append({"a": a_key, "b": b_key, "target": target, "shape": list(delta.shape)})
        del delta
        gc.collect()
    del adapter
    logging.info("Merged %d LoRA pairs into the plain action expert", len(merged))
    return traverse_util.unflatten_dict(flat, sep="/"), merged


class MobilityPolicy:
    def __init__(self, args):
        self.args = args
        params, merged = merge_lora(args.base_checkpoint, args.adapter)
        self.cfg = pi0_config.Pi0Config(
            pi05=True,
            # Match the 16-step horizon used when training this mobility LoRA.
            # The client executes 12 actions before requesting a new plan.
            action_horizon=16,
            max_token_len=200,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        )
        logging.info("Creating the branch-free merged Pi0.5 mobility model")
        self.model = self.cfg.load(params)
        del params
        gc.collect()
        self.sample = nnx_utils.module_jit(self.model.sample_actions)
        self.rng = jax.random.key(17)
        self.tokenizer = PaligemmaTokenizer(max_len=200)
        self.view_scorer = None
        if args.view_quality_checkpoint is not None:
            payload = torch.load(args.view_quality_checkpoint, map_location="cpu", weights_only=False)
            q = int(payload["pca_V"].shape[1])
            self.task_conditioned_view = str(payload.get("version", "")).endswith("task_conditioned_view_value_v2")
            if self.task_conditioned_view:
                self.view_scorer = TaskViewNet(q, int(payload["task_embedding_dim"]))
                self.view_scorer.load_state_dict(payload["model"])
            else:
                self.view_scorer = nn.Sequential(nn.Linear(q,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
                state = {k.removeprefix("net."): v for k,v in payload["model"].items()}
                self.view_scorer.load_state_dict(state)
            self.view_scorer.eval()
            self.feature_mean=np.asarray(payload["feature_mean"],np.float32); self.feature_std=np.asarray(payload["feature_std"],np.float32)
            self.pca_v=np.asarray(payload["pca_V"],np.float32)
            self.embed_graph,self.embed_state=nnx.split(self.model)
            def encode_view(st,batch):
                mdl=nnx.merge(self.embed_graph,st); tokens,_=mdl.PaliGemma.img(batch,train=False)
                return jnp.mean(tokens.astype(jnp.float32),axis=1)
            self.image_embed=jax.jit(encode_view)
            if self.task_conditioned_view:
                def encode_task(st,tokens):
                    mdl=nnx.merge(self.embed_graph,st)
                    return mdl.PaliGemma.llm(tokens,method="embed")
                self.task_embed=jax.jit(encode_task)
        self.action_gain=None
        self.action_gain_names=None
        self.action_gain_commands=None
        if args.action_gain_checkpoint is not None:
            gain_payload=torch.load(args.action_gain_checkpoint,map_location="cpu",weights_only=False)
            if self.view_scorer is None: raise ValueError("--action-gain-checkpoint requires --view-quality-checkpoint")
            if int(gain_payload["pca_dim"]) != q: raise ValueError(f"Action gain PCA dim {gain_payload['pca_dim']} != view PCA dim {q}")
            self.action_gain=nn.Sequential(nn.Linear(q+6,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
            self.action_gain.load_state_dict({k.removeprefix("net."):v for k,v in gain_payload["model"].items()});self.action_gain.eval()
            self.action_gain_names=list(gain_payload["action_names"])
            self.action_gain_commands=np.asarray(gain_payload["action_commands"],np.float32)
        self.continuous_policy=None
        if args.continuous_policy_checkpoint is not None:
            cp=torch.load(args.continuous_policy_checkpoint,map_location="cpu",weights_only=False)
            cq=int(cp["pca_V"].shape[1])
            class ContinuousPolicy(nn.Module):
                def __init__(self,d):
                    super().__init__();self.trunk=nn.Sequential(nn.Linear(d,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(.10),nn.Linear(192,128),nn.GELU());self.action=nn.Linear(128,3);self.direction=nn.Linear(128,6)
                def forward(self,z):
                    h=self.trunk(z);return torch.tanh(self.action(h)),self.direction(h)
            self.continuous_policy=ContinuousPolicy(cq);self.continuous_policy.load_state_dict(cp["model"]);self.continuous_policy.eval()
            self.cont_feature_mean=np.asarray(cp["feature_mean"],np.float32);self.cont_feature_std=np.asarray(cp["feature_std"],np.float32)
            self.cont_pca_v=np.asarray(cp["pca_V"],np.float32);self.cont_scale=np.asarray(cp["action_scale"],np.float32)
            self.cont_names=list(cp["action_names"])
            if self.view_scorer is None:raise ValueError("--continuous-policy-checkpoint requires --view-quality-checkpoint for shared image encoding")
        self.hierarchical_policy=None
        if args.hierarchical_policy_checkpoint is not None:
            hp=torch.load(args.hierarchical_policy_checkpoint,map_location="cpu",weights_only=False);hq=int(hp["pca_V"].shape[1])
            class HierarchicalPolicy(nn.Module):
                def __init__(self,d):
                    super().__init__();self.trunk=nn.Sequential(nn.Linear(d,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,128),nn.GELU());self.mode=nn.Linear(128,2);self.translation=nn.Linear(128,4);self.yaw=nn.Linear(128,2)
                def forward(self,x):h=self.trunk(x);return self.mode(h),self.translation(h),self.yaw(h)
            self.hierarchical_policy=HierarchicalPolicy(hq);self.hierarchical_policy.load_state_dict(hp["model"]);self.hierarchical_policy.eval();self.hier_feature_mean=np.asarray(hp["feature_mean"],np.float32);self.hier_feature_std=np.asarray(hp["feature_std"],np.float32);self.hier_pca_v=np.asarray(hp["pca_V"],np.float32);self.hier_translation_names=list(hp["translation_names"]);self.hier_yaw_names=list(hp["yaw_names"])
            if self.view_scorer is None:raise ValueError("--hierarchical-policy-checkpoint requires --view-quality-checkpoint")
        self.spatial_hierarchical_policy=None
        if args.spatial_hierarchical_policy_checkpoint is not None:
            sp=torch.load(args.spatial_hierarchical_policy_checkpoint,map_location="cpu",weights_only=False)
            class SpatialHierarchicalPolicy(nn.Module):
                def __init__(self):
                    super().__init__();self.token=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,64),nn.GELU());self.pos=nn.Parameter(torch.zeros(16,64));self.trunk=nn.Sequential(nn.Linear(1024,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU());self.mode=nn.Linear(128,2);self.translation=nn.Linear(128,4);self.yaw=nn.Linear(128,2)
                def forward(self,x):
                    h=self.token(x.reshape(-1,16,2048))+self.pos;h=self.trunk(h.flatten(1));return self.mode(h),self.translation(h),self.yaw(h)
            self.spatial_hierarchical_policy=SpatialHierarchicalPolicy();self.spatial_hierarchical_policy.load_state_dict(sp["model"]);self.spatial_hierarchical_policy.eval();self.spatial_translation_names=list(sp["translation_names"]);self.spatial_yaw_names=list(sp["yaw_names"])
            if self.view_scorer is None:raise ValueError("--spatial-hierarchical-policy-checkpoint requires --view-quality-checkpoint")
            def encode_spatial(st,batch):
                mdl=nnx.merge(self.embed_graph,st);tokens,_=mdl.PaliGemma.img(batch,train=False);b,n,d=tokens.shape;return tokens.reshape((b,16,16,d)).reshape((b,4,4,4,4,d)).mean(axis=(2,4)).astype(jnp.float32)
            self.spatial_embed=jax.jit(encode_spatial)
        norm = json.loads((args.base_checkpoint / "assets/norm_stats.json").read_text())["norm_stats"]
        self.state_mean = np.asarray(norm["state"]["mean"], np.float32)
        self.state_std = np.maximum(np.asarray(norm["state"]["std"], np.float32), 1e-6)
        self.action_mean = np.asarray(norm["actions"]["mean"], np.float32)
        self.action_std = np.maximum(np.asarray(norm["actions"]["std"], np.float32), 1e-6)
        self._metadata = {
            "stage": "mobility",
            "arm_locked": True,
            "adapter": str(args.adapter),
            "base_checkpoint": str(args.base_checkpoint),
            "merged_parameters": merged,
        }

    @property
    def metadata(self):
        return self._metadata

    def infer(self, raw: dict) -> dict:
        main = np.asarray(raw["observation/image"], dtype=np.float32)
        wrist = np.asarray(raw["observation/wrist_image"], dtype=np.float32)
        state_raw = np.asarray(raw["observation/state"], dtype=np.float32).reshape(-1)
        if main.shape != (224, 224, 3) or wrist.shape != (224, 224, 3):
            raise ValueError(f"Expected 224x224 RGB inputs, got {main.shape} and {wrist.shape}")
        state = np.zeros(32, np.float32)
        state[: min(32, state_raw.size)] = state_raw[:32]
        state = (state - self.state_mean) / self.state_std
        real_task = str(raw.get("prompt", "")).strip()
        if not real_task:
            raise ValueError("A real task prompt is required for mobility inference")
        task_prompt = real_task.lower()
        movement_prompt = self.args.movement_prompt or (
            f"Task: {real_task} Move the mobile base to a viewpoint suitable for this task."
        )
        feedback = raw.get("view_feedback")
        if isinstance(feedback, dict):
            score=float(feedback.get("score",0.0));delta=float(feedback.get("delta",0.0));trend=str(feedback.get("trend","unknown"))
            movement_prompt += (f" View quality={score:.3f}; change={delta:+.3f} ({trend}). "
                                "Improve it using only the mobile base; keep the arm still.")
        tokens, token_mask = self.tokenizer.tokenize(movement_prompt, state)
        right = np.zeros_like(main)
        obs = model_lib.Observation(
            images={
                "base_0_rgb": jnp.asarray(main[None] / 127.5 - 1.0),
                "left_wrist_0_rgb": jnp.asarray(wrist[None] / 127.5 - 1.0),
                "right_wrist_0_rgb": jnp.asarray(right[None] / 127.5 - 1.0),
            },
            image_masks={
                "base_0_rgb": jnp.ones(1, dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones(1, dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.zeros(1, dtype=jnp.bool_),
            },
            state=jnp.asarray(state[None]),
            tokenized_prompt=jnp.asarray(tokens[None]),
            tokenized_prompt_mask=jnp.asarray(token_mask[None]),
        )
        view_quality = None
        action_gains = None
        continuous_base_action = None
        continuous_direction_probabilities = None
        hierarchical_mode_probabilities = None
        hierarchical_translation_probabilities = None
        hierarchical_yaw_probabilities = None
        spatial_mode_probabilities = None
        spatial_translation_probabilities = None
        spatial_yaw_probabilities = None
        if self.view_scorer is not None:
            feature=np.asarray(self.image_embed(self.embed_state,jnp.asarray(main[None]/127.5-1.0))[0])
            state16=np.zeros(16,np.float32); state16[:min(16,state_raw.size)]=state_raw[:16]
            x=np.r_[feature,state16]
            z=((x-self.feature_mean)/np.maximum(self.feature_std,1e-6))@self.pca_v
            with torch.no_grad():
                zt=torch.tensor(z,dtype=torch.float32)
                if self.task_conditioned_view:
                    task_ids,task_mask=self.tokenizer.tokenize(real_task,None)
                    task_tokens=np.asarray(self.task_embed(self.embed_state,jnp.asarray(task_ids)[None])[0],np.float32)
                    task_vec=torch.tensor(task_tokens[np.asarray(task_mask,bool)].mean(0),dtype=torch.float32)
                    view_quality=float(torch.sigmoid(self.view_scorer(zt,task_vec)).item())
                else:
                    view_quality=float(torch.sigmoid(self.view_scorer(zt)).item())
                if self.action_gain is not None:
                    zr=zt.unsqueeze(0).repeat(6,1);onehot=torch.eye(6,dtype=torch.float32)
                    action_gains=self.action_gain(torch.cat([zr,onehot],dim=1)).reshape(-1).numpy()
                if self.continuous_policy is not None:
                    cz=((feature-self.cont_feature_mean)/np.maximum(self.cont_feature_std,1e-6))@self.cont_pca_v
                    ca,cl=self.continuous_policy(torch.tensor(cz,dtype=torch.float32))
                    continuous_base_action=(ca.numpy()*self.cont_scale).astype(np.float32)
                    continuous_direction_probabilities=torch.softmax(cl,dim=-1).numpy().astype(np.float32)
                if self.hierarchical_policy is not None:
                    hz=((feature-self.hier_feature_mean)/np.maximum(self.hier_feature_std,1e-6))@self.hier_pca_v
                    hm,ht,hy=self.hierarchical_policy(torch.tensor(hz,dtype=torch.float32))
                    hierarchical_mode_probabilities=torch.softmax(hm,dim=-1).numpy().astype(np.float32)
                    hierarchical_translation_probabilities=torch.softmax(ht,dim=-1).numpy().astype(np.float32)
                    hierarchical_yaw_probabilities=torch.softmax(hy,dim=-1).numpy().astype(np.float32)
                if self.spatial_hierarchical_policy is not None:
                    grid=np.asarray(self.spatial_embed(self.embed_state,jnp.asarray(main[None]/127.5-1.0)))
                    sm,st,sy=self.spatial_hierarchical_policy(torch.tensor(grid,dtype=torch.float32))
                    spatial_mode_probabilities=torch.softmax(sm[0],dim=-1).numpy().astype(np.float32)
                    spatial_translation_probabilities=torch.softmax(st[0],dim=-1).numpy().astype(np.float32)
                    spatial_yaw_probabilities=torch.softmax(sy[0],dim=-1).numpy().astype(np.float32)
        if bool(raw.get("score_only",False)):
            return {
                "actions":np.zeros((1,12),np.float32),
                "stage":"view_score",
                "arm_locked":True,
                "score_only":True,
                "view_quality":view_quality,
                "action_gains":action_gains,
                "action_gain_names":self.action_gain_names,
                "action_gain_commands":self.action_gain_commands,
                "continuous_base_action":continuous_base_action,
                "continuous_direction_probabilities":continuous_direction_probabilities,
                "continuous_direction_names":self.cont_names if self.continuous_policy is not None else None,
                "hierarchical_mode_probabilities":hierarchical_mode_probabilities,
                "hierarchical_translation_probabilities":hierarchical_translation_probabilities,
                "hierarchical_yaw_probabilities":hierarchical_yaw_probabilities,
                "hierarchical_translation_names":self.hier_translation_names if self.hierarchical_policy is not None else None,
                "hierarchical_yaw_names":self.hier_yaw_names if self.hierarchical_policy is not None else None,
                "spatial_mode_probabilities":spatial_mode_probabilities,
                "spatial_translation_probabilities":spatial_translation_probabilities,
                "spatial_yaw_probabilities":spatial_yaw_probabilities,
                "spatial_translation_names":self.spatial_translation_names if self.spatial_hierarchical_policy is not None else None,
                "spatial_yaw_names":self.spatial_yaw_names if self.spatial_hierarchical_policy is not None else None,
            }
        self.rng, sample_rng = jax.random.split(self.rng)
        normalized = np.asarray(self.sample(sample_rng, obs, num_steps=self.args.num_steps)[0])
        actions = normalized * self.action_std + self.action_mean
        safe = np.zeros((actions.shape[0], 12), np.float32)
        raw_base = np.clip(actions[:, 7:10], -self.args.max_base_command,
                           self.args.max_base_command)
        window = min(self.args.decision_action_window, len(raw_base))
        raw_command_strength = float(np.max(np.abs(raw_base[:window])))
        filtered_base = raw_base.copy()
        filtered_base[:, :2] = np.where(
            np.abs(filtered_base[:, :2]) < self.args.base_translation_deadband,
            0.0, filtered_base[:, :2])
        filtered_base[:, 2] = np.where(
            np.abs(filtered_base[:, 2]) < self.args.base_yaw_deadband,
            0.0, filtered_base[:, 2])
        safe[:, 7:10] = filtered_base
        safe[:, 11] = 1.0
        decision_base = safe[:window, 7:10]
        command_strength = float(np.max(np.abs(decision_base)))
        base_has_action = bool(np.any(decision_base != 0.0))
        return {
            "actions": safe,
            "stage": "mobility",
            "arm_locked": True,
            "mobility_command_strength": command_strength,
            "mobility_raw_command_strength": raw_command_strength,
            "mobility_base_has_action": base_has_action,
            "mobility_raw_base_window": raw_base[:window].tolist(),
            "mobility_filtered_base_window": decision_base.tolist(),
            "view_quality": view_quality,
            "action_gains": action_gains,
            "action_gain_names": self.action_gain_names,
            "action_gain_commands": self.action_gain_commands,
            "mobility_prompt": movement_prompt,
        }

    def close(self):
        self.sample = None
        if hasattr(self,"image_embed"): self.image_embed=None
        if hasattr(self,"spatial_embed"): self.spatial_embed=None
        self.model = None
        gc.collect()
        jax.clear_caches()
        gc.collect()


class SequentialDrawerPolicy:
    def __init__(self, args):
        self.args = args
        self.mobility = MobilityPolicy(args)
        self.grasp = None
        self.mode = "mobility"
        self.low_count = 0
        self.quality_count = 0
        self.mobility_inferences = 0
        self.motion_observed = False
        self.last_mobility_infer_time = None
        self.last_request_completed_at = None
        if self.args.decision_log is not None:
            self.args.decision_log.parent.mkdir(parents=True,exist_ok=True)

    def _write_decision(self, result, switch_reason=None):
        if self.args.decision_log is None: return
        row={"time":time.time(),"inference":self.mobility_inferences,"raw_strength":result.get("mobility_raw_command_strength"),"filtered_strength":result.get("mobility_command_strength"),"raw_base_window":result.get("mobility_raw_base_window"),"filtered_base_window":result.get("mobility_filtered_base_window"),"base_has_action":result.get("mobility_base_has_action"),"motion_observed":self.motion_observed,"low_count":self.low_count,"view_quality":result.get("view_quality"),"quality_count":self.quality_count,"switch_reason":switch_reason}
        with self.args.decision_log.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")

    @property
    def metadata(self):
        return {
            **self.mobility.metadata,
            "requested_stage": self.args.stage,
            "switch_rule": {
                "threshold": self.args.stop_command_threshold,
                "confirmations": self.args.stop_confirmations,
                "motion_start_threshold": self.args.motion_start_threshold,
                "minimum_mobility_inferences": self.args.minimum_mobility_inferences,
                "maximum_mobility_inferences": self.args.maximum_mobility_inferences,
                "translation_deadband": self.args.base_translation_deadband,
                "yaw_deadband": self.args.base_yaw_deadband,
                "decision_action_window": self.args.decision_action_window,
                "view_quality_checkpoint": str(self.args.view_quality_checkpoint) if self.args.view_quality_checkpoint else None,
                "view_quality_threshold": self.args.view_quality_threshold,
                "low_command_rule_requires_prior_motion": False,
                "maximum_inference_guard_requires_prior_motion": False,
            },
        }

    def _load_grasp(self):
        logging.info("B-like view confirmed. Releasing mobility policy before loading original grasp policy")
        self.mobility.close()
        self.mobility = None
        self.grasp = policy_config.create_trained_policy(
            train_config_lib.get_config("pi05_pretrain_human300_inference"),
            self.args.base_checkpoint,
            sample_kwargs={"num_steps": self.args.num_steps},
        )
        self.mode = "grasp"
        logging.info("Stage 2 active: original 75000 policy controls the arm and drawer interaction")

    def _reload_mobility_for_new_attempt(self):
        logging.info("New attempt detected after idle gap; releasing grasp policy and reloading mobility policy")
        self.grasp = None
        gc.collect()
        jax.clear_caches()
        gc.collect()
        self.mobility = MobilityPolicy(self.args)
        self.mode = "mobility"
        self.low_count = 0
        self.quality_count = 0
        self.mobility_inferences = 0
        self.motion_observed = False
        self.last_mobility_infer_time = None
        logging.info("Mobility stage reset for the new attempt")

    def infer(self, obs: dict) -> dict:
        request_started = time.monotonic()
        if bool(obs.get("reset_mobility",False)):
            if self.mode == "grasp":
                self._reload_mobility_for_new_attempt()
            else:
                self.low_count = 0
                self.quality_count = 0
                self.mobility_inferences = 0
                self.motion_observed = False
                self.last_mobility_infer_time = None
            self.last_request_completed_at=time.monotonic()
            return {
                "actions":np.zeros((1,12),np.float32),
                "stage":"mobility",
                "mobility_reset":True,
                "arm_locked":True,
            }
        if bool(obs.get("score_only",False)):
            if self.mode != "mobility" or self.mobility is None:
                raise RuntimeError("score_only is available only during mobility stage")
            result=self.mobility.infer(obs)
            self.last_request_completed_at=time.monotonic()
            return result
        if bool(obs.get("force_grasp_handoff",False)):
            if self.args.stage != "full":
                raise RuntimeError("force_grasp_handoff requires --stage full")
            if self.mode == "mobility":
                self._load_grasp()
            self.last_request_completed_at=time.monotonic()
            return {
                "actions":np.zeros((1,12),np.float32),
                "stage":"grasp",
                "switching_to_grasp":True,
                "switch_reason":"client_confirmed_high_view_quality",
                "arm_locked":False,
                "base_locked":True,
            }
        # Episode boundaries are explicit (reset_mobility); inference or simulation
        # latency must never be interpreted as a new episode.
        if self.mode == "grasp":
            result = self.grasp.infer(obs)
            grasp_actions = np.asarray(result["actions"]).copy()
            # Hold the learned B viewpoint while the original policy manipulates.
            grasp_actions[:, 7:11] = 0.0
            grasp_actions[:, 11] = -1.0
            result["actions"] = grasp_actions
            result["stage"] = "grasp"
            result["arm_locked"] = False
            result["base_locked"] = True
            self.last_request_completed_at = time.monotonic()
            return result
        result = self.mobility.infer(obs)
        now = time.monotonic()
        if (
            self.last_mobility_infer_time is not None
            and now - self.last_mobility_infer_time > self.args.mobility_reset_idle_seconds
        ):
            self.low_count = 0
            self.mobility_inferences = 0
            self.motion_observed = False
        self.last_mobility_infer_time = now
        strength = result["mobility_command_strength"]
        raw_strength = result["mobility_raw_command_strength"]
        base_has_action = result["mobility_base_has_action"]
        view_quality = result.get("view_quality")
        self.mobility_inferences += 1
        if strength >= self.args.motion_start_threshold:
            self.motion_observed = True
        can_consider_stopping = self.mobility_inferences >= self.args.minimum_mobility_inferences
        self.low_count = (
            self.low_count + 1
            if can_consider_stopping and self.motion_observed and not base_has_action
            else 0
        )
        self.quality_count = self.quality_count + 1 if (can_consider_stopping and view_quality is not None and view_quality >= self.args.view_quality_threshold) else 0
        result["stop_confirmation_count"] = self.low_count
        result["mobility_inferences"] = self.mobility_inferences
        result["motion_observed"] = self.motion_observed
        result["view_quality_confirmation_count"] = self.quality_count
        logging.info(
            "mobility inference=%d raw_strength=%.4f filtered_strength=%.4f "
            "base_has_action=%s motion_observed=%s stop_count=%d view_quality=%s quality_count=%d",
            self.mobility_inferences, raw_strength, strength, base_has_action,
            self.motion_observed, self.low_count, f'{view_quality:.3f}' if view_quality is not None else 'disabled', self.quality_count,
        )
        maximum_reached = (
            self.args.maximum_mobility_inferences > 0
            and self.mobility_inferences >= self.args.maximum_mobility_inferences
        )
        result["maximum_mobility_inferences_reached"] = maximum_reached
        if self.args.stage == "full" and (
            self.quality_count >= self.args.view_quality_confirmations or self.low_count >= self.args.stop_confirmations or maximum_reached
        ):
            result["switching_to_grasp"] = True
            result["switch_reason"] = (
                "confirmed_high_view_quality" if self.quality_count >= self.args.view_quality_confirmations else ("confirmed_low_mobility_command_after_motion" if self.low_count >= self.args.stop_confirmations else "maximum_mobility_inferences")
            )
            result["actions"][:] = 0.0
            self._write_decision(result,result["switch_reason"])
            self._load_grasp()
        else:
            self._write_decision(result)
        self.last_request_completed_at = time.monotonic()
        return result


def preflight(args):
    if not (args.base_checkpoint / "params").exists():
        raise FileNotFoundError(args.base_checkpoint / "params")
    if not (args.base_checkpoint / "assets/norm_stats.json").exists():
        raise FileNotFoundError(args.base_checkpoint / "assets/norm_stats.json")
    if not args.adapter.exists():
        raise FileNotFoundError(args.adapter)
    with np.load(args.adapter, allow_pickle=False) as z:
        keys = sorted(z.files)
    if len(keys) != 10:
        raise RuntimeError(f"Expected 10 LoRA arrays, found {len(keys)}")
    if args.view_quality_checkpoint is not None and not args.view_quality_checkpoint.exists():
        raise FileNotFoundError(args.view_quality_checkpoint)
    if args.action_gain_checkpoint is not None and not args.action_gain_checkpoint.exists():
        raise FileNotFoundError(args.action_gain_checkpoint)
    if args.continuous_policy_checkpoint is not None and not args.continuous_policy_checkpoint.exists():
        raise FileNotFoundError(args.continuous_policy_checkpoint)
    if args.hierarchical_policy_checkpoint is not None and not args.hierarchical_policy_checkpoint.exists():
        raise FileNotFoundError(args.hierarchical_policy_checkpoint)
    if args.spatial_hierarchical_policy_checkpoint is not None and not args.spatial_hierarchical_policy_checkpoint.exists():
        raise FileNotFoundError(args.spatial_hierarchical_policy_checkpoint)
    print(json.dumps({
        "status": "preflight_ok",
        "stage": args.stage,
        "base_checkpoint": str(args.base_checkpoint),
        "adapter": str(args.adapter),
        "adapter_arrays": len(keys),
        "movement_prompt": args.movement_prompt,
        "arm_locked_during_mobility": True,
        "view_quality_checkpoint": str(args.view_quality_checkpoint) if args.view_quality_checkpoint else None,
        "action_gain_checkpoint": str(args.action_gain_checkpoint) if args.action_gain_checkpoint else None,
        "continuous_policy_checkpoint": str(args.continuous_policy_checkpoint) if args.continuous_policy_checkpoint else None,
        "hierarchical_policy_checkpoint": str(args.hierarchical_policy_checkpoint) if args.hierarchical_policy_checkpoint else None,
        "spatial_hierarchical_policy_checkpoint": str(args.spatial_hierarchical_policy_checkpoint) if args.spatial_hierarchical_policy_checkpoint else None,
    }, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if not 1 <= args.num_steps <= 10:
        raise ValueError("--num-steps must be 1..10")
    if not 0 < args.max_base_command <= 1:
        raise ValueError("--max-base-command must be in (0, 1]")
    if not 0 <= args.base_translation_deadband < args.max_base_command:
        raise ValueError("translation deadband must be in [0, max-base-command)")
    if not 0 <= args.base_yaw_deadband < args.max_base_command:
        raise ValueError("yaw deadband must be in [0, max-base-command)")
    if args.decision_action_window < 1:
        raise ValueError("decision-action-window must be positive")
    if args.stop_confirmations < 1:
        raise ValueError("--stop-confirmations must be positive")
    if not 0 < args.view_quality_threshold < 1: raise ValueError("view-quality-threshold must be in (0,1)")
    if args.view_quality_confirmations < 1: raise ValueError("view-quality-confirmations must be positive")
    if not args.stop_command_threshold < args.motion_start_threshold <= 1:
        raise ValueError("Require stop threshold < motion-start threshold <= 1")
    if args.minimum_mobility_inferences < 1:
        raise ValueError("--minimum-mobility-inferences must be positive")
    if args.maximum_mobility_inferences < 0:
        raise ValueError("--maximum-mobility-inferences must be nonnegative")
    if (args.maximum_mobility_inferences and
            args.maximum_mobility_inferences < args.minimum_mobility_inferences):
        raise ValueError("maximum mobility inferences must be >= minimum")
    if args.episode_reset_idle_seconds <= 0:
        raise ValueError("--episode-reset-idle-seconds must be positive")
    if args.mobility_reset_idle_seconds <= 0:
        raise ValueError("--mobility-reset-idle-seconds must be positive")
    if args.preflight:
        preflight(args)
        return
    policy = SequentialDrawerPolicy(args)
    host_name = socket.gethostname()
    logging.info("Serving %s mode on %s:%d (%s)", args.stage, args.host, args.port, host_name)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
