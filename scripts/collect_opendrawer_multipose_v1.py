#!/usr/bin/env python3
"""Evaluate fixed OpenDrawer scenes at five reproducible robot base poses."""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path("/root/gpufree-data")
PILOT = ROOT / "view_discriminator_pilot"
for p in (PILOT, ROOT/"robocasa", ROOT/"robosuite", ROOT/"openpi/examples/robocasa"):
    sys.path.insert(0, str(p))
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
import collect_opendrawer_default100x10_v1 as base
sweep = base.sweep
original_reset = sweep._reset_for_episode

def complete(path, trials):
    s = base.read_json(path/"summary.json") or {}
    st = (((s.get("variants") or {}).get("baseline") or {}).get("stats") or {})
    return int(st.get("num_episodes", -1)) == trials and len(st.get("episode_results") or []) == trials

def set_reset_pose(target):
    target = np.asarray(target, np.float64).reshape(3).copy()
    def reset(env, args, fixed_meta):
        _, info = original_reset(env, args, fixed_meta)
        e = sweep._base_robocasa_env(env); robot = e.robots[0]
        qi = np.asarray(robot._ref_base_joint_pos_indexes, np.int64)
        vi = np.asarray([e.sim.model.get_joint_qvel_addr(n) for n in robot.robot_model.base_joints], np.int64)
        if qi.shape != (3,) or vi.shape != (3,):
            raise RuntimeError(f"Expected planar base, got {qi.shape}/{vi.shape}")
        e.sim.data.qpos[qi] = target; e.sim.data.qvel[vi] = 0; e.sim.forward()
        return sweep._refresh_observation(env), info
    sweep._reset_for_episode = reset

def poses_for(seed, scene_dir, image_size):
    root = scene_dir/"pose_generation"; d = root/f"seed{seed:03d}"
    if not (d/"poses.json").exists() or len(list(d.glob("*_base_qpos.npy"))) != 5:
        subprocess.run([sys.executable, str(PILOT/"preview_opendrawer_same_scene_multipose_v2.py"),
                        "--seed", str(seed), "--image-size", str(image_size), "--out-root", str(root)], check=True)
    manifest = json.loads((d/"poses.json").read_text(encoding="utf-8"))
    meta = {x["pose"]: x for x in manifest["poses"]}; result = []
    for f in sorted(d.glob("*_base_qpos.npy")):
        name = f.name.split("_",1)[1].removesuffix("_base_qpos.npy")
        result.append((name, np.load(f, allow_pickle=False), meta[name]))
    if len(result) != 5: raise RuntimeError(f"Expected five poses in {d}")
    return result

def aggregate(out, worker, seeds, trials):
    rows=[]
    for seed in seeds:
        scene=out/f"scene_seed_{seed:03d}"
        for pose_dir in sorted(x for x in scene.iterdir() if x.is_dir() and x.name.startswith("P")):
            s=base.read_json(pose_dir/"summary.json") or {}; st=(((s.get("variants") or {}).get("baseline") or {}).get("stats") or {})
            eps=st.get("episode_results") or []
            if not eps: continue
            req=base.read_json(pose_dir/"requested_pose.json") or {}; succ=sum(bool(x.get("success")) for x in eps)
            rows.append({"task":"OpenDrawer","scene_seed":seed,"pose":pose_dir.name,"worker":worker,
              "successes":succ,"trials":len(eps),"success_rate":succ/len(eps),
              "success_rate_beta11":(succ+1)/(len(eps)+2),"pose_metadata":req.get("pose_metadata"),
              "initial_geometry":((eps[0].get("geometry") or {}).get("initial") or {}),
              "right_external_camera_masked":True,"scene_pose_dir":str(pose_dir),
              "base_checkpoint":str(base.BASE_CHECKPOINT)})
    (out/f"feasibility_observations_worker{worker}.jsonl").write_text(
        "".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",required=True); p.add_argument("--worker",type=int,required=True)
    p.add_argument("--trials-per-pose",type=int,default=10); p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,required=True); p.add_argument("--horizon",type=int,default=330)
    p.add_argument("--image-size",type=int,default=256); p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--resume",action="store_true",default=True); a=p.parse_args()
    seeds=base.parse_seed_spec(a.seeds); a.out_root.mkdir(parents=True,exist_ok=True)
    sweep._aim_camera_at_target=base.default_camera_metadata
    sweep._geometry_snapshot=base.geometry_snapshot_with_relation
    sweep.imageio.mimwrite=base.compact_mimwrite
    for seed in seeds:
        scene=a.out_root/f"scene_seed_{seed:03d}"; scene.mkdir(parents=True,exist_ok=True)
        for name,qpos,meta in poses_for(seed,scene,a.image_size):
            out=scene/name
            if a.resume and complete(out,a.trials_per_pose): print(f"[skip] seed={seed} pose={name}",flush=True); continue
            out.mkdir(parents=True,exist_ok=True)
            (out/"requested_pose.json").write_text(json.dumps({"base_qpos":qpos.tolist(),"pose_metadata":meta},ensure_ascii=False,indent=2),encoding="utf-8")
            set_reset_pose(qpos); print(f"[run] worker={a.worker} seed={seed} pose={name}",flush=True)
            ea=sweep.Args(host=a.host,port=a.port,task="OpenDrawer",split="pretrain",num_trials=a.trials_per_pose,
              seed=seed,same_scene=True,freeze_episode_meta=True,horizon_override=a.horizon,disable_right_image=True,
              record_geometry=True,variants=["baseline"],look_at="none",log_dir=str(out),render_every=1_000_000,resume=a.resume)
            sweep.eval_main(ea); aggregate(a.out_root,a.worker,seeds,a.trials_per_pose)
    rows=aggregate(a.out_root,a.worker,seeds,a.trials_per_pose)
    print(json.dumps({"status":"complete","worker":a.worker,"pairs":len(rows)},ensure_ascii=False),flush=True)
if __name__=="__main__": main()

