"""GPU interface smoke check with synthetic inputs; never controls a simulator."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['lora','external'],required=True)
    p.add_argument('--base-checkpoint',required=True)
    p.add_argument('--weights',type=Path,default=Path(__file__).parent/'weights')
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).parent/'src'))
    sys.argv=[sys.argv[0],'--base-checkpoint',a.base_checkpoint,'--adapter',str(a.weights/'pure_mobility_lora_v2.npz'),'--num-steps','1']
    if a.mode=='external':
        import serve_joint_offline_value_reposition_v1 as module
        sys.argv+=['--view-quality-checkpoint',str(a.weights/'external_joint_reposition_v1.pt')]
        base=module.base
    else:
        import serve_seed17_drawer_stage12 as base
        sys.argv+=['--movement-prompt','Move the mobile base to a suitable viewpoint for grasping the drawer handle.']
    args=base.parse_args();started=time.time();policy=base.MobilityPolicy(args)
    raw={'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),'observation/state':np.zeros(32,np.float32),'prompt':'Open the left drawer.'}
    if a.mode=='external':
        raw.update(score_only=True)
        raw['observation/base_qpos']=policy.joint_state_mean.copy()
    result=policy.infer(raw)
    actions=np.asarray(result['actions']);assert np.isfinite(actions).all()
    if a.mode=='external':
        probs=np.asarray(result['joint_action_probabilities'])
        assert probs.shape==(6,) and np.isfinite(probs).all() and np.isclose(probs.sum(),1)
        assert 0<=result['joint_stop_probability']<=1
    else:
        assert actions.shape==(16,12),actions.shape
        assert np.allclose(actions[:,:7],0)
    print(json.dumps({'mode':a.mode,'status':'synthetic_gpu_inference_ok','actions_shape':list(actions.shape),'elapsed_seconds':time.time()-started,'performance_evaluation':False}),flush=True)
    policy.close()
if __name__=='__main__':main()
