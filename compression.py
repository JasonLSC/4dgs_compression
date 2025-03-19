import json
import sys
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
import os

from arguments import ModelParams, PipelineParams, OptimizationParams, get_compression_cfg
from scene import Scene, GaussianModel
from compression.compression_exp import run_compressions



if __name__ == "__main__":
    # Argument parser
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    cp = get_compression_cfg('./configs/compression_in_training.yaml')

    # Gaussian init.
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)
    if dataset.frame_ratio > 1:
        time_duration = [args.time_duration[0] / dataset.frame_ratio,  args.time_duration[1] / dataset.frame_ratio]
    else:
        time_duration = args.time_duration

    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim, 
                              time_duration=time_duration, rot_4d=args.rot_4d, 
                              force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0,
                              pruning_flag=cp.pruning.enabled)
    gaussians.training_setup(opt) # TODO: discard it in the future 

    
    # Load checkpoint
    if args.checkpoint:
        print("Start loading model")
        (model_params, first_iter) = torch.load(args.checkpoint)
        gaussians.restore(model_params, opt)
        print("Finish loading model")

    # Compress
    compr_path = os.path.join(os.path.dirname(args.checkpoint), "compression", f"post_training")
    gaussians.prune_to_square_shape() # It should not prune any pts

    compress_cfg_details = OmegaConf.load("./configs/post_training_compression.yaml")

    compr_results = run_compressions(gaussians, compr_path, OmegaConf.to_container(compress_cfg_details))

    with open(os.path.join(compr_path, 'filesize.json'), 'w') as fp:
        json.dump(compr_results, fp)