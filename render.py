#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Sicheng modified @ July, 4th, 2024

import json
import os
import random
import torch
from torch import nn
import glob
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
from PIL import Image
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from utils.general_utils import get_timestamp_str
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
import time
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

import fpnge

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def rendering(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size, decompress=False):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    gaussians.training_setup(opt)
    
    if checkpoint:
        print("Loading ckpt!")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, None)
        print("Finsh loading ckpt!")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # import pdb; pdb.set_trace()
    visualization = False
    if visualization:
        with torch.no_grad():
            # save mu_t for time mean
            torch.save(gaussians.get_t.cpu(), os.path.join(scene.model_path, 'gs_lifecenter.pth'))
            # save 6*sqrt(sigma) for time duration
            gs_lifetime = (gaussians.get_cov_t(1).sqrt()*6).cpu()
            torch.save(gs_lifetime, os.path.join(scene.model_path, 'gs_lifetime.pth'))


    if decompress:
        from compression.compression_exp import run_compressions, run_decompressions
        # import pdb; pdb.set_trace()
        for decompress_dir, decompressed_gaussian in run_decompressions(os.path.join(os.path.dirname(args.start_checkpoint), 
                                                                                      'compression', 
                                                                                      'post_training')): # 'iter_30000'
            try:
                decompress_path = os.path.join(os.path.dirname(args.start_checkpoint), 'compression', 'post_training', decompress_dir)
                tb_writer = prepare_output_and_logger(decompress_path)

                # to obatin rate
                filesize_log = os.path.join(os.path.dirname(args.start_checkpoint), 'compression', 'post_training', 'filesize.json')
                with open(filesize_log, 'r') as fp:
                    filesize_dict = json.load(fp)
                total_size = filesize_dict[decompress_dir]

                tb_writer.add_scalar('compression' + '/total size (MB)', total_size/1e6, 30000)
                tb_writer.add_scalar('compression' + '/bitrate (Mbps)', total_size/1e6/(time_duration[1] - time_duration[0])*8, 30000)

                # to obtain distortion
                scene.gaussians = decompressed_gaussian
                test_psnr = rendering_report(tb_writer, 30000, None, None, l1_loss, None, None, scene, render, (pipe, background), None, decompress_path)
                # import pdb; pdb.set_trace()
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Detailed error information: {repr(e)}")
    else:
        test_psnr = rendering_report(None, 30000, None, None, l1_loss, None, None, scene, render, (pipe, background), None)

         

def prepare_output_and_logger(path):    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def save_tensor_as_png(tensor, filename):
    np_img = (tensor.cpu() * 255).clamp(0, 255).byte()
    img = transforms.ToPILImage()(np_img)
    ## save png using 'PIL' lib
    # img.save(save_renders+f"/{viewpoint.image_name}.png")
    ## save png using 'fpnge' lib
    png = fpnge.fromPIL(img)
    with open(filename, 'wb') as f:
        f.write(png)

def rendering_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None, save_dir=None):
    psnr_test_iter = 0.0
    # Report test and samples of training set
    
    if save_dir is None:
        save_dir = scene.model_path+'/test/final'
    os.makedirs(save_dir, exist_ok=True)
    save_gt = save_dir+'/gt'
    os.makedirs(save_gt, exist_ok=True)
    save_renders = save_dir+'/renders'
    os.makedirs(save_renders, exist_ok=True)

    config = {'name': 'test', 'cameras' : scene.getTestCameras()}
    render_time = []
    with torch.no_grad():
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            msssim_test = 0.0
            # for idx in tqdm(range(len(config['cameras']))):
            for idx, batch_data in enumerate(tqdm(config['cameras'])):
                # batch_data = config['cameras'][idx]
                gt_image, viewpoint = batch_data
                gt_image = gt_image.cuda()
                viewpoint = viewpoint.cuda()
                
                torch.cuda.synchronize()
                start_time = time.time()

                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                render_time.append(elapsed_time)

                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                
                depth = easy_cmap(render_pkg['depth'][0])
                # if idx % 20 == 19:
                #     import pdb; pdb.set_trace()
                # alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                # if tb_writer and (idx < 5):
                #     grid = [gt_image, image, alpha, depth]
                #     grid = make_grid(grid, nrow=2)
                #     tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                        
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())

                # save imgs
                save_tensor_as_png(tensor=image, filename=save_renders+f"/{viewpoint.image_name}.png")
                save_tensor_as_png(tensor=gt_image, filename=save_gt+f"/{viewpoint.image_name}.png")
                
                del render_pkg
                del gt_image
                del viewpoint
                del image
                del depth
                # del alpha
                # del grid
                torch.cuda.empty_cache()

        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras']) 
        ssim_test /= len(config['cameras'])     
        msssim_test /= len(config['cameras'])        
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        if tb_writer:
            # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
        if config['name'] == 'test':
            psnr_test_iter = psnr_test.item()
    
    # concat frames from test views to form videos
    test_view_ids = [os.path.basename(vid)[:3] for vid in sorted(glob.glob(save_renders+"/*_0000.png"))]
    for vid in test_view_ids:
        cmd = f"ffmpeg -framerate 30 -i {save_renders}/{vid}_%04d.png -c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p {save_renders}/{vid}.mp4"
        os.system(cmd)
    
    return psnr_test_iter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")

    parser.add_argument("--decompress", action="store_true")
    
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
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,op.iterations,3000)]
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    rendering(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size, args.decompress)

    # All done
    print("\nRendering complete.")