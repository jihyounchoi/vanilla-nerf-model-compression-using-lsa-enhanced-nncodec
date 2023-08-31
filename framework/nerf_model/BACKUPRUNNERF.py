import os, sys
import warnings
import numpy as np
import imageio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from framework.nerf_model.run_nerf_helpers import *

# llff, deepvoxels, blender, LINEMOD : Dataset Variations
from framework.nerf_model.load_llff import load_llff_data
from framework.nerf_model.load_deepvoxels import load_dv_data
from framework.nerf_model.load_blender import load_blender_data
from framework.nerf_model.load_LINEMOD import load_LINEMOD_data

np.random.seed(0)
DEBUG = False

# Ignores UserWarning
# If you want to debug carefully, remove these lines
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

####### Import Device Information from configs/DEVICE.txt and Apply to variable DEVICE #######
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # Join with the relative path to get the full path
# file_path = os.path.join(current_directory, "../../nnc", "DEVICE.txt")

# with open(file_path, 'r') as file:
#     device_string = file.readline().strip()

# # Check if the parsed device is valid and available.
# if "cuda" in device_string and not torch.cuda.is_available():
#     raise ValueError(f"{device_string} is not available on this machine.")
    
# DEVICE = torch.device(device_string)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################################################################################


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

# used as "network_query_fn"
# embeddirs_fn : contains POISITIONAL ENCODING
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    Prepares inputs and applies network 'fn'.
    Returns output without post processing (raw)
    """
    
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    
    outputs_flat = batchify(fn, netchunk)(embedded) # Simple iteration, fn is forward of nerf network
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid out of memory.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays
        
    # viewdir = pose(direction) of camera = spherical coordinate
    # Origianlly direction is consists of theta, and phi
    # But in code implementation, it is normalized and consists of 3 parameters.
    
    if use_viewdirs: # use_viewdirs : bool to input view direction
        viewdirs = rays_d
        
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    # all_ret : stores output of rendering
    # chunk : Number of rays to process simultaneously
    # batchify_rays : makes batch same size as chunk
    all_ret = batchify_rays(rays, chunk, **kwargs) 
    
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """
    Rendering function to do inference
    """
    
    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    for i, c2w in enumerate(tqdm(render_poses)):
        
        ## Key part of rendering : render() ##
        ''' render_kwargs example
        
        render_kwargs_train = 
        {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        }   
    
        '''
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        
        
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(nerf_wrapper, multires, i_embed, use_viewdirs, multires_views, 
                netchunk, basedir, perturb, N_importance, N_samples, 
                white_bkgd, raw_noise_std, dataset_type, no_ndc, lindisp):

    """Instantiate NeRF's MLP model.
    
    [CUSTOM MODIFIED]
    
    Changements:
        Just receives NeRFWrapper model, and optimizer from train()
        
        do something to return these
        
            render_kwargs_train = {
                'network_query_fn' : network_query_fn,
                'perturb' : perturb,
                'N_importance' : N_importance,
                'network_fine' : model_fine,
                'N_samples' : N_samples,
                'network_fn' : model,
                'use_viewdirs' : use_viewdirs,
                'white_bkgd' : white_bkgd,
                'raw_noise_std' : raw_noise_std,
            }
        
    """
    embed_fn, input_ch = get_embedder(multires, i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    
    if use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(multires_views, i_embed)
        
    output_ch = 5 if N_importance > 0 else 4
    skips = [4]
    
    model = nerf_wrapper.model

    model_fine = None
    if N_importance > 0:
        print('model_fine is used')
        model_fine = nerf_wrapper.model_fine

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=netchunk)

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : perturb,
        'N_importance' : N_importance,
        'network_fine' : model_fine,
        'N_samples' : N_samples,
        'network_fn' : model,
        'use_viewdirs' : use_viewdirs,
        'white_bkgd' : white_bkgd,
        'raw_noise_std' : raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if dataset_type != 'llff' or no_ndc:
        print('Without NDC')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test


# VOLUME RENDERING
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    
    # Discrete version of volume sampling
    # ReLU is used cuz there may negative value of density
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
    dists = z_vals[...,1:] - z_vals[...,:-1] # distance between sampled coordinates
    # [1e10] : Assume last distance as almost infinity
    
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(DEVICE)], -1)  # [N_rays, N_samples]
    
    # Normalize distances
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    # Sigmoid : To make rgb value to 0 ~ 1 (multiplied by 255 later)
    rgb = torch.sigmoid(raw[...,:3]) # [N_rays, N_samples, 3]
    noise = 0. # noise ~ Gaussian(,)
    
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).to(DEVICE) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
            
    # Volume Sampling
    # recap) raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    
    # weights = alpha * T_i
    # cumprod = cumulative product (누적곱)
    
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), (1.-alpha + 1e-10)], -1), -1)[:, :-1].to(DEVICE) # alpha * T_i
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(DEVICE), (1.-alpha + 1e-10).to(DEVICE)], -1), -1)[:, :-1] # alpha * T_i

    
    # rgb_map : C(r) = sum(weights * rgb)
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # / 좌우 순서가...맞나?
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw = False,
                lindisp = False,
                perturb = 0.,
                N_importance = 0,
                network_fine = None,
                white_bkgd = False,
                raw_noise_std = 0.,
                pytest = False,
                verbose = False):
    
    """Volumetric rendering."""
    
    ############################# UNPACKING ###############################
    N_rays = ray_batch.shape[0] # number of rays
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    #######################################################################
    
    ########################## STRATIFIED SAMPLING ########################
    '''recap) stratified sampling
    1. Uniformly split between tn ~ tf
    2. Sample single point of each interval, BY UNIFORM RANDOM SAMPLING
    '''

    # t_vals = torch.linspace(0., 1., steps=N_samples) # linspace between 0 ~ 1, number of N_samples
    t_vals = torch.linspace(0., 1., steps=N_samples).to(DEVICE)
    
    

    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(DEVICE)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    
    # pts : includes all sample points of rays
    # [N_rays, N_samples, 3]
    # rays_o + t * rays_d
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] 
    ########################################################################
    
    ##################### HIERARCHICAL VOLUME SAMPLING #####################
#     raw = run_network(pts)

    # Query output of coarse-sampled points, and reveices weight of each point
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest) # raw2outputs : includes volume rendering
    
    if N_importance > 0: # Number of points to sample in single ray, default : 128

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # find mid value of z_vals
        
        # z_samples : Sampled points by fine(hierarchical volume) sampling
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)

        # In this case, network_query_fn refers to fine network
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        
        # recap paper) 
        # Final rendering color is calculated by output of FINE NETWORK
        # So, result of fine network overwrite coarse network's output

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret
    ########################################################################


def train(nerf_wrapper, basedir_save, basedir, datadir, i_save, N_iters,
          N_rand=32*32*4, chunk=1024*32, netchunk=1024*64, no_batching=False, 
          N_samples=64, N_importance=0, perturb=1., use_viewdirs=False, i_embed=0, 
          multires=10, multires_views=4, raw_noise_std=0., 
          render_only=False, render_test=False, render_factor=0,
          precrop_iters=0, precrop_frac=0.5, dataset_type='llff', testskip=8,
          shape='greek', white_bkgd=False, half_res=False, factor=8,
          no_ndc=False, lindisp=False, spherify=False, llffhold=8):
    
    # # USE ONLY LSA PARAMETER TO OPTIMIZE
    
    # params = [param for name, param in nerf_wrapper.model.named_parameters() if '.weight_scaling' in name]
    # nerf_wrapper.tuning_optimizer = torch.optim.Adam(params, lr=0.1)
    optimizer = nerf_wrapper.tuning_optimizer
    optimizer = torch.optim.Adam(nerf_wrapper.model.parameters(), lr=1)


    # Original NeRF training code include precrop process
    # But it is depricated when tuning LSA parameter of pretrained model
    precrop_iters = 0

    print('Device to run NeRF-Training:', DEVICE)
    
    # Load data
    K = None

    if dataset_type == 'llff':
        
        # hwf : height, width, focal length of image
        # poses : 3 x 4 sized extrinsic matrix, pose to render when train
        # render_poses : extrinsic matrix, pose to render when inference
        # near, far : depth between camera and object. i.e. min-max depth to do sampling
        # i_train, i_val, i_test : index list of train/validation/test
        
        images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=spherify)
        
        hwf = poses[0,:3,-1] # hwf : height, width, focal length of image
        poses = poses[:,:3,:4] # 3 x 4 sized extrinsic matrix, pose to render when training
        
        print('Loaded llff')
        
        if not isinstance(i_test, list):
            i_test = [i_test]

        if llffhold > 0:
            # print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(images.shape[0])[::llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        # print('DEFINING BOUNDS')
        if no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        # print('NEAR FAR', near, far)
        
        
    elif dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(datadir, half_res, testskip)
        print('Loaded blender')
        
        i_train, i_val, i_test = i_split
        
        near = 2.
        far = 6.

        if white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(datadir, half_res, testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=shape,
                                                                 basedir=datadir,
                                                                 testskip=testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    # K : Intrinsic Parameter
    # * Intrinsic Parameter : Matrix which maps normalized plane to image plane

    if K is None:
        # Normalized Plane : (0, 0) -> Center of Image
        # Image Plane : (0, 0) -> Top-Left of Image
        # Thus, both scaling and transitions are required
        # Scaling : x focal length_x, y (+ assume f_x, y are same)/ Transition : W,H x 0.5
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if render_test:
        render_poses = np.array(poses[i_test])


    ####################### CUSTOM MODIFIED #########################
    # Original : Call create_nerf() and receives nerf model (from scratch or pretrained checkpoint)
    # Modified : Send model 
    
    
    render_kwargs_train, render_kwargs_test = create_nerf(
        nerf_wrapper = nerf_wrapper,
        multires = multires,
        i_embed = i_embed,
        use_viewdirs = use_viewdirs,
        multires_views = multires_views,
        netchunk = netchunk,
        basedir = basedir,
        perturb = perturb,
        N_importance = N_importance,
        N_samples = N_samples,
        white_bkgd = white_bkgd,
        raw_noise_std = raw_noise_std,
        dataset_type = dataset_type,
        no_ndc = no_ndc,
        lindisp = lindisp
    )

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(DEVICE)

    # Short circuit if only rendering out from trained model
    if render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir_save, 'renderonly_{}_{:06d}'.format('test' if render_test else 'path'))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = N_rand
    use_batching = not no_batching
    
    if use_batching:
        print('Use Batching')
        # For random ray batching
        # print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        # print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        # print('shuffle rays')
        np.random.shuffle(rays_rgb)

        # print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(DEVICE)
        
    poses = torch.Tensor(poses).to(DEVICE)
    
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(DEVICE)
    
    loss_list = np.array([])
    psnr_list = np.array([])
    
    
    for i in range(0, N_iters):
    # for i in tqdm(range(0, N_iters)):
        
        nerf_wrapper.global_step += 1

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(DEVICE)
            pose = poses[img_i, :3,:4]
            
            K = torch.Tensor(K).to(DEVICE)

            if N_rand is not None:
                
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                
                if i < precrop_iters:
                    dH = int(H//2 * precrop_frac)
                    dW = int(W//2 * precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    
                    if i == 0:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                
                
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train)
        
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        
        
        psnr_list = np.append(psnr_list, psnr.item())
        loss_list = np.append(loss_list, loss.item())
        
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)



        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
            
            
        # Rest Is Logging
        # i_save == 0 --> Train Without Save, Only Returns Final Model, and Performance Log (.txt)
        # i_save != 0 --> Save Also Inferenced Image and .mp4 Video At First Step, and Each i_save Iteration
        
        if (i_save != 0) and (nerf_wrapper.global_step == 1 or nerf_wrapper.global_step % i_save == 0):
            
            print(f"Save Status & Performance of Iteration {nerf_wrapper.global_step}")
            
            path = os.path.join(basedir_save, 'reconstructed', f'ckpt_step{nerf_wrapper.global_step}.pt')
            
            # Save Checkpoint
            torch.save({
                'global_step': nerf_wrapper.global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'tuning_optimizer_state_dict': nerf_wrapper.tuning_optimizer.state_dict(),
            }, path)
                        
            # Render and Save Images of Inferenced Sequence
            testsavedir = os.path.join(basedir_save, f'testset_step{nerf_wrapper.global_step}')
            os.makedirs(testsavedir, exist_ok=True)
            
            with torch.no_grad():
                rgbs, _ = render_path(render_poses, hwf, K, chunk, render_kwargs_test, gt_imgs=images[i_test], savedir = testsavedir)

            # Create Video from Saved Images
            moviebase = os.path.join(basedir_save, 'movies')
            os.makedirs(moviebase, exist_ok=True)

            imageio.mimwrite(moviebase + f"/step{nerf_wrapper.global_step}_rgb.mp4", to8b(rgbs), fps=30, quality=8)

            print('Saving Process Done')
        
        append_results_to_file(basedir = basedir_save, psnr_value = psnr.item(), loss_value = loss.item())
    
    print(f"Epoch Done. \n\n")
    
    return psnr_list.mean(), loss_list.mean()

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
